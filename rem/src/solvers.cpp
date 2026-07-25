#include "detail.hpp"

#include <numeric>

#if defined(REM_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace rem::detail {
namespace {

constexpr double kBreakdownFactor = 64.0;

#if defined(REM_USE_ACCELERATE)
using LapackInt = __LAPACK_int;
#else
using LapackInt = int;
extern "C" void dgesv_(
    const LapackInt*,
    const LapackInt*,
    double*,
    const LapackInt*,
    LapackInt*,
    double*,
    const LapackInt*,
    LapackInt*);
#endif

class VectorKernels {
public:
    VectorKernels(ThreadPool& pool, std::size_t size)
        : pool_(pool), size_(size), partial_(pool.size(), 0.0),
          parallel_(size >= 4096 && pool.size() > 1) {}

    double dot(const double* lhs, const double* rhs) {
        if (!parallel_) {
            double sum = 0.0;
            for (std::size_t i = 0; i < size_; ++i) {
                sum += lhs[i] * rhs[i];
            }
            return sum;
        }
        std::fill(partial_.begin(), partial_.end(), 0.0);
        pool_.parallel_for(size_, [&](std::size_t begin, std::size_t end, std::uint32_t tid) {
            double sum = 0.0;
            for (std::size_t i = begin; i < end; ++i) {
                sum += lhs[i] * rhs[i];
            }
            partial_[tid] = sum;
        });
        return std::accumulate(partial_.begin(), partial_.end(), 0.0);
    }

    double norm(const double* values) { return std::sqrt(std::max(0.0, dot(values, values))); }

    void axpy(double alpha, const double* x, double* y) {
        apply([=](std::size_t i) { y[i] += alpha * x[i]; });
    }

    void scale(double alpha, double* x) {
        apply([=](std::size_t i) { x[i] *= alpha; });
    }

    template <class Function>
    void apply(Function&& function) {
        if (!parallel_) {
            for (std::size_t i = 0; i < size_; ++i) {
                function(i);
            }
            return;
        }
        pool_.parallel_for(size_, [&](std::size_t begin, std::size_t end, std::uint32_t) {
            for (std::size_t i = begin; i < end; ++i) {
                function(i);
            }
        });
    }

private:
    ThreadPool& pool_;
    std::size_t size_;
    std::vector<double> partial_;
    bool parallel_;
};

void apply_left_preconditioned(
    const AssignmentOperator& op,
    std::span<const double> input,
    std::span<double> output,
    VectorKernels& kernels) {
    op.matvec(input, output);
    kernels.apply([&](std::size_t i) { output[i] *= op.jacobi_inverse[i]; });
}

}  // namespace

double l2_norm(std::span<const double> values) noexcept {
    double scale = 0.0;
    double sum_squares = 1.0;
    for (const double value : values) {
        const double absolute = std::abs(value);
        if (absolute == 0.0) {
            continue;
        }
        if (scale < absolute) {
            const double ratio = scale / absolute;
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = absolute;
        } else {
            const double ratio = absolute / scale;
            sum_squares += ratio * ratio;
        }
    }
    return scale == 0.0 ? 0.0 : scale * std::sqrt(sum_squares);
}

double true_residual(
    const AssignmentOperator& op,
    std::span<const double> x,
    std::span<const double> rhs,
    std::vector<double>& workspace) {
    workspace.resize(op.size());
    op.matvec(x, workspace);
    for (std::size_t i = 0; i < workspace.size(); ++i) {
        workspace[i] = rhs[i] - workspace[i];
    }
    return l2_norm(workspace);
}

SolverResult solve_gmres(
    const AssignmentOperator& op,
    std::span<const double> rhs,
    const Options& options) {
    const std::size_t n = op.size();
    const std::size_t restart =
        std::max<std::size_t>(1, std::min<std::size_t>(options.gmres_restart, n));
    SolverResult result;
    result.x.assign(n, 0.0);
    AlignedDoubles basis((restart + 1U) * n, 0.0);
    AlignedDoubles hessenberg((restart + 1U) * restart, 0.0);
    AlignedDoubles cosines(restart, 0.0);
    AlignedDoubles sines(restart, 0.0);
    AlignedDoubles g(restart + 1U, 0.0);
    AlignedDoubles y(restart, 0.0);
    AlignedDoubles orthogonalization_correction(restart, 0.0);
    AlignedDoubles work(n, 0.0);
    AlignedDoubles ax(n, 0.0);
    AlignedDoubles residual(n, 0.0);
    AlignedDoubles preconditioned_rhs(n, 0.0);
    std::vector<double> true_workspace(n);
    VectorKernels kernels(*op.pool, n);

    kernels.apply([&](std::size_t i) {
        preconditioned_rhs[i] = rhs[i] * op.jacobi_inverse[i];
    });
    const double rhs_norm = l2_norm(rhs);
    const double preconditioned_rhs_norm = kernels.norm(preconditioned_rhs.data());
    const double true_tolerance =
        options.absolute_tolerance + options.relative_tolerance * rhs_norm;
    const double krylov_tolerance =
        options.absolute_tolerance +
        options.relative_tolerance * preconditioned_rhs_norm;
    const double breakdown =
        kBreakdownFactor * std::numeric_limits<double>::epsilon();

    auto update_solution = [&](std::size_t used) -> bool {
        std::fill(y.begin(), y.end(), 0.0);
        for (std::size_t reverse = used; reverse-- > 0;) {
            double sum = g[reverse];
            for (std::size_t column = reverse + 1U; column < used; ++column) {
                sum -= hessenberg[column * (restart + 1U) + reverse] * y[column];
            }
            const double diagonal =
                hessenberg[reverse * (restart + 1U) + reverse];
            if (std::abs(diagonal) <= breakdown || !std::isfinite(diagonal)) {
                return false;
            }
            y[reverse] = sum / diagonal;
        }
        for (std::size_t column = 0; column < used; ++column) {
            kernels.axpy(y[column], basis.data() + column * n, result.x.data());
        }
        return true;
    };

    while (result.iterations < options.max_iterations) {
        op.matvec(result.x, ax);
        kernels.apply([&](std::size_t i) {
            residual[i] = (rhs[i] - ax[i]) * op.jacobi_inverse[i];
        });
        const double beta = kernels.norm(residual.data());
        result.residual = true_residual(op, result.x, rhs, true_workspace);
        result.relative_residual =
            rhs_norm == 0.0 ? result.residual : result.residual / rhs_norm;
        if (result.residual <= true_tolerance) {
            result.converged = true;
            return result;
        }
        if (!std::isfinite(beta) || beta <= breakdown) {
            result.warning = "GMRES preconditioned residual broke down";
            return result;
        }

        std::fill(hessenberg.begin(), hessenberg.end(), 0.0);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        std::copy(residual.begin(), residual.end(), basis.begin());
        kernels.scale(1.0 / beta, basis.data());

        std::size_t used = 0;
        bool early_stop = false;
        bool arnoldi_breakdown = false;
        for (std::size_t column = 0;
             column < restart && result.iterations < options.max_iterations;
             ++column) {
            apply_left_preconditioned(
                op, std::span<const double>(basis.data() + column * n, n), work,
                kernels);
            const double before_orthogonalization = kernels.norm(work.data());
            double* hessenberg_column =
                hessenberg.data() + column * (restart + 1U);
            const bool use_classical =
                options.orthogonalization ==
                    Orthogonalization::ClassicalGramSchmidt
#if defined(REM_USE_ACCELERATE)
                || (options.orthogonalization == Orthogonalization::Auto &&
                    n >= 4096)
#endif
                ;
            if (use_classical) {
#if defined(REM_USE_ACCELERATE)
                if (n <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    const int rows = static_cast<int>(n);
                    const int columns = static_cast<int>(column + 1U);
                    cblas_dgemv(
                        CblasColMajor, CblasTrans, rows, columns, 1.0, basis.data(),
                        rows, work.data(), 1, 0.0, hessenberg_column, 1);
                    cblas_dgemv(
                        CblasColMajor, CblasNoTrans, rows, columns, -1.0,
                        basis.data(), rows, hessenberg_column, 1, 1.0, work.data(),
                        1);
                    cblas_dgemv(
                        CblasColMajor, CblasTrans, rows, columns, 1.0, basis.data(),
                        rows, work.data(), 1, 0.0,
                        orthogonalization_correction.data(), 1);
                    cblas_dgemv(
                        CblasColMajor, CblasNoTrans, rows, columns, -1.0,
                        basis.data(), rows, orthogonalization_correction.data(), 1,
                        1.0, work.data(), 1);
                    for (std::size_t row = 0; row <= column; ++row) {
                        hessenberg_column[row] +=
                            orthogonalization_correction[row];
                    }
                } else
#endif
                {
                    for (std::size_t row = 0; row <= column; ++row) {
                        hessenberg_column[row] =
                            kernels.dot(work.data(), basis.data() + row * n);
                    }
                    for (std::size_t row = 0; row <= column; ++row) {
                        kernels.axpy(
                            -hessenberg_column[row], basis.data() + row * n,
                            work.data());
                    }
                    for (std::size_t row = 0; row <= column; ++row) {
                        orthogonalization_correction[row] =
                            kernels.dot(work.data(), basis.data() + row * n);
                    }
                    for (std::size_t row = 0; row <= column; ++row) {
                        hessenberg_column[row] +=
                            orthogonalization_correction[row];
                        kernels.axpy(
                            -orthogonalization_correction[row],
                            basis.data() + row * n, work.data());
                    }
                }
            } else {
                for (std::size_t row = 0; row <= column; ++row) {
                    const double coefficient =
                        kernels.dot(work.data(), basis.data() + row * n);
                    hessenberg_column[row] = coefficient;
                    kernels.axpy(
                        -coefficient, basis.data() + row * n, work.data());
                }
            }
            double next_norm = kernels.norm(work.data());
            // Selective reorthogonalization retains MGS speed for the common case.
            if (!use_classical &&
                next_norm < 0.5 * before_orthogonalization) {
                for (std::size_t row = 0; row <= column; ++row) {
                    const double correction =
                        kernels.dot(work.data(), basis.data() + row * n);
                    hessenberg_column[row] += correction;
                    kernels.axpy(
                        -correction, basis.data() + row * n, work.data());
                }
                next_norm = kernels.norm(work.data());
            }
            hessenberg[column * (restart + 1U) + column + 1U] = next_norm;
            if (next_norm > breakdown * std::max(1.0, before_orthogonalization)) {
                std::copy_n(
                    work.begin(), static_cast<std::ptrdiff_t>(n),
                    basis.begin() +
                        static_cast<std::ptrdiff_t>((column + 1U) * n));
                kernels.scale(1.0 / next_norm, basis.data() + (column + 1U) * n);
            } else {
                arnoldi_breakdown = true;
            }

            for (std::size_t row = 0; row < column; ++row) {
                double& upper = hessenberg[column * (restart + 1U) + row];
                double& lower = hessenberg[column * (restart + 1U) + row + 1U];
                const double rotated = cosines[row] * upper + sines[row] * lower;
                lower = -sines[row] * upper + cosines[row] * lower;
                upper = rotated;
            }
            double& diagonal =
                hessenberg[column * (restart + 1U) + column];
            double& subdiagonal =
                hessenberg[column * (restart + 1U) + column + 1U];
            const double magnitude = std::hypot(diagonal, subdiagonal);
            if (magnitude <= breakdown || !std::isfinite(magnitude)) {
                cosines[column] = 1.0;
                sines[column] = 0.0;
            } else {
                cosines[column] = diagonal / magnitude;
                sines[column] = subdiagonal / magnitude;
            }
            diagonal = cosines[column] * diagonal + sines[column] * subdiagonal;
            subdiagonal = 0.0;
            g[column + 1U] = -sines[column] * g[column];
            g[column] = cosines[column] * g[column];
            ++result.iterations;
            used = column + 1U;
            if (std::abs(g[column + 1U]) <= krylov_tolerance ||
                arnoldi_breakdown) {
                early_stop = true;
                break;
            }
        }

        if (!update_solution(used)) {
            result.warning = "GMRES triangular solve broke down";
            break;
        }
        result.residual = true_residual(op, result.x, rhs, true_workspace);
        result.relative_residual =
            rhs_norm == 0.0 ? result.residual : result.residual / rhs_norm;
        if (result.residual <= true_tolerance) {
            result.converged = true;
            return result;
        }
        if (arnoldi_breakdown) {
            result.warning = "GMRES Arnoldi process broke down before convergence";
            return result;
        }
        (void)early_stop;
    }
    result.residual = true_residual(op, result.x, rhs, true_workspace);
    result.relative_residual =
        rhs_norm == 0.0 ? result.residual : result.residual / rhs_norm;
    result.converged = result.residual <= true_tolerance;
    if (!result.converged && result.warning.empty()) {
        result.warning = "GMRES reached the maximum iteration count";
    }
    return result;
}

SolverResult solve_bicgstab(
    const AssignmentOperator& op,
    std::span<const double> rhs,
    const Options& options) {
    const std::size_t n = op.size();
    SolverResult result;
    result.x.assign(n, 0.0);
    AlignedDoubles r(n), r_hat(n), p(n, 0.0), v(n, 0.0), s(n), t(n), ax(n);
    std::vector<double> true_workspace(n);
    VectorKernels kernels(*op.pool, n);
    kernels.apply([&](std::size_t i) {
        r[i] = rhs[i] * op.jacobi_inverse[i];
        r_hat[i] = r[i];
    });
    const double rhs_norm = l2_norm(rhs);
    const double tolerance =
        options.absolute_tolerance + options.relative_tolerance * rhs_norm;
    const double breakdown =
        kBreakdownFactor * std::numeric_limits<double>::epsilon();
    auto near_breakdown = [breakdown](double value, double scale) {
        const double floor = std::numeric_limits<double>::min() / breakdown;
        return std::abs(value) <= breakdown * std::max(scale, floor);
    };
    double rho_previous = 1.0;
    double alpha = 1.0;
    double omega = 1.0;

    for (std::uint32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        const double rho = kernels.dot(r_hat.data(), r.data());
        const double rho_scale =
            kernels.norm(r_hat.data()) * kernels.norm(r.data());
        if (!std::isfinite(rho) || near_breakdown(rho, rho_scale)) {
            result.warning = "BiCGSTAB rho breakdown";
            break;
        }
        const double beta = (rho / rho_previous) * (alpha / omega);
        kernels.apply([&](std::size_t i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        });
        apply_left_preconditioned(op, p, v, kernels);
        const double denominator = kernels.dot(r_hat.data(), v.data());
        const double denominator_scale =
            kernels.norm(r_hat.data()) * kernels.norm(v.data());
        if (!std::isfinite(denominator) ||
            near_breakdown(denominator, denominator_scale)) {
            result.warning = "BiCGSTAB alpha breakdown";
            break;
        }
        alpha = rho / denominator;
        kernels.apply([&](std::size_t i) { s[i] = r[i] - alpha * v[i]; });
        if (kernels.norm(s.data()) <= tolerance) {
            kernels.axpy(alpha, p.data(), result.x.data());
            result.iterations = iteration + 1U;
            result.residual = true_residual(op, result.x, rhs, true_workspace);
            result.relative_residual =
                rhs_norm == 0.0 ? result.residual : result.residual / rhs_norm;
            result.converged = result.residual <= tolerance;
            if (!result.converged) {
                result.warning = "BiCGSTAB preconditioned residual was misleading";
            }
            return result;
        }
        apply_left_preconditioned(op, s, t, kernels);
        const double tt = kernels.dot(t.data(), t.data());
        if (!std::isfinite(tt) || near_breakdown(tt, tt)) {
            result.warning = "BiCGSTAB omega denominator breakdown";
            break;
        }
        omega = kernels.dot(t.data(), s.data()) / tt;
        if (!std::isfinite(omega) || std::abs(omega) <= breakdown) {
            result.warning = "BiCGSTAB omega breakdown";
            break;
        }
        kernels.apply([&](std::size_t i) {
            result.x[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        });
        result.iterations = iteration + 1U;
        if (kernels.norm(r.data()) <= tolerance) {
            result.residual = true_residual(op, result.x, rhs, true_workspace);
            result.relative_residual =
                rhs_norm == 0.0 ? result.residual : result.residual / rhs_norm;
            if (result.residual <= tolerance) {
                result.converged = true;
                return result;
            }
        }
        rho_previous = rho;
    }
    result.residual = true_residual(op, result.x, rhs, true_workspace);
    result.relative_residual =
        rhs_norm == 0.0 ? result.residual : result.residual / rhs_norm;
    result.converged = result.residual <= tolerance;
    if (!result.converged && result.warning.empty()) {
        result.warning = "BiCGSTAB reached the maximum iteration count";
    }
    return result;
}

std::vector<double> solve_dense_many(
    AlignedDoubles matrix_column_major,
    std::span<const double> rhs_major,
    std::size_t right_hand_sides,
    std::size_t dimension,
    std::string& error) {
    if (dimension >
            static_cast<std::size_t>(std::numeric_limits<LapackInt>::max()) ||
        right_hand_sides >
            static_cast<std::size_t>(std::numeric_limits<LapackInt>::max())) {
        error = "LAPACK integer range exceeded";
        return {};
    }
    AlignedDoubles solution(rhs_major.begin(), rhs_major.end());
    std::vector<LapackInt> pivots(dimension);
    const LapackInt n = static_cast<LapackInt>(dimension);
    const LapackInt nrhs = static_cast<LapackInt>(right_hand_sides);
    const LapackInt leading_dimension = n;
    LapackInt info = 0;
    dgesv_(
        &n, &nrhs, matrix_column_major.data(), &leading_dimension, pivots.data(),
        solution.data(), &leading_dimension, &info);
    if (info < 0) {
        error = "LAPACK dgesv received an invalid argument";
        return {};
    }
    if (info > 0) {
        error = "LAPACK dgesv found a singular reduced assignment matrix";
        return {};
    }
    return {solution.begin(), solution.end()};
}

}  // namespace rem::detail
