#include "rem/api.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>

namespace py = pybind11;

namespace {

void require_c_contiguous(const py::array& array, const char* name) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
}

template <class T>
void require_dtype(const py::array& array, const char* name) {
    if (!array.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(
            std::string(name) + " has the wrong dtype; hidden force-casts are disabled");
    }
}

rem::Backend parse_backend(const std::string& value) {
    if (value == "auto") return rem::Backend::Auto;
    if (value == "dense") return rem::Backend::Dense;
    if (value == "sparse" || value == "explicit_sparse")
        return rem::Backend::ExplicitSparse;
    if (value == "matrix_free") return rem::Backend::MatrixFree;
    throw py::value_error("invalid backend: " + value);
}

rem::Solver parse_solver(const std::string& value) {
    if (value == "auto") return rem::Solver::Auto;
    if (value == "direct") return rem::Solver::Direct;
    if (value == "gmres") return rem::Solver::GMRES;
    if (value == "bicgstab") return rem::Solver::BiCGSTAB;
    throw py::value_error("invalid solver: " + value);
}

rem::GraphStrategy parse_graph(const std::string& value) {
    if (value == "auto") return rem::GraphStrategy::Auto;
    if (value == "hamming_ball") return rem::GraphStrategy::HammingBall;
    if (value == "pairwise" || value == "blocked_pairwise")
        return rem::GraphStrategy::BlockedPairwise;
    throw py::value_error("invalid graph_strategy: " + value);
}

rem::SparseStorage parse_sparse_storage(const std::string& value) {
    if (value == "auto") return rem::SparseStorage::Auto;
    if (value == "topology" || value == "topology_only")
        return rem::SparseStorage::TopologyOnly;
    if (value == "valued") return rem::SparseStorage::Valued;
    throw py::value_error("invalid sparse_storage: " + value);
}

rem::Orthogonalization parse_orthogonalization(const std::string& value) {
    if (value == "auto") return rem::Orthogonalization::Auto;
    if (value == "mgs" || value == "modified_gram_schmidt")
        return rem::Orthogonalization::ModifiedGramSchmidt;
    if (value == "cgs" || value == "classical_gram_schmidt")
        return rem::Orthogonalization::ClassicalGramSchmidt;
    throw py::value_error("invalid orthogonalization: " + value);
}

rem::Options parse_options(const py::dict& values) {
    rem::Options options;
    auto assign = [&](const char* key, auto& target) {
        if (values.contains(key)) target = values[key].cast<std::decay_t<decltype(target)>>();
    };
    if (values.contains("backend"))
        options.backend = parse_backend(values["backend"].cast<std::string>());
    if (values.contains("solver"))
        options.solver = parse_solver(values["solver"].cast<std::string>());
    if (values.contains("graph_strategy"))
        options.graph_strategy =
            parse_graph(values["graph_strategy"].cast<std::string>());
    if (values.contains("sparse_storage"))
        options.sparse_storage =
            parse_sparse_storage(values["sparse_storage"].cast<std::string>());
    if (values.contains("orthogonalization"))
        options.orthogonalization =
            parse_orthogonalization(values["orthogonalization"].cast<std::string>());
    assign("max_hamming_distance", options.max_hamming_distance);
    assign("relative_tolerance", options.relative_tolerance);
    assign("absolute_tolerance", options.absolute_tolerance);
    assign("gmres_restart", options.gmres_restart);
    assign("max_iterations", options.max_iterations);
    assign("threads", options.threads);
    assign("memory_budget_bytes", options.memory_budget_bytes);
    assign("project_to_probability_simplex", options.project_to_probability_simplex);
    assign("cache_topology", options.cache_topology);
    assign("cache_values", options.cache_values);
    assign("near_zero_threshold", options.near_zero_threshold);
    return options;
}

py::dict diagnostics_to_dict(const rem::Diagnostics& diagnostics) {
    py::dict output;
    output["selected_backend"] = rem::to_string(diagnostics.selected_backend);
    output["selected_solver"] = rem::to_string(diagnostics.selected_solver);
    output["selected_graph_strategy"] =
        rem::to_string(diagnostics.selected_graph_strategy);
    output["selected_sparse_storage"] =
        rem::to_string(diagnostics.selected_sparse_storage);
    output["selected_orthogonalization"] =
        rem::to_string(diagnostics.selected_orthogonalization);
    output["input_states"] = diagnostics.input_states;
    output["unique_states"] = diagnostics.unique_states;
    output["retained_edges"] = diagnostics.retained_edges;
    output["hamming_ball_size"] = diagnostics.hamming_ball_size;
    output["estimated_memory_bytes"] = diagnostics.estimated_memory_bytes;
    output["peak_working_memory_bytes"] = diagnostics.peak_working_memory_bytes;
    output["iterations"] = diagnostics.iterations;
    output["threads"] = diagnostics.threads;
    output["sampled_density"] = diagnostics.sampled_density;
    output["residual_norm"] = diagnostics.residual_norm;
    output["relative_residual"] = diagnostics.relative_residual;
    output["setup_seconds"] = diagnostics.setup_seconds;
    output["solve_seconds"] = diagnostics.solve_seconds;
    output["projection_seconds"] = diagnostics.projection_seconds;
    output["converged"] = diagnostics.converged;
    output["used_safe_element_fallback"] = diagnostics.used_safe_element_fallback;
    output["jacobi_had_zero_diagonal"] = diagnostics.jacobi_had_zero_diagonal;
    output["backend_reason"] = diagnostics.backend_reason;
    output["threading_backend"] = diagnostics.threading_backend;
    output["warning"] = diagnostics.warning;
    return output;
}

template <class T>
py::array_t<T> vector_array(const std::vector<T>& values) {
    py::array_t<T> output(values.size());
    if (!values.empty()) {
        std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return output;
}

py::dict result_to_dict(const rem::Result& result, std::uint32_t qubits) {
    py::dict output;
    output["quasi_probabilities"] = vector_array(result.quasi_probabilities);
    if (result.projected_probabilities) {
        output["projected_probabilities"] =
            vector_array(*result.projected_probabilities);
    } else {
        output["projected_probabilities"] = py::none();
    }
    const std::size_t words = (static_cast<std::size_t>(qubits) + 63U) / 64U;
    py::array_t<std::uint64_t> states(
        {static_cast<py::ssize_t>(result.unique_packed_states.size() / words),
         static_cast<py::ssize_t>(words)});
    if (!result.unique_packed_states.empty()) {
        std::memcpy(
            states.mutable_data(), result.unique_packed_states.data(),
            result.unique_packed_states.size() * sizeof(std::uint64_t));
    }
    output["unique_packed_states"] = std::move(states);
    output["original_to_unique"] = vector_array(result.original_to_unique);
    output["diagnostics"] = diagnostics_to_dict(result.diagnostics);
    return output;
}

std::tuple<std::span<const std::uint64_t>, std::size_t> validate_states(
    const py::array& states, std::uint32_t qubits, std::size_t expected_count) {
    require_dtype<std::uint64_t>(states, "packed_bitstrings");
    require_c_contiguous(states, "packed_bitstrings");
    const std::size_t words = (static_cast<std::size_t>(qubits) + 63U) / 64U;
    std::size_t count = 0;
    if (states.ndim() == 1 && words == 1) {
        count = static_cast<std::size_t>(states.shape(0));
    } else if (
        states.ndim() == 2 &&
        static_cast<std::size_t>(states.shape(1)) == words) {
        count = static_cast<std::size_t>(states.shape(0));
    } else {
        throw py::value_error(
            "packed_bitstrings must have shape [K] for Q<=64 or [K,ceil(Q/64)]");
    }
    if (count != expected_count) {
        throw py::value_error("packed_bitstrings and raw_probabilities disagree on K");
    }
    return {
        std::span<const std::uint64_t>(
            static_cast<const std::uint64_t*>(states.data()), count * words),
        count};
}

void validate_calibration(const py::array& cals, std::uint32_t qubits) {
    require_dtype<double>(cals, "confusion_matrices");
    require_c_contiguous(cals, "confusion_matrices");
    if (cals.ndim() != 3 || cals.shape(0) != qubits || cals.shape(1) != 2 ||
        cals.shape(2) != 2) {
        throw py::value_error("confusion_matrices must have shape [Q,2,2]");
    }
}

}  // namespace

PYBIND11_MODULE(rem_cpp, module) {
    module.doc() = "Packed C++ sparse readout-error mitigation";
    module.def(
        "mitigate_packed",
        [](const py::array& states,
           const py::array& probabilities,
           const py::array& cals,
           std::uint32_t qubits,
           const py::dict& option_values) {
            require_dtype<double>(probabilities, "raw_probabilities");
            require_c_contiguous(probabilities, "raw_probabilities");
            if (probabilities.ndim() != 1) {
                throw py::value_error("raw_probabilities must be one-dimensional");
            }
            const std::size_t count =
                static_cast<std::size_t>(probabilities.shape(0));
            auto [packed, packed_count] = validate_states(states, qubits, count);
            validate_calibration(cals, qubits);
            const auto probability_view = std::span<const double>(
                static_cast<const double*>(probabilities.data()), count);
            const auto calibration_view = std::span<const double>(
                static_cast<const double*>(cals.data()),
                static_cast<std::size_t>(qubits) * 4U);
            const rem::Options options = parse_options(option_values);
            rem::Result result;
            {
                py::gil_scoped_release release;
                result = rem::mitigate(
                    packed, packed_count, qubits, probability_view, calibration_view,
                    options);
            }
            return result_to_dict(result, qubits);
        },
        py::arg("packed_bitstrings"), py::arg("raw_probabilities"),
        py::arg("confusion_matrices"), py::arg("measured_qubits"),
        py::arg("options") = py::dict());

    module.def(
        "mitigate_packed_many",
        [](const py::array& states,
           const py::array& probabilities,
           const py::array& cals,
           std::uint32_t qubits,
           const py::dict& option_values) {
            require_dtype<double>(probabilities, "raw_probabilities");
            require_c_contiguous(probabilities, "raw_probabilities");
            if (probabilities.ndim() != 2) {
                throw py::value_error(
                    "batched raw_probabilities must have shape [RHS,K]");
            }
            const std::size_t right_hand_sides =
                static_cast<std::size_t>(probabilities.shape(0));
            const std::size_t count =
                static_cast<std::size_t>(probabilities.shape(1));
            auto [packed, packed_count] = validate_states(states, qubits, count);
            validate_calibration(cals, qubits);
            const auto probability_view = std::span<const double>(
                static_cast<const double*>(probabilities.data()),
                right_hand_sides * count);
            const auto calibration_view = std::span<const double>(
                static_cast<const double*>(cals.data()),
                static_cast<std::size_t>(qubits) * 4U);
            const rem::Options options = parse_options(option_values);
            std::vector<rem::Result> results;
            {
                py::gil_scoped_release release;
                results = rem::mitigate_many(
                    packed, packed_count, qubits, probability_view, right_hand_sides,
                    calibration_view, options);
            }
            py::list output;
            for (const auto& result : results) {
                output.append(result_to_dict(result, qubits));
            }
            return output;
        },
        py::arg("packed_bitstrings"), py::arg("raw_probabilities"),
        py::arg("confusion_matrices"), py::arg("measured_qubits"),
        py::arg("options") = py::dict());

    module.def("project_probability_simplex", [](const py::array& values) {
        require_dtype<double>(values, "values");
        require_c_contiguous(values, "values");
        if (values.ndim() != 1) {
            throw py::value_error("values must be one-dimensional");
        }
        const auto view = std::span<const double>(
            static_cast<const double*>(values.data()),
            static_cast<std::size_t>(values.shape(0)));
        return vector_array(rem::project_probability_simplex(view));
    });
}
