from importlib import import_module, util as importlib_util
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from scipy.sparse import csr_matrix, spmatrix
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial.distance import pdist, squareform

MatrixFormat = Literal["dense", "sparse", "auto"]
BackendImpl = Literal["python", "cpp", "auto"]


def _execution_seconds(result: Any) -> Optional[float]:
    """Return backend-reported execution time when Qiskit exposes it."""
    value = getattr(result, "time_taken", None)
    if isinstance(value, (int, float)) and np.isfinite(value) and value >= 0:
        return float(value)
    return None


def _load_cpp_backend():
    """Load an installed extension or a local CMake development build."""
    try:
        return import_module("rem_cpp")
    except ImportError as initial_error:
        project_root = Path(__file__).resolve().parent
        candidates = sorted(
            project_root.glob("rem/build*/rem_cpp*.so"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            spec = importlib_util.spec_from_file_location("rem_cpp", candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        raise initial_error


def _pack_bitstrings(bitstrings: List[str], qubits: int) -> np.ndarray:
    """Pack displayed MSB-left bitstrings into state-major LSB-first words."""
    words = (qubits + 63) // 64
    if words == 1:
        return np.ascontiguousarray(
            np.fromiter((int(state, 2) for state in bitstrings), dtype=np.uint64)
        )
    packed = np.zeros((len(bitstrings), words), dtype=np.uint64)
    for row, state in enumerate(bitstrings):
        for q, bit in enumerate(reversed(state)):
            if bit == "1":
                packed[row, q // 64] |= np.uint64(1) << np.uint64(q % 64)
    return packed


def _mitigate_probs_cpp(
    raw_probs: Dict[str, float],
    measured_qubits: List[int],
    confusion_matrices_all: np.ndarray,
    matrix_format: MatrixFormat,
    solver_tol: float,
    reduce_kwargs: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    backend = _load_cpp_backend()
    bitstrings = list(raw_probs)
    qubits = len(measured_qubits)
    if any(len(state) != qubits for state in bitstrings):
        raise ValueError("bitstring length must match the number of measured qubits")
    packed = _pack_bitstrings(bitstrings, qubits)
    raw = np.ascontiguousarray(list(raw_probs.values()), dtype=np.float64)
    calibrations = np.ascontiguousarray(
        confusion_matrices_all[np.asarray(measured_qubits, dtype=int)],
        dtype=np.float64,
    )
    options = {
        "backend": {
            "dense": "dense",
            "sparse": "explicit_sparse",
            "auto": "auto",
        }[matrix_format],
        "solver": "direct" if matrix_format == "dense" else "auto",
        "max_hamming_distance": int(reduce_kwargs.get("d", 3)),
        "relative_tolerance": float(solver_tol),
        "absolute_tolerance": 0.0,
        "project_to_probability_simplex": False,
    }
    for name in (
        "gmres_restart",
        "max_iterations",
        "threads",
        "memory_budget_bytes",
        "cache_topology",
        "cache_values",
        "graph_strategy",
        "sparse_storage",
        "orthogonalization",
        "near_zero_threshold",
    ):
        if name in reduce_kwargs:
            options[name] = reduce_kwargs[name]
    output = backend.mitigate_packed(packed, raw, calibrations, qubits, options)
    unique_mapping = np.asarray(output["original_to_unique"])
    quasi_vector = np.asarray(output["quasi_probabilities"])
    quasi = {
        state: float(quasi_vector[int(unique_mapping[index])])
        for index, state in enumerate(bitstrings)
    }
    diagnostics = dict(output["diagnostics"])
    diagnostics["backend_impl"] = "cpp"
    return quasi, diagnostics


def get_physical_active_qubits(circuit: QuantumCircuit) -> List[int]:
    if circuit.layout is None:
        return list(range(circuit.num_qubits))
    return circuit.layout.final_index_layout(filter_ancillas=True)


def get_measured_physical_qubits(circuit: QuantumCircuit) -> List[int]:
    """Physical qubits read out by the circuit, ordered by destination clbit."""
    measured_ordered: List[Optional[int]] = [None] * circuit.num_clbits
    for instruction in circuit.data:
        if instruction.operation.name == "measure":
            for q, c in zip(instruction.qubits, instruction.clbits):
                measured_ordered[circuit.find_bit(c).index] = circuit.find_bit(q).index
    return [q for q in measured_ordered if q is not None]


def get_nearest_probabilities(
    probabilities: Dict[str, float], qubits_count: int
) -> Dict[str, float]:
    """Closest valid probability distribution (L2) to a quasi-distribution."""
    sorted_probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1]))
    nearest_probabilities: Dict[str, float] = {}
    remaining = len(sorted_probabilities)
    beta = 0.0
    for state, value in sorted_probabilities.items():
        weighted_beta = beta / remaining
        if value + weighted_beta < 0:
            beta += value
            remaining -= 1
        else:
            nearest_probabilities[state] = value + weighted_beta
    return nearest_probabilities


def _generate_calibration_circuits(n_qubits: int, local: bool):
    if not local:
        zeros = QuantumCircuit(n_qubits)
        zeros.measure_all()
        ones = QuantumCircuit(n_qubits)
        for i in range(n_qubits):
            ones.x(i)
        ones.measure_all()
        return [zeros, ones]

    circuits = []
    qreg = QuantumRegister(n_qubits, name="q")
    creg = ClassicalRegister(1, name="meas")
    base = QuantumCircuit(qreg, creg)
    for i in range(n_qubits):
        zero = base.copy()
        zero.measure(i, creg[0])
        one = base.copy()
        one.x(i)
        one.measure(i, creg[0])
        circuits.extend([zero, one])
    return circuits


def _calibrate_confusion_matrices(
    calibration_probs: List[Dict[str, float]], n_qubits: int, local: bool
) -> np.ndarray:
    confusion_matrix = np.zeros([n_qubits, 2, 2], dtype=float)
    if not local:
        zeros, ones = calibration_probs[0], calibration_probs[1]
        for state, p0 in zeros.items():
            for q, bit in enumerate(reversed(state)):
                confusion_matrix[q, 0 if bit == "0" else 1, 0] += p0
        for state, p1 in ones.items():
            for q, bit in enumerate(reversed(state)):
                confusion_matrix[q, 0 if bit == "0" else 1, 1] += p1
    else:
        for i, prob in enumerate(calibration_probs):
            initial_state = i % 2
            for state, p in prob.items():
                confusion_matrix[i // 2, 0 if state == "0" else 1, initial_state] += p
    return confusion_matrix


def _get_reduced_confusion_matrix(
    confusion_matrices_all: np.ndarray,
    bitstrings: List[str],
    measured_qubits: List[int],
    matrix_format: MatrixFormat = "auto",
    dtype=np.float64,
    min_sparse_size: int = 1_000,
    max_dense_size: int = 8_000,
    max_sparsity_ratio: float = 0.2,
    d: int = 3,
):
    NQUBITS = len(measured_qubits)
    confusion_matrices = np.array([confusion_matrices_all[q] for q in measured_qubits])
    n = len(bitstrings)

    bitstrings_array = (
        np.array(bitstrings, dtype="S").view(np.uint8).reshape(len(bitstrings), -1) - ord("0")
    )
    hamming_distances = pdist(bitstrings_array, metric="hamming") * NQUBITS
    hamming_distance_isclose = np.less_equal(hamming_distances, d, out=hamming_distances)
    nonzero = np.count_nonzero(hamming_distance_isclose) + n
    sparsity_ratio = nonzero / ((n * (n - 1) / 2) + n)

    is_large_enough = n >= min_sparse_size
    is_sparse_enough = sparsity_ratio <= max_sparsity_ratio
    is_too_big_for_dense = n > max_dense_size
    decision_boundary = (is_large_enough and is_sparse_enough) or is_too_big_for_dense
    use_sparse = matrix_format == "sparse" or (matrix_format == "auto" and decision_boundary)

    if use_sparse:
        idx_1d = np.flatnonzero(hamming_distance_isclose)
        i_u = (n - 2 - np.floor(np.sqrt(-8 * idx_1d + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).astype(int)
        offset = i_u * (n - 1) - (i_u * (i_u - 1) // 2)
        j_u = idx_1d - offset + i_u + 1
        mask_indices_rows = np.concatenate([i_u, j_u, np.arange(n)])
        mask_indices_cols = np.concatenate([j_u, i_u, np.arange(n)])
        transition_probs_1d = np.ones(len(mask_indices_rows), dtype=dtype)
        for k, confusion_matrix in enumerate(confusion_matrices):
            bits_k = bitstrings_array[:, NQUBITS - 1 - k]
            transition_probs_1d *= confusion_matrix[bits_k[mask_indices_rows], bits_k[mask_indices_cols]]
        col_sums = np.bincount(mask_indices_cols, weights=transition_probs_1d, minlength=n)
        transition_probs_1d /= col_sums[mask_indices_cols]
        return csr_matrix((transition_probs_1d, (mask_indices_rows, mask_indices_cols)), shape=(n, n)), True

    mask_matrix = squareform(hamming_distance_isclose)
    np.fill_diagonal(mask_matrix, 1.0)
    transition_probs = mask_matrix
    for k, confusion_matrix in enumerate(confusion_matrices):
        qubit_k_values = bitstrings_array[:, NQUBITS - 1 - k]
        row_indices, col_indices = np.ix_(qubit_k_values, qubit_k_values)
        transition_probs = transition_probs * confusion_matrix[row_indices, col_indices]
    norm = np.sum(transition_probs, axis=0, keepdims=True)
    return transition_probs / norm, False


def _mitigate_probs(
    raw_probs: Dict[str, float],
    qubits_count: int,
    measured_qubits: List[int],
    confusion_matrices_all: np.ndarray,
    matrix_format: MatrixFormat = "auto",
    solver_tol: float = 1e-6,
    backend_impl: BackendImpl = "auto",
    project_to_probability_simplex: bool = True,
    return_details: bool = False,
    **reduce_kwargs,
) -> Any:
    if backend_impl not in ("python", "cpp", "auto"):
        raise ValueError("backend_impl must be 'python', 'cpp', or 'auto'")
    if backend_impl in ("cpp", "auto"):
        try:
            quasi, diagnostics = _mitigate_probs_cpp(
                raw_probs,
                measured_qubits,
                confusion_matrices_all,
                matrix_format,
                solver_tol,
                reduce_kwargs,
            )
        except (ImportError, ModuleNotFoundError):
            if backend_impl == "cpp":
                raise RuntimeError(
                    "backend_impl='cpp' requested, but rem_cpp is not built or installed"
                ) from None
        else:
            mitigated = (
                get_nearest_probabilities(quasi, qubits_count)
                if project_to_probability_simplex
                else quasi
            )
            if return_details:
                return mitigated, quasi, diagnostics
            return mitigated

    bitstrings = list(raw_probs.keys())
    raw_dense_array = np.array(list(raw_probs.values()), dtype=float)

    reduced_confusion_matrix, use_sparse = _get_reduced_confusion_matrix(
        confusion_matrices_all, bitstrings, measured_qubits, matrix_format, **reduce_kwargs
    )

    if use_sparse:
        mitigated, info = sparse_linalg.lgmres(reduced_confusion_matrix, raw_dense_array, rtol=solver_tol)
        mitigated /= np.sum(mitigated)
    else:
        mitigated = np.linalg.solve(reduced_confusion_matrix, raw_dense_array)

    quasi = dict(zip(bitstrings, mitigated))
    output = (
        get_nearest_probabilities(quasi, qubits_count)
        if project_to_probability_simplex
        else quasi
    )
    if return_details:
        residual = np.linalg.norm(reduced_confusion_matrix @ mitigated - raw_dense_array)
        diagnostics = {
            "selected_backend": "explicit_sparse" if use_sparse else "dense",
            "selected_solver": "lgmres" if use_sparse else "numpy_solve",
            "residual_norm": float(residual),
            "converged": bool(not use_sparse or info == 0),
            "backend_impl": "python",
        }
        return output, quasi, diagnostics
    return output


def _counts_to_probs(counts: Dict[str, float]) -> Dict[str, float]:
    counts = {state.replace(" ", ""): value for state, value in counts.items()}
    shots = sum(counts.values())
    return {state: value / shots for state, value in counts.items()}


def mitigate_readout_errors(
    circuit: QuantumCircuit,
    backend,
    shots: int = 10_000,
    local: bool = True,
    matrix_format: MatrixFormat = "auto",
    optimization_level: int = 1,
    calibration_shots: Optional[int] = None,
    n_qubits: Optional[int] = None,
    seed_transpiler: Optional[int] = None,
    backend_impl: BackendImpl = "auto",
    project_to_probability_simplex: bool = True,
    return_diagnostics: bool = False,
    **solver_options,
) -> Dict[str, Any]:
    """Run ``circuit`` on ``backend`` and return raw and readout-mitigated probabilities.

    Args:
        circuit: Circuit to execute (measurements included). May measure a subset
            of its qubits.
        backend: A qiskit backend that supports ``.run(...)`` (e.g. AerSimulator).
        shots: Shots for the user circuit.
        local: If True calibrate per-qubit (2 * n_qubits circuits); if False use two
            all-zeros / all-ones circuits.
        matrix_format: "dense", "sparse" or "auto".
        optimization_level: Transpilation level for the user circuit.
        calibration_shots: Shots per calibration circuit (defaults to ``shots``).
        n_qubits: Number of physical qubits to calibrate (defaults to ``backend.num_qubits``).
        seed_transpiler: Optional transpiler seed for reproducibility.
        backend_impl: Computational core: "python", "cpp", or "auto".
        project_to_probability_simplex: If true, return the nearest valid
            distribution under ``"mitigated"``. The quasi-distribution remains
            available when ``return_diagnostics`` is true.
        return_diagnostics: Include the quasi-distribution and solver diagnostics.
        solver_options: C++ options such as ``d``, ``threads``,
            ``gmres_restart``, and ``memory_budget_bytes``.

    Returns:
        Dict with keys ``"raw"`` and ``"mitigated"``, plus ``"measured_physical_qubits"``.
    """
    transpiled = transpile(
        circuit, backend, optimization_level=optimization_level, seed_transpiler=seed_transpiler
    )
    measured_qubits = get_measured_physical_qubits(transpiled)
    qubits_count = transpiled.num_clbits

    if n_qubits is None:
        n_qubits = backend.num_qubits
    if calibration_shots is None:
        calibration_shots = shots

    calib_circuits = _generate_calibration_circuits(n_qubits, local)
    calib_result = backend.run(calib_circuits, shots=calibration_shots).result()
    calibration_device_seconds = _execution_seconds(calib_result)
    calib_probs = [_counts_to_probs(calib_result.get_counts(i)) for i in range(len(calib_circuits))]
    confusion_matrices = _calibrate_confusion_matrices(calib_probs, n_qubits, local)

    raw_result = backend.run(transpiled, shots=shots).result()
    raw_device_seconds = _execution_seconds(raw_result)
    raw_counts = raw_result.get_counts()
    raw_probs = _counts_to_probs(raw_counts)

    mitigation_output = _mitigate_probs(
        raw_probs,
        qubits_count,
        measured_qubits,
        confusion_matrices,
        matrix_format,
        backend_impl=backend_impl,
        project_to_probability_simplex=project_to_probability_simplex,
        return_details=return_diagnostics,
        **solver_options,
    )
    if return_diagnostics:
        mitigated_probs, quasi_probs, diagnostics = mitigation_output
    else:
        mitigated_probs = mitigation_output

    output = {
        "raw": raw_probs,
        "mitigated": mitigated_probs,
        "measured_physical_qubits": measured_qubits,
    }
    if return_diagnostics:
        if (
            calibration_device_seconds is not None
            and raw_device_seconds is not None
        ):
            diagnostics["device_seconds"] = (
                calibration_device_seconds + raw_device_seconds
            )
            diagnostics["device_time_source"] = "qiskit_result.time_taken"
        diagnostics["calibration_device_seconds"] = calibration_device_seconds
        diagnostics["raw_device_seconds"] = raw_device_seconds
        output["quasi"] = quasi_probs
        output["diagnostics"] = diagnostics
    return output
