
import argparse
import csv
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

from benchmark_utils import measure_call
from sparse_rem import _mitigate_probs, mitigate_readout_errors


DEFAULT_SIZES = [256, 1_024, 4_096]
DEFAULT_REPORTS_DIRECTORY = Path("results")
BENCHMARK_SEED = 2026


FIELDNAMES = [
    "timestamp",
    "scope",
    "benchmark_mode",
    "implementation",
    "requested_states",
    "possible_states",
    "actual_states",
    "n_qubits",
    "target_depth",
    "circuit_depth",
    "two_qubit_gates",
    "simulator_method",
    "shots",
    "repetition",
    "wall_seconds",
    "device_seconds",
    "device_time_source",
    "comparison_seconds",
    "comparison_time_source",
    "cpu_seconds",
    "baseline_rss_mib",
    "peak_rss_mib",
    "peak_rss_delta_mib",
    "raw_fidelity",
    "mitigated_fidelity",
    "selected_backend",
    "selected_solver",
    "residual_norm",
    "max_abs_difference_from_python",
    "status",
    "error",
]
LEGACY_OPTIONAL_FIELDS = {
    "benchmark_mode",
    "target_depth",
    "circuit_depth",
    "two_qubit_gates",
    "simulator_method",
}
LEGACY_FIELD_ALIASES: dict[str, str] = {}


def make_readout_noise_model(
    probability_0_to_1: float = 0.04,
    probability_1_to_0: float = 0.08,
) -> NoiseModel:
    """Create independent asymmetric readout noise for every measured qubit."""
    if not 0 <= probability_0_to_1 < 1:
        raise ValueError("probability_0_to_1 must be in [0, 1)")
    if not 0 <= probability_1_to_0 < 1:
        raise ValueError("probability_1_to_0 must be in [0, 1)")

    error = ReadoutError(
        [
            [1 - probability_0_to_1, probability_0_to_1],
            [probability_1_to_0, 1 - probability_1_to_0],
        ]
    )
    noise_model = NoiseModel()
    noise_model.add_all_qubit_readout_error(error)
    return noise_model


def state_space(requested_states: int) -> tuple[int, int]:
    """Return qubit count and the nearest covering power-of-two state space."""
    if requested_states < 2:
        raise ValueError("Every requested size must be at least 2")
    n_qubits = math.ceil(math.log2(requested_states))
    return n_qubits, 2**n_qubits


def make_uniform_circuit(n_qubits: int) -> QuantumCircuit:
    """Create a measured uniform-superposition circuit."""
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    circuit.measure_all()
    return circuit


def make_deep_ghz_circuit(
    n_qubits: int,
    target_depth: int,
    seed: int = 42,
) -> QuantumCircuit:
    """Create a deep entangled circuit with a known two-state ideal output.

    Random RZ and alternating CZ layers increase depth while changing only the
    relative phase of the two GHZ components, so the ideal measurement
    distribution remains concentrated on all-zero and all-one bitstrings.
    ``target_depth`` includes the final measurement layer.
    """
    minimum_depth = n_qubits + 1
    if target_depth < minimum_depth:
        raise ValueError(
            f"target_depth must be at least {minimum_depth} for a "
            f"{n_qubits}-qubit GHZ circuit"
        )

    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)

    random_generator = random.Random(seed)
    target_unitary_depth = target_depth - 1
    layer = 0
    while circuit.depth() < target_unitary_depth:
        for qubit in range(n_qubits):
            circuit.rz(
                random_generator.uniform(-math.pi, math.pi),
                qubit,
            )
        if circuit.depth() >= target_unitary_depth:
            break
        for qubit in range(layer % 2, n_qubits - 1, 2):
            circuit.cz(qubit, qubit + 1)
        layer += 1

    circuit.measure_all()
    return circuit


def ideal_uniform_distribution(n_qubits: int) -> dict[str, float]:
    """Return the exact uniform distribution over all computational states."""
    states_count = 2**n_qubits
    probability = 1.0 / states_count
    return {
        format(index, f"0{n_qubits}b"): probability
        for index in range(states_count)
    }


def ideal_ghz_distribution(n_qubits: int) -> dict[str, float]:
    """Return the exact computational-basis distribution of a GHZ state."""
    return {
        "0" * n_qubits: 0.5,
        "1" * n_qubits: 0.5,
    }


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Count all two-qubit operations in a circuit."""
    return sum(
        instruction.operation.num_qubits == 2
        for instruction in circuit.data
    )


def _run_local(
    backend_impl: str,
    circuit: QuantumCircuit,
    backend: AerSimulator,
    noise_model: NoiseModel,
    n_qubits: int,
    shots: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one complete local pipeline with an explicitly selected core."""
    del noise_model  # The local backend already owns this model.
    output, metrics = measure_call(
        mitigate_readout_errors,
        circuit,
        backend,
        shots=shots,
        local=True,
        matrix_format="auto",
        n_qubits=n_qubits,
        seed_transpiler=BENCHMARK_SEED,
        backend_impl=backend_impl,
        project_to_probability_simplex=True,
        return_diagnostics=True,
        d=3,
    )
    diagnostics = output["diagnostics"]
    metrics["device_seconds"] = diagnostics.get("device_seconds")
    metrics["device_time_source"] = diagnostics.get("device_time_source", "")
    return output, metrics


def run_python(
    circuit: QuantumCircuit,
    backend: AerSimulator,
    noise_model: NoiseModel,
    n_qubits: int,
    shots: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run and profile the original NumPy/SciPy computational core."""
    return _run_local("python", circuit, backend, noise_model, n_qubits, shots)


def run_cpp(
    circuit: QuantumCircuit,
    backend: AerSimulator,
    noise_model: NoiseModel,
    n_qubits: int,
    shots: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the same local pipeline with only the computational core changed."""
    return _run_local("cpp", circuit, backend, noise_model, n_qubits, shots)


RUNNERS = {
    "python": run_python,
    "cpp": run_cpp,
}


def result_row(
    *,
    implementation: str,
    benchmark_mode: str,
    requested_states: int | str,
    possible_states: int,
    n_qubits: int,
    target_depth: int | str,
    circuit_depth: int,
    two_qubit_gates: int,
    simulator_method: str,
    shots: int,
    repetition: int,
    output: dict[str, Any],
    metrics: dict[str, float],
    ideal: dict[str, float],
) -> dict[str, Any]:
    """Convert one successful local measured run to a CSV row."""
    mitigated = output["mitigated"]
    raw = output.get("raw") or {}
    diagnostics = output.get("diagnostics") or {}
    device_seconds = metrics.get("device_seconds")
    has_device_time = (
        isinstance(device_seconds, (int, float))
        and math.isfinite(device_seconds)
        and device_seconds >= 0
    )
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scope": "end_to_end",
        "benchmark_mode": benchmark_mode,
        "implementation": implementation,
        "requested_states": requested_states,
        "possible_states": possible_states,
        "actual_states": len(raw) if raw else len(mitigated),
        "n_qubits": n_qubits,
        "target_depth": target_depth,
        "circuit_depth": circuit_depth,
        "two_qubit_gates": two_qubit_gates,
        "simulator_method": simulator_method,
        "shots": shots,
        "repetition": repetition,
        "wall_seconds": metrics["wall_seconds"],
        "device_seconds": device_seconds if has_device_time else "",
        "device_time_source": metrics.get("device_time_source", ""),
        "comparison_seconds": metrics["wall_seconds"],
        "comparison_time_source": "wall_clock",
        "cpu_seconds": metrics.get("cpu_seconds", ""),
        "baseline_rss_mib": metrics.get("baseline_rss_mib", ""),
        "peak_rss_mib": metrics.get("peak_rss_mib", ""),
        "peak_rss_delta_mib": metrics.get("peak_rss_delta_mib", ""),
        "raw_fidelity": hellinger_fidelity(ideal, raw) if raw else "",
        "mitigated_fidelity": hellinger_fidelity(ideal, mitigated),
        "selected_backend": diagnostics.get("selected_backend", ""),
        "selected_solver": diagnostics.get("selected_solver", ""),
        "residual_norm": diagnostics.get("residual_norm", ""),
        "max_abs_difference_from_python": "",
        "status": "ok",
        "error": "",
    }


def failure_row(
    *,
    implementation: str,
    benchmark_mode: str,
    requested_states: int | str,
    possible_states: int,
    n_qubits: int,
    target_depth: int | str,
    circuit_depth: int,
    two_qubit_gates: int,
    simulator_method: str,
    shots: int,
    repetition: int,
    error: Exception,
) -> dict[str, Any]:
    """Convert one failed measured run to a CSV row without stopping the suite."""
    row = {field: "" for field in FIELDNAMES}
    row.update(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scope": "end_to_end",
            "benchmark_mode": benchmark_mode,
            "implementation": implementation,
            "requested_states": requested_states,
            "possible_states": possible_states,
            "n_qubits": n_qubits,
            "target_depth": target_depth,
            "circuit_depth": circuit_depth,
            "two_qubit_gates": two_qubit_gates,
            "simulator_method": simulator_method,
            "shots": shots,
            "repetition": repetition,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    )
    return row


def append_row(output_path: Path, row: dict[str, Any]) -> None:
    """Append one row immediately, preserving partial results after failures."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )
    if not write_header:
        with output_path.open(newline="", encoding="utf-8") as existing_file:
            existing_header = next(csv.reader(existing_file), [])
        if existing_header != FIELDNAMES:
            raise ValueError(
                f"Cannot append the current schema to {output_path}; "
                "use a new output path."
            )
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    """Load a benchmark CSV, including reports from before deep-circuit mode."""
    with input_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        actual_fields = {
            LEGACY_FIELD_ALIASES.get(field, field)
            for field in (reader.fieldnames or ())
        }
        expected_fields = set(FIELDNAMES)
        required_missing = (
            expected_fields - actual_fields - LEGACY_OPTIONAL_FIELDS
        )
        if required_missing:
            raise ValueError(
                f"Unexpected benchmark schema in {input_path}: "
                f"{reader.fieldnames!r}"
            )
        rows = []
        for legacy_row in reader:
            normalized_row = dict(legacy_row)
            for legacy_field, current_field in LEGACY_FIELD_ALIASES.items():
                if legacy_field in normalized_row:
                    normalized_row[current_field] = normalized_row[legacy_field]
            row = {
                field: normalized_row.get(field, "")
                for field in FIELDNAMES
            }
            if not row["benchmark_mode"]:
                row["benchmark_mode"] = (
                    "local_core"
                    if row["scope"] == "local_core"
                    else "state_space"
                )
            rows.append(row)
        return rows


def next_report_path(reports_directory: Path) -> Path:
    """Create the reports directory and return the next indexed CSV path."""
    reports_directory.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"benchmark_(\d+)\.csv")
    indices = []

    for path in reports_directory.glob("benchmark_*.csv"):
        match = pattern.fullmatch(path.name)
        if match:
            indices.append(int(match.group(1)))

    next_index = max(indices, default=0) + 1
    return reports_directory / f"benchmark_{next_index:03d}.csv"


def run_benchmark(
    *,
    sizes: list[int] | None,
    implementations: list[str],
    warmups: int,
    repeats: int,
    shots_per_state: int,
    output_path: Path | None,
    qubits: int | None = None,
    target_depth: int = 30,
    fixed_shots: int = 10_000,
) -> list[dict[str, Any]]:
    """Run warmups and measured repetitions for every requested configuration."""
    noise_model = make_readout_noise_model()
    local_backend = AerSimulator(
        method="matrix_product_state",
        noise_model=noise_model,
        seed_simulator=BENCHMARK_SEED,
    )
    rows: list[dict[str, Any]] = []

    cases: list[dict[str, Any]] = []
    if qubits is not None:
        circuit = make_deep_ghz_circuit(qubits, target_depth)
        cases.append(
            {
                "benchmark_mode": "qubits_depth",
                "requested_states": "",
                "possible_states": 1 << qubits,
                "n_qubits": qubits,
                "target_depth": target_depth,
                "shots": fixed_shots,
                "circuit": circuit,
                "ideal": ideal_ghz_distribution(qubits),
            }
        )
    else:
        for requested_states in sizes or DEFAULT_SIZES:
            n_qubits, possible_states = state_space(requested_states)
            cases.append(
                {
                    "benchmark_mode": "state_space",
                    "requested_states": requested_states,
                    "possible_states": possible_states,
                    "n_qubits": n_qubits,
                    "target_depth": "",
                    "shots": possible_states * shots_per_state,
                    "circuit": make_uniform_circuit(n_qubits),
                    "ideal": ideal_uniform_distribution(n_qubits),
                }
            )

    for case in cases:
        benchmark_mode = case["benchmark_mode"]
        requested_states = case["requested_states"]
        possible_states = case["possible_states"]
        n_qubits = case["n_qubits"]
        target_depth_value = case["target_depth"]
        shots = case["shots"]
        circuit = case["circuit"]
        ideal = case["ideal"]
        circuit_depth = circuit.depth()
        two_qubit_gates = count_two_qubit_gates(circuit)

        if benchmark_mode == "qubits_depth":
            print(
                f"\nDeep-circuit mode: {n_qubits} qubits, "
                f"target depth {target_depth_value}, "
                f"actual depth {circuit_depth}, "
                f"{two_qubit_gates} two-qubit gates, {shots} shots"
            )
        else:
            print(
                f"\nSize {requested_states}: {n_qubits} qubits, "
                f"{possible_states} possible states, {shots} shots"
            )

        for implementation in implementations:
            runner = RUNNERS[implementation]
            simulator_method = "matrix_product_state"
            print(
                f"  {implementation}"
                + (f": {warmups} warmup run(s)" if warmups else ""),
                flush=True,
            )

            for warmup in range(warmups):
                try:
                    runner(
                        circuit,
                        local_backend,
                        noise_model,
                        n_qubits,
                        shots,
                    )
                    print(
                        f"    warmup {warmup + 1}/{warmups} complete",
                        flush=True,
                    )
                except Exception as error:
                    print(
                        f"    warmup {warmup + 1}/{warmups} failed: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )

            for repetition in range(1, repeats + 1):
                print(
                    f"    measured run {repetition}/{repeats}",
                    flush=True,
                )
                try:
                    output, metrics = runner(
                        circuit,
                        local_backend,
                        noise_model,
                        n_qubits,
                        shots,
                    )
                    row = result_row(
                        implementation=implementation,
                        benchmark_mode=benchmark_mode,
                        requested_states=requested_states,
                        possible_states=possible_states,
                        n_qubits=n_qubits,
                        target_depth=target_depth_value,
                        circuit_depth=circuit_depth,
                        two_qubit_gates=two_qubit_gates,
                        simulator_method=simulator_method,
                        shots=shots,
                        repetition=repetition,
                        output=output,
                        metrics=metrics,
                        ideal=ideal,
                    )
                    memory_text = (
                        f", peak={row['peak_rss_mib']:.1f}MiB"
                        if isinstance(row["peak_rss_mib"], (int, float))
                        else ""
                    )
                    print(
                        f"      wall={row['wall_seconds']:.3f}s"
                        f"{memory_text}, "
                        f"fidelity={row['mitigated_fidelity']:.4f}",
                        flush=True,
                    )
                except Exception as error:
                    row = failure_row(
                        implementation=implementation,
                        benchmark_mode=benchmark_mode,
                        requested_states=requested_states,
                        possible_states=possible_states,
                        n_qubits=n_qubits,
                        target_depth=target_depth_value,
                        circuit_depth=circuit_depth,
                        two_qubit_gates=two_qubit_gates,
                        simulator_method=simulator_method,
                        shots=shots,
                        repetition=repetition,
                        error=error,
                    )
                    print(f"      FAILED: {row['error']}", flush=True)

                rows.append(row)
                if output_path is not None:
                    append_row(output_path, row)

    if output_path is not None:
        print(f"\nResults saved to: {output_path.resolve()}")
    return rows


def _make_core_problem(
    n_qubits: int,
    unique_states: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Create deterministic raw data shared exactly by both local cores."""
    if not 1 <= n_qubits <= 63:
        raise ValueError("core benchmark currently supports 1 <= n_qubits <= 63")
    universe = 1 << n_qubits
    if not 1 <= unique_states <= universe:
        raise ValueError("unique_states must be in [1, 2**n_qubits]")
    rng = np.random.default_rng(seed)
    states = np.sort(rng.choice(universe, size=unique_states, replace=False))
    values = rng.random(unique_states)
    values /= values.sum()
    raw = {
        format(int(state), f"0{n_qubits}b"): float(value)
        for state, value in zip(states, values)
    }

    matrices = np.empty((n_qubits, 2, 2), dtype=np.float64)
    p_0_to_1 = rng.uniform(0.01, 0.04, size=n_qubits)
    p_1_to_0 = rng.uniform(0.02, 0.08, size=n_qubits)
    matrices[:, 0, 0] = 1.0 - p_0_to_1
    matrices[:, 1, 0] = p_0_to_1
    matrices[:, 0, 1] = p_1_to_0
    matrices[:, 1, 1] = 1.0 - p_1_to_0
    return raw, matrices


def benchmark_local_core(
    *,
    n_qubits: int = 16,
    unique_states: int = 1_000,
    max_hamming_distance: int = 3,
    warmups: int = 1,
    repeats: int = 7,
    seed: int = BENCHMARK_SEED,
) -> list[dict[str, Any]]:
    """Benchmark Python and C++ using identical raw data and calibrations."""
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    raw, matrices = _make_core_problem(n_qubits, unique_states, seed)
    measured_qubits = list(range(n_qubits))
    outputs: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []

    def run_once(implementation: str):
        _, quasi, diagnostics = _mitigate_probs(
            raw,
            n_qubits,
            measured_qubits,
            matrices,
            matrix_format="auto",
            solver_tol=1e-7,
            backend_impl=implementation,
            project_to_probability_simplex=False,
            return_details=True,
            d=max_hamming_distance,
        )
        return quasi, diagnostics

    for implementation in ("python", "cpp"):
        for _ in range(warmups):
            run_once(implementation)
        for repetition in range(1, repeats + 1):
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            quasi, diagnostics = run_once(implementation)
            cpu_seconds = time.process_time() - cpu_start
            wall_seconds = time.perf_counter() - wall_start
            outputs[implementation] = quasi
            rows.append(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "scope": "local_core",
                    "benchmark_mode": "local_core",
                    "implementation": implementation,
                    "requested_states": unique_states,
                    "possible_states": 1 << n_qubits,
                    "actual_states": unique_states,
                    "n_qubits": n_qubits,
                    "target_depth": "",
                    "circuit_depth": "",
                    "two_qubit_gates": "",
                    "simulator_method": "",
                    "shots": "",
                    "repetition": repetition,
                    "wall_seconds": wall_seconds,
                    "device_seconds": "",
                    "device_time_source": "",
                    "comparison_seconds": wall_seconds,
                    "comparison_time_source": "wall_clock",
                    "cpu_seconds": cpu_seconds,
                    "baseline_rss_mib": "",
                    "peak_rss_mib": "",
                    "peak_rss_delta_mib": "",
                    "raw_fidelity": "",
                    "mitigated_fidelity": "",
                    "selected_backend": diagnostics.get("selected_backend", ""),
                    "selected_solver": diagnostics.get("selected_solver", ""),
                    "residual_norm": diagnostics.get("residual_norm", ""),
                    "max_abs_difference_from_python": "",
                    "status": "ok",
                    "error": "",
                }
            )

    keys = list(raw)
    max_difference = max(
        abs(outputs["python"][key] - outputs["cpp"][key]) for key in keys
    )
    for row in rows:
        if row["implementation"] == "cpp":
            row["max_abs_difference_from_python"] = max_difference
    return rows


def _row_summary_size(row: dict[str, Any]) -> int:
    """Choose a stable numeric size for state-space and deep-circuit rows."""
    for field in ("requested_states", "possible_states", "actual_states"):
        try:
            value = int(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return compact per-implementation medians for notebook display."""
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            key = (
                str(row.get("scope", "")),
                _row_summary_size(row),
                str(row["implementation"]),
            )
            groups.setdefault(key, []).append(row)

    python_wall = {
        (scope, requested_states): statistics.median(
            float(row["wall_seconds"])
            for row in group
        )
        for (scope, requested_states, implementation), group in groups.items()
        if implementation == "python"
    }
    summaries = []
    for (scope, requested_states, implementation), group in sorted(groups.items()):
        wall = statistics.median(float(row["wall_seconds"]) for row in group)
        device_values = []
        for row in group:
            try:
                device_value = float(row["device_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(device_value) and device_value >= 0:
                device_values.append(device_value)
        summaries.append(
            {
                "scope": scope,
                "requested_states": requested_states,
                "implementation": implementation,
                "wall_median_s": wall,
                "device_median_s": (
                    statistics.median(device_values) if device_values else None
                ),
                "wall_speedup_vs_python": (
                    python_wall[(scope, requested_states)] / wall
                    if (scope, requested_states) in python_wall
                    else None
                ),
                "runs": len(group),
            }
        )
    return summaries


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a dependency-free local summary table."""
    summaries = summarize_rows(rows)
    show_device = any(
        item["device_median_s"] is not None for item in summaries
    )

    header = (
        f"{'scope':<12} {'states':>8} {'implementation':<14} "
        f"{'wall median':>12}"
    )
    if show_device:
        header += f" {'device median':>14}"
    header += f" {'vs Python':>11}"
    print(header)

    for item in summaries:
        line = (
            f"{item['scope']:<12} {item['requested_states']:>8} "
            f"{item['implementation']:<14} "
            f"{item['wall_median_s']:>11.6f}s"
        )
        if show_device:
            device = (
                f"{item['device_median_s']:.6f}s"
                if item["device_median_s"] is not None
                else ""
            )
            line += f" {device:>14}"
        speedup = (
            f"{item['wall_speedup_vs_python']:.3f}x"
            if item["wall_speedup_vs_python"] is not None
            else ""
        )
        print(line + f" {speedup:>11}")


def _scaling_series(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    aggregation: str = "median",
) -> dict[str, list[tuple[int, float, float, float]]]:
    """Aggregate positive finite samples as K, central value, min, and max."""
    if aggregation not in ("median", "maximum"):
        raise ValueError("aggregation must be 'median' or 'maximum'")
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if (
            row.get("status") != "ok"
            or row.get("scope") != "end_to_end"
            or row.get("benchmark_mode", "") not in ("", "state_space")
        ):
            continue
        try:
            value = float(row[metric])
            size = int(row["requested_states"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0 or size <= 0:
            continue
        key = str(row["implementation"]), size
        grouped.setdefault(key, []).append(value)

    series: dict[str, list[tuple[int, float, float, float]]] = {}
    for (implementation, size), values in grouped.items():
        series.setdefault(implementation, []).append(
            (
                size,
                (
                    statistics.median(values)
                    if aggregation == "median"
                    else max(values)
                ),
                min(values),
                max(values),
            )
        )
    for points in series.values():
        points.sort(key=lambda point: point[0])
    return series


def plot_scaling(
    rows: list[dict[str, Any]],
    *,
    output_directory: Path | None = None,
    filename_prefix: str = "rem_scaling",
) -> tuple[Any, Any]:
    """Plot local end-to-end runtime and peak RSS for Python and C++."""
    from matplotlib import pyplot as plt

    runtime = _scaling_series(rows, "wall_seconds")
    memory = _scaling_series(rows, "peak_rss_mib")
    if not runtime:
        raise ValueError("No positive end-to-end runtime samples are available")
    if not memory:
        raise ValueError("No positive end-to-end peak-RSS samples are available")

    styles = {
        "python": {
            "label": "Python",
            "color": "#4C78A8",
            "marker": "o",
        },
        "cpp": {
            "label": "C++",
            "color": "#F58518",
            "marker": "s",
        },
    }
    implementation_order = ("python", "cpp")
    def draw(
        series: dict[str, list[tuple[int, float, float, float]]],
        *,
        ylabel: str,
        title: str,
    ):
        figure, axis = plt.subplots(figsize=(9.0, 5.4))
        all_sizes: set[int] = set()
        for implementation in implementation_order:
            points = series.get(implementation)
            if not points:
                continue
            style = styles[implementation]
            sizes = np.asarray([point[0] for point in points], dtype=float)
            medians = np.asarray([point[1] for point in points], dtype=float)
            minima = np.asarray([point[2] for point in points], dtype=float)
            maxima = np.asarray([point[3] for point in points], dtype=float)
            all_sizes.update(int(size) for size in sizes)
            axis.plot(
                sizes,
                medians,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.2,
                markersize=6,
                label=style["label"],
            )
            axis.fill_between(
                sizes,
                minima,
                maxima,
                color=style["color"],
                alpha=0.14,
                linewidth=0,
            )

        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Number of observed states, K")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xticks(sorted(all_sizes))
        axis.set_xticklabels(
            [
                rf"$2^{{{int(math.log2(size))}}}$"
                if size > 0 and size & (size - 1) == 0
                else f"{size:,}"
                for size in sorted(all_sizes)
            ]
        )
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.10)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        return figure

    runtime_figure = draw(
        runtime,
        ylabel="Median runtime, s",
        title="REM runtime",
    )
    memory_figure = draw(
        memory,
        ylabel="Peak RSS, MiB",
        title="REM memory",
    )

    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        runtime_path = output_directory / f"{filename_prefix}_runtime.svg"
        memory_path = output_directory / f"{filename_prefix}_memory.svg"
        runtime_figure.savefig(runtime_path, format="svg", bbox_inches="tight")
        memory_figure.savefig(memory_path, format="svg", bbox_inches="tight")
        print(f"Runtime plot saved to: {runtime_path.resolve()}")
        print(f"Memory plot saved to: {memory_path.resolve()}")
    return runtime_figure, memory_figure


CORE_SCALING_FIELDNAMES = [
    "timestamp",
    "suite",
    "variant",
    "label",
    "matrix_power",
    "matrix_size",
    "n_qubits",
    "max_hamming_distance",
    "repetition",
    "wall_seconds",
    "cpu_seconds",
    "baseline_rss_mib",
    "peak_rss_mib",
    "peak_rss_delta_mib",
    "selected_backend",
    "selected_solver",
    "selected_sparse_storage",
    "retained_edges",
    "iterations",
    "residual_norm",
    "quasi_sum",
    "quasi_l2",
    "status",
    "error",
]

CORE_VARIANT_LABELS = {
    "python_auto": "Python",
    "cpp_auto": "C++",
    "cpp_dense": "Dense",
    "cpp_sparse_valued": "Sparse CSR",
    "cpp_sparse_topology": "Sparse topology",
    "cpp_matrix_free": "Matrix-free",
}

CORE_SCALING_WORKER = r"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from benchmark_utils import measure_call
from sparse_rem import _load_cpp_backend, _mitigate_probs


def make_problem(n_qubits: int, matrix_size: int, seed: int):
    rng = np.random.default_rng(seed)
    universe = 1 << n_qubits
    states = np.sort(
        rng.choice(universe, size=matrix_size, replace=False)
    ).astype(np.uint64)
    probabilities = rng.random(matrix_size)
    probabilities /= probabilities.sum()
    bitstrings = [
        format(int(state), f"0{n_qubits}b")
        for state in states
    ]
    raw = {
        state: float(probability)
        for state, probability in zip(bitstrings, probabilities)
    }

    calibrations = np.empty((n_qubits, 2, 2), dtype=np.float64)
    probability_0_to_1 = rng.uniform(0.01, 0.04, size=n_qubits)
    probability_1_to_0 = rng.uniform(0.02, 0.08, size=n_qubits)
    calibrations[:, 0, 0] = 1.0 - probability_0_to_1
    calibrations[:, 1, 0] = probability_0_to_1
    calibrations[:, 0, 1] = probability_1_to_0
    calibrations[:, 1, 1] = 1.0 - probability_1_to_0
    return states, probabilities, bitstrings, raw, calibrations


config = json.loads(sys.argv[1])
variant = config["variant"]
n_qubits = int(config["n_qubits"])
matrix_size = int(config["matrix_size"])
cutoff = int(config["max_hamming_distance"])
states, probabilities, bitstrings, raw, calibrations = make_problem(
    n_qubits,
    matrix_size,
    int(config["seed"]),
)
measured_qubits = list(range(n_qubits))
cpp_backend = None
if variant != "python_auto":
    cpp_backend = _load_cpp_backend()


def solve():
    if variant == "python_auto":
        _, quasi, diagnostics = _mitigate_probs(
            raw,
            n_qubits,
            measured_qubits,
            calibrations,
            matrix_format="auto",
            solver_tol=float(config["relative_tolerance"]),
            backend_impl="python",
            project_to_probability_simplex=False,
            return_details=True,
            d=cutoff,
        )
        vector = np.fromiter(
            (quasi[state] for state in bitstrings),
            dtype=np.float64,
            count=matrix_size,
        )
        return vector, diagnostics

    variant_options = {
        "cpp_auto": {
            "backend": "auto",
            "solver": "auto",
            "sparse_storage": "auto",
            "cache_values": False,
        },
        "cpp_dense": {
            "backend": "dense",
            "solver": "direct",
            "sparse_storage": "auto",
            "cache_values": False,
        },
        "cpp_sparse_valued": {
            "backend": "explicit_sparse",
            "solver": "gmres",
            "sparse_storage": "valued",
            "cache_values": True,
        },
        "cpp_sparse_topology": {
            "backend": "explicit_sparse",
            "solver": "gmres",
            "sparse_storage": "topology_only",
            "cache_values": False,
        },
        "cpp_matrix_free": {
            "backend": "matrix_free",
            "solver": "gmres",
            "sparse_storage": "auto",
            "cache_values": False,
        },
    }
    if variant not in variant_options:
        raise ValueError(f"Unknown core benchmark variant: {variant}")
    options = {
        **variant_options[variant],
        "max_hamming_distance": cutoff,
        "relative_tolerance": float(config["relative_tolerance"]),
        "absolute_tolerance": 0.0,
        "gmres_restart": int(config["gmres_restart"]),
        "max_iterations": int(config["max_iterations"]),
        "threads": int(config["threads"]),
        "memory_budget_bytes": int(config["memory_budget_bytes"]),
        "cache_topology": True,
        "project_to_probability_simplex": False,
    }
    output = cpp_backend.mitigate_packed(
        np.ascontiguousarray(states, dtype=np.uint64),
        np.ascontiguousarray(probabilities, dtype=np.float64),
        np.ascontiguousarray(calibrations, dtype=np.float64),
        n_qubits,
        options,
    )
    vector = np.asarray(
        output["quasi_probabilities"],
        dtype=np.float64,
    )
    return vector, dict(output["diagnostics"])


(quasi_vector, diagnostics), metrics = measure_call(
    solve,
    sample_interval_s=float(config["sample_interval_seconds"]),
)
payload = {
    "wall_seconds": metrics["wall_seconds"],
    "cpu_seconds": metrics["cpu_seconds"],
    "baseline_rss_mib": metrics["baseline_rss_mib"],
    "peak_rss_mib": metrics["peak_rss_mib"],
    "peak_rss_delta_mib": metrics["peak_rss_delta_mib"],
    "selected_backend": diagnostics.get("selected_backend", ""),
    "selected_solver": diagnostics.get("selected_solver", ""),
    "selected_sparse_storage": diagnostics.get(
        "selected_sparse_storage", ""
    ),
    "retained_edges": diagnostics.get("retained_edges", ""),
    "iterations": diagnostics.get("iterations", ""),
    "residual_norm": diagnostics.get("residual_norm", ""),
    "converged": bool(diagnostics.get("converged", True)),
    "quasi_sum": float(np.sum(quasi_vector)),
    "quasi_l2": float(np.linalg.norm(quasi_vector)),
}
if config.get("return_vector", False):
    payload["quasi_probabilities"] = quasi_vector.tolist()
print(json.dumps(payload, allow_nan=False))
"""


def _run_isolated_core_case(
    *,
    variant: str,
    matrix_power: int,
    max_hamming_distance: int,
    seed: int,
    threads: int,
    memory_budget_bytes: int,
    relative_tolerance: float,
    gmres_restart: int,
    max_iterations: int,
    timeout_seconds: float,
    return_vector: bool = False,
) -> dict[str, Any]:
    """Run one core solve in a fresh process for unbiased peak-RSS sampling."""
    if variant not in CORE_VARIANT_LABELS:
        raise ValueError(f"Unknown core benchmark variant: {variant}")
    if not 1 <= matrix_power <= 30:
        raise ValueError("matrix_power must be in [1, 30]")
    configuration = {
        "variant": variant,
        "matrix_size": 1 << matrix_power,
        "n_qubits": matrix_power,
        "max_hamming_distance": max_hamming_distance,
        "seed": seed,
        "threads": threads,
        "memory_budget_bytes": memory_budget_bytes,
        "relative_tolerance": relative_tolerance,
        "gmres_restart": gmres_restart,
        "max_iterations": max_iterations,
        "sample_interval_seconds": 0.002,
        "return_vector": return_vector,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            CORE_SCALING_WORKER,
            json.dumps(configuration),
        ],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant} at K=2^{matrix_power} failed:\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    output_lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not output_lines:
        raise RuntimeError(f"{variant} worker returned no JSON output")
    return json.loads(output_lines[-1])


def verify_core_variants(
    variants: list[str],
    *,
    matrix_power: int = 8,
    max_hamming_distance: int = 3,
    seed: int = BENCHMARK_SEED,
    threads: int = 0,
    memory_budget_bytes: int = 6 * 1024**3,
    relative_tolerance: float = 1e-8,
    timeout_seconds: float = 600.0,
) -> dict[str, float]:
    """Check all selected variants against Python or dense C++ on one case."""
    if not variants:
        raise ValueError("At least one variant is required")
    reference_variant = (
        "python_auto"
        if "python_auto" in variants
        else "cpp_dense"
    )
    ordered_variants = list(dict.fromkeys([reference_variant, *variants]))
    vectors: dict[str, np.ndarray] = {}
    for variant in ordered_variants:
        result = _run_isolated_core_case(
            variant=variant,
            matrix_power=matrix_power,
            max_hamming_distance=max_hamming_distance,
            seed=seed,
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
            relative_tolerance=relative_tolerance,
            gmres_restart=30,
            max_iterations=300,
            timeout_seconds=timeout_seconds,
            return_vector=True,
        )
        if not result["converged"]:
            raise RuntimeError(f"{variant} did not converge in verification")
        vectors[variant] = np.asarray(
            result["quasi_probabilities"],
            dtype=np.float64,
        )

    reference = vectors[reference_variant]
    differences = {}
    for variant in variants:
        difference = float(np.max(np.abs(vectors[variant] - reference)))
        differences[variant] = difference
        if not np.allclose(
            vectors[variant],
            reference,
            rtol=2e-6,
            atol=2e-8,
        ):
            raise AssertionError(
                f"{variant} disagrees with {reference_variant}: "
                f"max |difference| = {difference:.3e}"
            )
    return differences


def _append_core_scaling_row(
    output_path: Path,
    row: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CORE_SCALING_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                field: row.get(field, "")
                for field in CORE_SCALING_FIELDNAMES
            }
        )


def load_core_scaling_rows(input_path: Path) -> list[dict[str, Any]]:
    """Load an isolated-core scaling report."""
    with input_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != CORE_SCALING_FIELDNAMES:
            raise ValueError(
                f"Unexpected core-scaling schema in {input_path}: "
                f"{reader.fieldnames!r}"
            )
        return list(reader)


def run_isolated_core_scaling(
    *,
    suite: str,
    variants: list[str],
    matrix_powers: list[int],
    dense_max_power: int,
    repeats: int,
    warmups: int,
    max_hamming_distance: int,
    threads: int,
    memory_budget_bytes: int,
    output_path: Path,
    overwrite: bool = False,
    verify: bool = True,
    relative_tolerance: float = 1e-7,
    gmres_restart: int = 30,
    max_iterations: int = 300,
    timeout_seconds: float = 3_600.0,
) -> list[dict[str, Any]]:
    """Benchmark core variants with every measured sample in a fresh process."""
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; enable overwrite or change path"
            )
        output_path.unlink()
    if verify:
        verification_power = min(min(matrix_powers), dense_max_power)
        verification = verify_core_variants(
            variants,
            matrix_power=verification_power,
            max_hamming_distance=max_hamming_distance,
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
            relative_tolerance=min(relative_tolerance, 1e-8),
            timeout_seconds=timeout_seconds,
        )
        print("Verification max |difference|:", verification)

    rows: list[dict[str, Any]] = []
    for matrix_power in matrix_powers:
        for variant in variants:
            if variant == "cpp_dense" and matrix_power > dense_max_power:
                continue
            matrix_size = 1 << matrix_power
            configuration_name = (
                f"{CORE_VARIANT_LABELS[variant]}, K=2^{matrix_power}"
            )
            print(f"\n{configuration_name}", flush=True)
            for warmup in range(warmups):
                print(
                    f"  isolated warmup {warmup + 1}/{warmups}",
                    flush=True,
                )
                _run_isolated_core_case(
                    variant=variant,
                    matrix_power=matrix_power,
                    max_hamming_distance=max_hamming_distance,
                    seed=BENCHMARK_SEED + matrix_power,
                    threads=threads,
                    memory_budget_bytes=memory_budget_bytes,
                    relative_tolerance=relative_tolerance,
                    gmres_restart=gmres_restart,
                    max_iterations=max_iterations,
                    timeout_seconds=timeout_seconds,
                )

            for repetition in range(1, repeats + 1):
                print(
                    f"  measured run {repetition}/{repeats}",
                    flush=True,
                )
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "suite": suite,
                    "variant": variant,
                    "label": CORE_VARIANT_LABELS[variant],
                    "matrix_power": matrix_power,
                    "matrix_size": matrix_size,
                    "n_qubits": matrix_power,
                    "max_hamming_distance": max_hamming_distance,
                    "repetition": repetition,
                    "status": "ok",
                    "error": "",
                }
                try:
                    result = _run_isolated_core_case(
                        variant=variant,
                        matrix_power=matrix_power,
                        max_hamming_distance=max_hamming_distance,
                        seed=BENCHMARK_SEED + matrix_power,
                        threads=threads,
                        memory_budget_bytes=memory_budget_bytes,
                        relative_tolerance=relative_tolerance,
                        gmres_restart=gmres_restart,
                        max_iterations=max_iterations,
                        timeout_seconds=timeout_seconds,
                    )
                    row.update(result)
                    if not result["converged"]:
                        row["status"] = "failed"
                        row["error"] = "iterative solver did not converge"
                    print(
                        f"    wall={result['wall_seconds']:.6f}s, "
                        f"peak RSS={result['peak_rss_mib']:.1f} MiB, "
                        f"backend={result['selected_backend']}",
                        flush=True,
                    )
                except Exception as error:
                    row["status"] = "failed"
                    row["error"] = f"{type(error).__name__}: {error}"
                    print(f"    FAILED: {row['error']}", flush=True)
                rows.append(row)
                _append_core_scaling_row(output_path, row)
    print(f"\nCore scaling report saved to: {output_path.resolve()}")
    return rows


def _core_scaling_series(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, list[tuple[int, float, float, float]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            matrix_size = int(row["matrix_size"])
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if matrix_size <= 0 or not math.isfinite(value) or value <= 0:
            continue
        grouped.setdefault(
            (str(row["variant"]), matrix_size),
            [],
        ).append(value)

    output: dict[str, list[tuple[int, float, float, float]]] = {}
    for (variant, matrix_size), values in grouped.items():
        output.setdefault(variant, []).append(
            (
                matrix_size,
                statistics.median(values),
                min(values),
                max(values),
            )
        )
    for points in output.values():
        points.sort(key=lambda point: point[0])
    return output


def plot_core_scaling(
    rows: list[dict[str, Any]],
    *,
    title_prefix: str,
    output_directory: Path | None = None,
    filename_prefix: str = "core_scaling",
) -> tuple[Any, Any]:
    """Plot separate log-log core-runtime and isolated-worker peak-RSS charts."""
    from matplotlib import pyplot as plt

    runtime = _core_scaling_series(rows, "wall_seconds")
    memory = _core_scaling_series(rows, "peak_rss_mib")
    if not runtime or not memory:
        raise ValueError("No successful core-scaling samples are available")

    styles = {
        "python_auto": ("#4C78A8", "o"),
        "cpp_auto": ("#F58518", "s"),
        "cpp_dense": ("#E45756", "D"),
        "cpp_sparse_valued": ("#54A24B", "^"),
        "cpp_sparse_topology": ("#72B7B2", "v"),
        "cpp_matrix_free": ("#B279A2", "P"),
    }

    def draw(
        series: dict[str, list[tuple[int, float, float, float]]],
        *,
        ylabel: str,
        title: str,
    ):
        figure, axis = plt.subplots(figsize=(9.5, 5.6))
        all_sizes: set[int] = set()
        for variant in CORE_VARIANT_LABELS:
            points = series.get(variant)
            if not points:
                continue
            color, marker = styles[variant]
            sizes = np.asarray([point[0] for point in points], dtype=float)
            medians = np.asarray([point[1] for point in points], dtype=float)
            minima = np.asarray([point[2] for point in points], dtype=float)
            maxima = np.asarray([point[3] for point in points], dtype=float)
            all_sizes.update(int(size) for size in sizes)
            axis.plot(
                sizes,
                medians,
                color=color,
                marker=marker,
                linewidth=2.1,
                markersize=6,
                label=CORE_VARIANT_LABELS[variant],
            )
            axis.fill_between(
                sizes,
                minima,
                maxima,
                color=color,
                alpha=0.13,
                linewidth=0,
            )

        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Number of observed states, K")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        ordered_sizes = sorted(all_sizes)
        axis.set_xticks(ordered_sizes)
        axis.set_xticklabels(
            [rf"$2^{{{int(math.log2(size))}}}$" for size in ordered_sizes]
        )
        axis.grid(True, which="major", alpha=0.28)
        axis.grid(True, which="minor", alpha=0.10)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        return figure

    runtime_figure = draw(
        runtime,
        ylabel="Median runtime, s",
        title=f"{title_prefix} Runtime",
    )
    memory_figure = draw(
        memory,
        ylabel="Peak RSS, MiB",
        title=f"{title_prefix} Memory",
    )
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        runtime_path = (
            output_directory / f"{filename_prefix}_runtime.svg"
        )
        memory_path = (
            output_directory / f"{filename_prefix}_memory.svg"
        )
        runtime_figure.savefig(runtime_path, format="svg", bbox_inches="tight")
        memory_figure.savefig(memory_path, format="svg", bbox_inches="tight")
        print(f"Runtime plot saved to: {runtime_path.resolve()}")
        print(f"Memory plot saved to: {memory_path.resolve()}")
    return runtime_figure, memory_figure


Q64_MEMORY_VARIANT_LABELS = {
    "cpp_sparse_valued": "Sparse CSR",
    "cpp_matrix_free": "Matrix-free",
}

Q64_MEMORY_FIELDNAMES = [
    "timestamp",
    "suite",
    "variant",
    "label",
    "qubits",
    "words_per_state",
    "matrix_size",
    "clusters",
    "cluster_radius",
    "max_hamming_distance",
    "repetition",
    "wall_seconds",
    "cpu_seconds",
    "baseline_rss_mib",
    "peak_rss_mib",
    "peak_rss_delta_mib",
    "selected_backend",
    "selected_solver",
    "selected_graph_strategy",
    "selected_sparse_storage",
    "retained_edges",
    "estimated_working_memory_bytes",
    "operator_graph_bytes",
    "iterations",
    "residual_norm",
    "quasi_sum",
    "quasi_l2",
    "status",
    "error",
]

Q64_MEMORY_WORKER = r"""
from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np

from benchmark_utils import measure_call
from sparse_rem import _load_cpp_backend


def clustered_states(qubits: int, count: int, clusters: int, radius: int):
    if clusters != 2:
        raise ValueError("the Q=64+ benchmark currently requires two clusters")
    if count % clusters:
        raise ValueError("matrix_size must be divisible by clusters")
    words = (qubits + 63) // 64
    per_cluster = count // clusters
    available = sum(
        math.comb(qubits, rank) for rank in range(radius + 1)
    )
    if available < per_cluster:
        raise ValueError(
            f"a radius-{radius} cluster contains only {available} states, "
            f"but {per_cluster} are required"
        )

    zero = np.zeros(words, dtype=np.uint64)
    one = np.full(words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    tail_bits = qubits % 64
    if tail_bits:
        one[-1] = np.uint64((1 << tail_bits) - 1)

    output = []
    for center in (zero, one):
        cluster = [center.copy()]
        for rank in range(1, radius + 1):
            for changed in itertools.combinations(range(qubits), rank):
                state = center.copy()
                for qubit in changed:
                    state[qubit // 64] ^= np.uint64(1 << (qubit % 64))
                cluster.append(state)
                if len(cluster) == per_cluster:
                    break
            if len(cluster) == per_cluster:
                break
        output.extend(cluster)
    return np.ascontiguousarray(np.stack(output), dtype=np.uint64)


configuration = json.loads(sys.argv[1])
variant = configuration["variant"]
qubits = int(configuration["qubits"])
matrix_size = int(configuration["matrix_size"])
clusters = int(configuration["clusters"])
cluster_radius = int(configuration["cluster_radius"])
distance = int(configuration["max_hamming_distance"])
states = clustered_states(qubits, matrix_size, clusters, cluster_radius)

rng = np.random.default_rng(int(configuration["seed"]) + qubits)
probabilities = rng.random(matrix_size)
probabilities /= probabilities.sum()
calibrations = np.empty((qubits, 2, 2), dtype=np.float64)
p_0_to_1 = rng.uniform(0.0005, 0.0010, size=qubits)
p_1_to_0 = rng.uniform(0.0008, 0.0015, size=qubits)
calibrations[:, 0, 0] = 1.0 - p_0_to_1
calibrations[:, 1, 0] = p_0_to_1
calibrations[:, 0, 1] = p_1_to_0
calibrations[:, 1, 1] = 1.0 - p_1_to_0

options = {
    "backend": (
        "explicit_sparse"
        if variant == "cpp_sparse_valued"
        else "matrix_free"
    ),
    "solver": "gmres",
    "graph_strategy": "blocked_pairwise",
    "sparse_storage": "valued",
    "max_hamming_distance": distance,
    "relative_tolerance": float(configuration["relative_tolerance"]),
    "gmres_restart": int(configuration["gmres_restart"]),
    "max_iterations": int(configuration["max_iterations"]),
    "threads": int(configuration["threads"]),
    "memory_budget_bytes": int(configuration["memory_budget_bytes"]),
}
cpp = _load_cpp_backend()
(result, metrics) = measure_call(
    cpp.mitigate_packed,
    states,
    np.ascontiguousarray(probabilities, dtype=np.float64),
    np.ascontiguousarray(calibrations, dtype=np.float64),
    qubits,
    options,
    sample_interval_s=0.001,
)
diagnostics = dict(result["diagnostics"])
edges = int(diagnostics["retained_edges"])
operator_graph_bytes = (
    8 * (matrix_size + 1) + 12 * edges
    if variant == "cpp_sparse_valued"
    else 0
)
quasi = np.asarray(result["quasi_probabilities"], dtype=np.float64)
output = {
    **metrics,
    "selected_backend": diagnostics["selected_backend"],
    "selected_solver": diagnostics["selected_solver"],
    "selected_graph_strategy": diagnostics["selected_graph_strategy"],
    "selected_sparse_storage": diagnostics["selected_sparse_storage"],
    "retained_edges": edges,
    "estimated_working_memory_bytes": int(
        diagnostics["peak_working_memory_bytes"]
    ),
    "operator_graph_bytes": operator_graph_bytes,
    "iterations": int(diagnostics["iterations"]),
    "residual_norm": float(diagnostics["residual_norm"]),
    "converged": bool(diagnostics["converged"]),
    "quasi_sum": float(np.sum(quasi)),
    "quasi_l2": float(np.linalg.norm(quasi)),
}
if configuration.get("return_vector"):
    output["quasi_probabilities"] = quasi.tolist()
print(json.dumps(output, separators=(",", ":")))
"""


def _run_isolated_q64_memory_case(
    *,
    variant: str,
    qubits: int,
    matrix_size: int,
    clusters: int,
    cluster_radius: int,
    max_hamming_distance: int,
    seed: int,
    threads: int,
    memory_budget_bytes: int,
    relative_tolerance: float,
    gmres_restart: int,
    max_iterations: int,
    timeout_seconds: float,
    return_vector: bool = False,
) -> dict[str, Any]:
    """Run one Q=64+ memory case in a fresh process."""
    if variant not in Q64_MEMORY_VARIANT_LABELS:
        raise ValueError(f"Unknown Q=64+ memory variant: {variant}")
    configuration = {
        "variant": variant,
        "qubits": qubits,
        "matrix_size": matrix_size,
        "clusters": clusters,
        "cluster_radius": cluster_radius,
        "max_hamming_distance": max_hamming_distance,
        "seed": seed,
        "threads": threads,
        "memory_budget_bytes": memory_budget_bytes,
        "relative_tolerance": relative_tolerance,
        "gmres_restart": gmres_restart,
        "max_iterations": max_iterations,
        "return_vector": return_vector,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            Q64_MEMORY_WORKER,
            json.dumps(configuration),
        ],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant} at Q={qubits}, K={matrix_size} failed:\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    output_lines = [
        line for line in completed.stdout.splitlines() if line.strip()
    ]
    if not output_lines:
        raise RuntimeError(f"{variant} worker returned no JSON output")
    return json.loads(output_lines[-1])


def _append_q64_memory_row(
    output_path: Path,
    row: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Q64_MEMORY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                field: row.get(field, "")
                for field in Q64_MEMORY_FIELDNAMES
            }
        )


def load_q64_memory_rows(input_path: Path) -> list[dict[str, Any]]:
    """Load a Q=64+ sparse versus matrix-free memory report."""
    with input_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != Q64_MEMORY_FIELDNAMES:
            raise ValueError(
                f"Unexpected Q=64+ memory schema in {input_path}: "
                f"{reader.fieldnames!r}"
            )
        return list(reader)


def verify_q64_memory_variants(
    *,
    qubits: int = 65,
    matrix_size: int = 256,
    clusters: int = 2,
    cluster_radius: int = 2,
    max_hamming_distance: int = 4,
    seed: int = BENCHMARK_SEED,
    threads: int = 2,
    memory_budget_bytes: int = 6 * 1024**3,
    timeout_seconds: float = 600.0,
) -> float:
    """Verify sparse CSR and matrix-free on the first multiword path."""
    vectors = {}
    for variant in Q64_MEMORY_VARIANT_LABELS:
        result = _run_isolated_q64_memory_case(
            variant=variant,
            qubits=qubits,
            matrix_size=matrix_size,
            clusters=clusters,
            cluster_radius=cluster_radius,
            max_hamming_distance=max_hamming_distance,
            seed=seed,
            threads=threads,
            memory_budget_bytes=memory_budget_bytes,
            relative_tolerance=1e-8,
            gmres_restart=10,
            max_iterations=50,
            timeout_seconds=timeout_seconds,
            return_vector=True,
        )
        if not result["converged"]:
            raise RuntimeError(f"{variant} did not converge in verification")
        vectors[variant] = np.asarray(
            result["quasi_probabilities"],
            dtype=np.float64,
        )
    difference = float(
        np.max(
            np.abs(
                vectors["cpp_sparse_valued"]
                - vectors["cpp_matrix_free"]
            )
        )
    )
    if not np.allclose(
        vectors["cpp_sparse_valued"],
        vectors["cpp_matrix_free"],
        rtol=2e-6,
        atol=2e-8,
    ):
        raise AssertionError(
            "Q=64+ matrix-free disagrees with valued CSR: "
            f"max |difference| = {difference:.3e}"
        )
    return difference


def run_q64_memory_benchmark(
    *,
    qubit_counts: list[int],
    matrix_sizes: list[int],
    clusters: int,
    cluster_radius: int,
    max_hamming_distance: int,
    repeats: int,
    warmups: int,
    threads: int,
    memory_budget_bytes: int,
    output_path: Path,
    overwrite: bool = False,
    verify: bool = True,
    timeout_seconds: float = 600.0,
) -> list[dict[str, Any]]:
    """Measure valued CSR and matrix-free memory over stepped K values."""
    if not qubit_counts or min(qubit_counts) < 1:
        raise ValueError("qubit_counts must contain positive values")
    if not matrix_sizes or min(matrix_sizes) < 2:
        raise ValueError("matrix_sizes must contain values of at least two")
    if repeats < 1 or warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")
    if clusters != 2:
        raise ValueError("the Q=64+ memory benchmark requires two clusters")
    if any(matrix_size % clusters for matrix_size in matrix_sizes):
        raise ValueError("every matrix size must be divisible by clusters")
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_path} already exists; enable overwrite or change path"
            )
        output_path.unlink()
    if verify:
        difference = verify_q64_memory_variants(
            clusters=clusters,
            cluster_radius=cluster_radius,
            max_hamming_distance=max_hamming_distance,
            threads=min(threads, 2) if threads else 2,
            memory_budget_bytes=memory_budget_bytes,
            timeout_seconds=timeout_seconds,
        )
        print(f"Verification max |Sparse CSR - Matrix-free|: {difference:.3e}")

    variants = list(Q64_MEMORY_VARIANT_LABELS)
    rows: list[dict[str, Any]] = []
    for qubits in qubit_counts:
        words = (qubits + 63) // 64
        for matrix_size in matrix_sizes:
            case_seed = BENCHMARK_SEED + 100 * qubits + matrix_size
            for variant in variants:
                label = Q64_MEMORY_VARIANT_LABELS[variant]
                print(
                    f"\n{label}, Q={qubits}, K={matrix_size}, "
                    f"{words} word(s)/state",
                    flush=True,
                )
                for warmup in range(warmups):
                    print(
                        f"  isolated warmup {warmup + 1}/{warmups}",
                        flush=True,
                    )
                    _run_isolated_q64_memory_case(
                        variant=variant,
                        qubits=qubits,
                        matrix_size=matrix_size,
                        clusters=clusters,
                        cluster_radius=cluster_radius,
                        max_hamming_distance=max_hamming_distance,
                        seed=case_seed,
                        threads=threads,
                        memory_budget_bytes=memory_budget_bytes,
                        relative_tolerance=1e-7,
                        gmres_restart=10,
                        max_iterations=50,
                        timeout_seconds=timeout_seconds,
                    )
                for repetition in range(1, repeats + 1):
                    print(
                        f"  measured run {repetition}/{repeats}",
                        flush=True,
                    )
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "suite": "q64plus_memory_scaling",
                        "variant": variant,
                        "label": label,
                        "qubits": qubits,
                        "words_per_state": words,
                        "matrix_size": matrix_size,
                        "clusters": clusters,
                        "cluster_radius": cluster_radius,
                        "max_hamming_distance": max_hamming_distance,
                        "repetition": repetition,
                        "status": "ok",
                        "error": "",
                    }
                    try:
                        result = _run_isolated_q64_memory_case(
                            variant=variant,
                            qubits=qubits,
                            matrix_size=matrix_size,
                            clusters=clusters,
                            cluster_radius=cluster_radius,
                            max_hamming_distance=max_hamming_distance,
                            seed=case_seed,
                            threads=threads,
                            memory_budget_bytes=memory_budget_bytes,
                            relative_tolerance=1e-7,
                            gmres_restart=10,
                            max_iterations=50,
                            timeout_seconds=timeout_seconds,
                        )
                        row.update(result)
                        if not result["converged"]:
                            row["status"] = "failed"
                            row["error"] = "iterative solver did not converge"
                        print(
                            f"    peak RSS={result['peak_rss_mib']:.1f} MiB, "
                            f"working set="
                            f"{result['estimated_working_memory_bytes'] / 1024**2:.1f} MiB, "
                            f"wall={result['wall_seconds']:.4f}s",
                            flush=True,
                        )
                    except Exception as error:
                        row["status"] = "failed"
                        row["error"] = f"{type(error).__name__}: {error}"
                        print(f"    FAILED: {row['error']}", flush=True)
                    rows.append(row)
                    _append_q64_memory_row(output_path, row)
    print(f"\nQ=64+ memory report saved to: {output_path.resolve()}")
    return rows


def _q64_memory_series(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    divisor: float = 1.0,
) -> dict[tuple[int, str], list[tuple[int, float, float, float]]]:
    grouped: dict[tuple[int, str, int], list[float]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        try:
            qubits = int(row["qubits"])
            matrix_size = int(row["matrix_size"])
            value = float(row[metric]) / divisor
        except (KeyError, TypeError, ValueError):
            continue
        if (
            qubits <= 0
            or matrix_size <= 0
            or not math.isfinite(value)
            or value <= 0
        ):
            continue
        grouped.setdefault(
            (qubits, str(row["variant"]), matrix_size),
            [],
        ).append(value)
    output: dict[
        tuple[int, str],
        list[tuple[int, float, float, float]],
    ] = {}
    for (qubits, variant, matrix_size), values in grouped.items():
        output.setdefault((qubits, variant), []).append(
            (
                matrix_size,
                statistics.median(values),
                min(values),
                max(values),
            )
        )
    for points in output.values():
        points.sort(key=lambda point: point[0])
    return output


def plot_q64_memory(
    rows: list[dict[str, Any]],
    *,
    output_directory: Path | None = None,
    filename_prefix: str = "q64plus_matrix_free",
) -> tuple[Any, Any]:
    """Plot measured process memory and C++ working-set estimates."""
    from matplotlib import pyplot as plt

    peak_rss = _q64_memory_series(rows, "peak_rss_mib")
    working_set = _q64_memory_series(
        rows,
        "estimated_working_memory_bytes",
        divisor=1024**2,
    )
    if not peak_rss or not working_set:
        raise ValueError("No successful Q=64+ memory samples are available")

    styles = {
        "cpp_sparse_valued": ("#54A24B", "^"),
        "cpp_matrix_free": ("#B279A2", "P"),
    }
    qubit_counts = sorted(
        {
            int(row["qubits"])
            for row in rows
            if row.get("status") == "ok"
        }
    )

    def draw(
        series: dict[
            tuple[int, str],
            list[tuple[int, float, float, float]],
        ],
        *,
        ylabel: str,
        title: str,
    ):
        figure, axes = plt.subplots(
            1,
            len(qubit_counts),
            figsize=(6.3 * len(qubit_counts), 5.4),
            sharey=True,
            squeeze=False,
        )
        for panel, qubits in enumerate(qubit_counts):
            axis = axes[0, panel]
            sizes: set[int] = set()
            for variant, label in Q64_MEMORY_VARIANT_LABELS.items():
                points = series.get((qubits, variant))
                if not points:
                    continue
                color, marker = styles[variant]
                x = np.asarray(
                    [point[0] for point in points],
                    dtype=float,
                )
                medians = np.asarray(
                    [point[1] for point in points],
                    dtype=float,
                )
                minima = np.asarray(
                    [point[2] for point in points],
                    dtype=float,
                )
                maxima = np.asarray(
                    [point[3] for point in points],
                    dtype=float,
                )
                sizes.update(int(size) for size in x)
                axis.plot(
                    x,
                    medians,
                    color=color,
                    marker=marker,
                    linewidth=2.2,
                    markersize=7,
                    label=label,
                )
                axis.fill_between(
                    x,
                    minima,
                    maxima,
                    color=color,
                    alpha=0.14,
                    linewidth=0,
                )
            ordered_sizes = sorted(sizes)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xticks(ordered_sizes)
            axis.set_xticklabels(
                [
                    rf"$2^{{{int(math.log2(size))}}}$"
                    if size > 0 and size & (size - 1) == 0
                    else f"{size:,}"
                    for size in ordered_sizes
                ]
            )
            axis.set_xlabel("Number of observed states, K")
            if panel == 0:
                axis.set_ylabel(ylabel)
                axis.legend(frameon=False)
            words = (qubits + 63) // 64
            axis.set_title(
                f"Q={qubits}, {words} word"
                f"{'s' if words != 1 else ''}/state",
                loc="left",
                fontsize=11,
            )
            axis.grid(True, which="major", alpha=0.28)
            axis.grid(True, which="minor", alpha=0.10)
            axis.spines[["top", "right"]].set_visible(False)
        figure.suptitle(title, x=0.01, ha="left", fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        return figure

    peak_figure = draw(
        peak_rss,
        ylabel="Peak RSS, MiB",
        title="Process memory scaling at Q=64+",
    )
    working_figure = draw(
        working_set,
        ylabel="Estimated working set, MiB",
        title="C++ working-set scaling at Q=64+",
    )
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        peak_path = output_directory / f"{filename_prefix}_peak_rss.svg"
        working_path = (
            output_directory / f"{filename_prefix}_working_set.svg"
        )
        peak_figure.savefig(peak_path, format="svg", bbox_inches="tight")
        working_figure.savefig(
            working_path,
            format="svg",
            bbox_inches="tight",
        )
        print(f"Peak-RSS plot saved to: {peak_path.resolve()}")
        print(f"Working-set plot saved to: {working_path.resolve()}")

    plt.close(peak_figure)
    plt.close(working_figure)
    return peak_figure, working_figure


def print_q64_memory_summary(rows: list[dict[str, Any]]) -> None:
    """Print median memory and runtime trade-offs by Q and K."""
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            grouped.setdefault(
                (
                    int(row["qubits"]),
                    int(row["matrix_size"]),
                    str(row["variant"]),
                ),
                [],
            ).append(row)
    print(
        f"{'Q':>4} {'K':>7} {'words':>5} {'edges':>12} "
        f"{'CSR RSS':>10} {'MF RSS':>10} "
        f"{'CSR set':>10} {'MF set':>10} "
        f"{'set ratio':>10} {'MF/CSR time':>12}"
    )
    for qubits in sorted({key[0] for key in grouped}):
        matrix_sizes = sorted(
            {key[1] for key in grouped if key[0] == qubits}
        )
        for matrix_size in matrix_sizes:
            sparse = grouped.get(
                (qubits, matrix_size, "cpp_sparse_valued")
            )
            matrix_free = grouped.get(
                (qubits, matrix_size, "cpp_matrix_free")
            )
            if not sparse or not matrix_free:
                continue

            def median(group: list[dict[str, Any]], field: str) -> float:
                return statistics.median(
                    float(row[field]) for row in group
                )

            sparse_rss = median(sparse, "peak_rss_mib")
            matrix_free_rss = median(matrix_free, "peak_rss_mib")
            sparse_set = (
                median(sparse, "estimated_working_memory_bytes") / 1024**2
            )
            matrix_free_set = (
                median(matrix_free, "estimated_working_memory_bytes")
                / 1024**2
            )
            time_ratio = (
                median(matrix_free, "wall_seconds")
                / median(sparse, "wall_seconds")
            )
            print(
                f"{qubits:>4} {matrix_size:>7,} "
                f"{(qubits + 63) // 64:>5} "
                f"{int(median(sparse, 'retained_edges')):>12,} "
                f"{sparse_rss:>9.1f}M {matrix_free_rss:>9.1f}M "
                f"{sparse_set:>9.1f}M {matrix_free_set:>9.1f}M "
                f"{sparse_set / matrix_free_set:>9.1f}x "
                f"{time_ratio:>11.2f}x"
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("end-to-end", "local-core"),
        default="end-to-end",
        help=(
            "Local full pipeline, or exact-input Python/C++ core comparison."
        ),
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help="Requested reduced-matrix sizes.",
    )
    parser.add_argument(
        "--powers",
        type=int,
        nargs=2,
        metavar=("MIN_POWER", "MAX_POWER"),
        help="Use every inclusive power of two from 2**MIN_POWER to 2**MAX_POWER.",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        default=None,
        help=(
            "Run one deep GHZ benchmark with this many qubits instead of the "
            "state-space scaling suite."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=30,
        help="Target circuit depth for --qubits; includes measurement.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10_000,
        help="Fixed shot count for --qubits deep-circuit mode.",
    )
    parser.add_argument(
        "--implementations",
        choices=tuple(RUNNERS),
        nargs="+",
        default=["python", "cpp"],
        help="Local implementations to benchmark.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shots-per-state", type=int, default=10)
    parser.add_argument(
        "--max-hamming-distance",
        type=int,
        default=3,
        help="Hamming cutoff for --scope local-core.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit CSV path. Normally generated automatically.",
    )
    parser.add_argument(
        "--reports-directory",
        type=Path,
        default=DEFAULT_REPORTS_DIRECTORY,
        help="Directory for automatically indexed benchmark reports.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save separate log-log runtime and peak-RSS plots (end-to-end only).",
    )
    parser.add_argument(
        "--plot-directory",
        type=Path,
        default=None,
        help="Plot destination; defaults to the CSV report directory.",
    )
    arguments = parser.parse_args()

    if arguments.warmups < 0:
        parser.error("--warmups cannot be negative")
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    if arguments.shots_per_state < 1:
        parser.error("--shots-per-state must be at least 1")
    if arguments.shots < 1:
        parser.error("--shots must be at least 1")
    if arguments.max_hamming_distance < 0:
        parser.error("--max-hamming-distance cannot be negative")
    selected_size_modes = sum(
        option is not None
        for option in (arguments.sizes, arguments.powers, arguments.qubits)
    )
    if selected_size_modes > 1:
        parser.error("--sizes, --powers, and --qubits are mutually exclusive")
    if arguments.qubits is not None:
        if arguments.scope != "end-to-end":
            parser.error("--qubits currently requires --scope end-to-end")
        if arguments.qubits < 2:
            parser.error("--qubits must be at least 2")
        if arguments.depth < arguments.qubits + 1:
            parser.error(
                "--depth must be at least --qubits + 1 for the GHZ circuit"
            )
        arguments.sizes = None
    elif arguments.powers is not None:
        minimum_power, maximum_power = arguments.powers
        if minimum_power < 0 or maximum_power < minimum_power:
            parser.error("--powers requires 0 <= MIN_POWER <= MAX_POWER")
        if maximum_power > 62:
            parser.error("--powers currently supports MAX_POWER <= 62")
        arguments.sizes = [
            1 << power for power in range(minimum_power, maximum_power + 1)
        ]
    elif arguments.sizes is None:
        arguments.sizes = list(DEFAULT_SIZES)
    if arguments.plot and arguments.scope != "end-to-end":
        parser.error("--plot currently requires --scope end-to-end")
    if arguments.plot and arguments.qubits is not None:
        parser.error("--plot is for state-space scaling, not --qubits mode")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    output_path = arguments.output
    if output_path is None:
        output_path = next_report_path(arguments.reports_directory)

    if arguments.scope == "local-core":
        rows: list[dict[str, Any]] = []
        for requested_states in arguments.sizes:
            n_qubits, possible_states = state_space(requested_states)
            rows.extend(
                benchmark_local_core(
                    n_qubits=n_qubits,
                    unique_states=min(requested_states, possible_states),
                    max_hamming_distance=arguments.max_hamming_distance,
                    warmups=arguments.warmups,
                    repeats=arguments.repeats,
                )
            )
        for row in rows:
            append_row(output_path, row)
        print_summary(rows)
        print(f"\nResults saved to: {output_path.resolve()}")
        return


    rows = run_benchmark(
        sizes=arguments.sizes,
        implementations=arguments.implementations,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        shots_per_state=arguments.shots_per_state,
        output_path=output_path,
        qubits=arguments.qubits,
        target_depth=arguments.depth,
        fixed_shots=arguments.shots,
    )
    print_summary(rows)
    if arguments.plot:
        plot_scaling(
            rows,
            output_directory=arguments.plot_directory or output_path.parent,
            filename_prefix=output_path.stem,
        )


if __name__ == "__main__":
    main()