"""End-to-end benchmark for the original sparse REM and Haiqu SDK.

This script benchmarks complete mitigation pipelines, not the isolated matrix
solver. It supports both the original state-space-size sweep and a fixed-qubit,
fixed-depth circuit mode for larger experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qiskit import QuantumCircuit
from qiskit.quantum_info import hellinger_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError

from haiqu_rem import measure_call, mitigate_readout_errors_haiqu
from sparse_rem import mitigate_readout_errors

DEFAULT_SIZES = [250, 500, 1_000, 2_000, 4_000, 8_000]
DEFAULT_REPORTS_DIRECTORY = Path("benchmarks_reports")
FIELDNAMES = [
    "timestamp",
    "implementation",
    "benchmark_mode",
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
    "cpu_seconds",
    "baseline_rss_mib",
    "peak_rss_mib",
    "peak_rss_delta_mib",
    "haiqu_job_id",
    "device_time_seconds",
    "pre_device_pipeline_seconds",
    "raw_fidelity",
    "mitigated_fidelity",
    "status",
    "error",
]


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

    Random RZ and alternating CZ layers increase circuit depth without changing
    the ideal GHZ measurement probabilities. ``target_depth`` includes the
    final measurement layer.
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

        start_qubit = layer % 2
        for qubit in range(start_qubit, n_qubits - 1, 2):
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
    """Return the exact measurement distribution of an ideal GHZ state."""
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


def run_original(
    circuit: QuantumCircuit,
    backend: AerSimulator,
    noise_model: NoiseModel,
    n_qubits: int,
    shots: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run and profile the complete original Python mitigation pipeline."""
    del noise_model  # The local backend already owns this model.
    return measure_call(
        mitigate_readout_errors,
        circuit,
        backend,
        shots=shots,
        local=True,
        matrix_format="auto",
        n_qubits=n_qubits,
    )


def run_haiqu(
    circuit: QuantumCircuit,
    backend: AerSimulator,
    noise_model: NoiseModel,
    n_qubits: int,
    shots: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Run and profile the complete Haiqu readout-mitigation pipeline."""
    del backend, n_qubits  # Haiqu owns the cloud backend and transpilation.
    output = mitigate_readout_errors_haiqu(
        circuit=circuit,
        shots=shots,
        device_id="aer_simulator",
        noise_model=noise_model,
        device_options=(
            {"method": "matrix_product_state"}
            if circuit.num_qubits > 12
            else None
        ),
        include_raw=False,
        job_name=(
            f"rem-benchmark-{circuit.num_qubits}q-"
            f"d{circuit.depth()}"
        ),
    )
    return output, output["metrics"]["mitigated"]


RUNNERS = {
    "original": run_original,
    "haiqu": run_haiqu,
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
    """Convert one successful measured run to a CSV row."""
    mitigated = output["mitigated"]
    raw = output.get("raw") or {}
    haiqu_job = (output.get("haiqu_jobs") or {}).get("mitigated") or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "implementation": implementation,
        "benchmark_mode": benchmark_mode,
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
        **metrics,
        "haiqu_job_id": haiqu_job.get("job_id") or haiqu_job.get("id", ""),
        "device_time_seconds": haiqu_job.get("time", ""),
        "pre_device_pipeline_seconds": haiqu_job.get(
            "pre_device_pipeline_time",
            "",
        ),
        "raw_fidelity": hellinger_fidelity(ideal, raw) if raw else "",
        "mitigated_fidelity": hellinger_fidelity(ideal, mitigated),
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
            "implementation": implementation,
            "benchmark_mode": benchmark_mode,
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
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    qubits: int | None,
    target_depth: int,
    fixed_shots: int,
    implementations: list[str],
    warmups: int,
    repeats: int,
    shots_per_state: int,
    haiqu_cooldown_seconds: float,
    output_path: Path,
) -> None:
    """Run warmups and measured repetitions for every requested configuration."""
    noise_model = make_readout_noise_model()
    local_backend = AerSimulator(
        method="matrix_product_state",
        noise_model=noise_model,
    )

    cases: list[dict[str, Any]] = []
    if qubits is not None:
        circuit = make_deep_ghz_circuit(qubits, target_depth)
        cases.append(
            {
                "benchmark_mode": "qubits_depth",
                "requested_states": "",
                "possible_states": 2**qubits,
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
            circuit = make_uniform_circuit(n_qubits)
            cases.append(
                {
                    "benchmark_mode": "state_space",
                    "requested_states": requested_states,
                    "possible_states": possible_states,
                    "n_qubits": n_qubits,
                    "target_depth": "",
                    "shots": possible_states * shots_per_state,
                    "circuit": circuit,
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
                f"\nQubit/depth mode: {n_qubits} qubits, "
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
            simulator_method = (
                "matrix_product_state"
                if implementation == "original" or n_qubits > 12
                else "automatic"
            )
            print(f"  {implementation}: warmup", flush=True)

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
                finally:
                    if implementation == "haiqu" and haiqu_cooldown_seconds:
                        print(
                            f"    Haiqu cooldown: "
                            f"{haiqu_cooldown_seconds:g}s",
                            flush=True,
                        )
                        time.sleep(haiqu_cooldown_seconds)

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
                    device_time = row["device_time_seconds"]
                    device_text = (
                        f", device={device_time}s"
                        if device_time != ""
                        else ""
                    )
                    print(
                        f"      wall={row['wall_seconds']:.3f}s"
                        f"{device_text}, "
                        f"peak={row['peak_rss_mib']:.1f}MiB, "
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

                append_row(output_path, row)
                if implementation == "haiqu" and haiqu_cooldown_seconds:
                    print(
                        f"    Haiqu cooldown: {haiqu_cooldown_seconds:g}s",
                        flush=True,
                    )
                    time.sleep(haiqu_cooldown_seconds)

    print(f"\nResults saved to: {output_path.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    benchmark_mode = parser.add_mutually_exclusive_group()
    benchmark_mode.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Requested reduced-matrix sizes. Uses the default size sweep when "
            "neither --sizes nor --qubits is provided."
        ),
    )
    benchmark_mode.add_argument(
        "--qubits",
        type=int,
        default=None,
        help="Use a fixed-qubit deep GHZ benchmark instead of the size sweep.",
    )
    parser.add_argument(
        "--implementations",
        choices=tuple(RUNNERS),
        nargs="+",
        default=list(RUNNERS),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shots-per-state", type=int, default=10)
    parser.add_argument(
        "--depth",
        type=int,
        default=30,
        help=(
            "Target circuit depth, including measurements, in --qubits mode."
        ),
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=10_000,
        help="Fixed shot count in --qubits mode.",
    )
    parser.add_argument(
        "--haiqu-cooldown",
        type=float,
        default=15.0,
        help="Seconds to wait after each Haiqu job; excluded from wall time.",
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
    arguments = parser.parse_args()

    if arguments.warmups < 0:
        parser.error("--warmups cannot be negative")
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    if arguments.shots_per_state < 1:
        parser.error("--shots-per-state must be at least 1")
    if arguments.shots < 1:
        parser.error("--shots must be at least 1")
    if arguments.qubits is not None:
        if arguments.qubits < 2:
            parser.error("--qubits must be at least 2")
        minimum_depth = arguments.qubits + 1
        if arguments.depth < minimum_depth:
            parser.error(
                f"--depth must be at least {minimum_depth} for "
                f"{arguments.qubits} qubits"
            )
    if arguments.haiqu_cooldown < 0:
        parser.error("--haiqu-cooldown cannot be negative")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    load_dotenv()

    if "haiqu" in arguments.implementations:
        if not os.getenv("HAIQU_API_KEY"):
            raise RuntimeError("HAIQU_API_KEY was not found in .env")
        from haiqu.sdk import haiqu

        print(haiqu.login(raise_on_error=True))
        print(
            haiqu.init(
                "Readout mitigation benchmark",
                experiment_description=(
                    "End-to-end original sparse REM and Haiqu SDK comparison"
                ),
                log_source_code=False,
            )
        )

    output_path = arguments.output
    if output_path is None:
        output_path = next_report_path(arguments.reports_directory)

    run_benchmark(
        sizes=arguments.sizes,
        qubits=arguments.qubits,
        target_depth=arguments.depth,
        fixed_shots=arguments.shots,
        implementations=arguments.implementations,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        shots_per_state=arguments.shots_per_state,
        haiqu_cooldown_seconds=arguments.haiqu_cooldown,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()