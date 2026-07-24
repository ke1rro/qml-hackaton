"""End-to-end benchmark for the original sparse REM and Haiqu SDK.

This script benchmarks complete mitigation pipelines, not the isolated matrix
solver. It uses the same Qiskit circuits, readout noise model, shot counts, and
target state-space sizes for both implementations.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
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
    "requested_states",
    "possible_states",
    "actual_states",
    "n_qubits",
    "shots",
    "repetition",
    "wall_seconds",
    "cpu_seconds",
    "baseline_rss_mib",
    "peak_rss_mib",
    "peak_rss_delta_mib",
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


def ideal_uniform_distribution(n_qubits: int) -> dict[str, float]:
    """Return the exact uniform distribution over all computational states."""
    states_count = 2**n_qubits
    probability = 1.0 / states_count
    return {
        format(index, f"0{n_qubits}b"): probability
        for index in range(states_count)
    }


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
        job_name=f"rem-benchmark-{circuit.num_qubits}q",
    )
    return output, output["metrics"]["mitigated"]


RUNNERS = {
    "original": run_original,
    "haiqu": run_haiqu,
}


def result_row(
    *,
    implementation: str,
    requested_states: int,
    possible_states: int,
    n_qubits: int,
    shots: int,
    repetition: int,
    output: dict[str, Any],
    metrics: dict[str, float],
    ideal: dict[str, float],
) -> dict[str, Any]:
    """Convert one successful measured run to a CSV row."""
    mitigated = output["mitigated"]
    raw = output.get("raw") or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "implementation": implementation,
        "requested_states": requested_states,
        "possible_states": possible_states,
        "actual_states": len(raw) if raw else len(mitigated),
        "n_qubits": n_qubits,
        "shots": shots,
        "repetition": repetition,
        **metrics,
        "raw_fidelity": hellinger_fidelity(ideal, raw) if raw else "",
        "mitigated_fidelity": hellinger_fidelity(ideal, mitigated),
        "status": "ok",
        "error": "",
    }


def failure_row(
    *,
    implementation: str,
    requested_states: int,
    possible_states: int,
    n_qubits: int,
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
            "requested_states": requested_states,
            "possible_states": possible_states,
            "n_qubits": n_qubits,
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
    sizes: list[int],
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

    for requested_states in sizes:
        n_qubits, possible_states = state_space(requested_states)
        shots = possible_states * shots_per_state
        circuit = make_uniform_circuit(n_qubits)
        ideal = ideal_uniform_distribution(n_qubits)

        print(
            f"\nSize {requested_states}: {n_qubits} qubits, "
            f"{possible_states} possible states, {shots} shots"
        )

        for implementation in implementations:
            runner = RUNNERS[implementation]
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
                        requested_states=requested_states,
                        possible_states=possible_states,
                        n_qubits=n_qubits,
                        shots=shots,
                        repetition=repetition,
                        output=output,
                        metrics=metrics,
                        ideal=ideal,
                    )
                    print(
                        f"      wall={row['wall_seconds']:.3f}s, "
                        f"peak={row['peak_rss_mib']:.1f}MiB, "
                        f"fidelity={row['mitigated_fidelity']:.4f}",
                        flush=True,
                    )
                except Exception as error:
                    row = failure_row(
                        implementation=implementation,
                        requested_states=requested_states,
                        possible_states=possible_states,
                        n_qubits=n_qubits,
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
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Requested reduced-matrix sizes.",
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
        implementations=arguments.implementations,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
        shots_per_state=arguments.shots_per_state,
        haiqu_cooldown_seconds=arguments.haiqu_cooldown,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()