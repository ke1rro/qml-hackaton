import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(__file__).resolve().parents[1] / "build-python"
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(ROOT))

from sparse_rem import _mitigate_probs  # noqa: E402


def make_problem(qubits: int, count: int, seed: int):
    if qubits > 63:
        raise ValueError("this benchmark generator currently supports Q <= 63")
    rng = np.random.default_rng(seed)
    count = min(count, 1 << qubits)
    states = np.sort(rng.choice(1 << qubits, size=count, replace=False))
    strings = [format(int(state), f"0{qubits}b") for state in states]
    raw_values = rng.random(count)
    raw_values /= raw_values.sum()
    raw = dict(zip(strings, raw_values))
    cals = np.empty((qubits, 2, 2), dtype=np.float64)
    p10 = rng.uniform(0.01, 0.04, size=qubits)
    p01 = rng.uniform(0.01, 0.04, size=qubits)
    cals[:, 0, 0] = 1 - p10
    cals[:, 1, 0] = p10
    cals[:, 0, 1] = p01
    cals[:, 1, 1] = 1 - p01
    return raw, cals


def measure(function, repetitions: int):
    function()
    samples = []
    last = None
    for _ in range(repetitions):
        start = time.perf_counter()
        last = function()
        samples.append(time.perf_counter() - start)
    return {
        "minimum_s": min(samples),
        "median_s": statistics.median(samples),
        "maximum_s": max(samples),
        "stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }, last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, default=16)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    raw, cals = make_problem(args.q, args.k, args.seed)
    measured = list(range(args.q))

    def python_run():
        return _mitigate_probs(
            raw,
            args.q,
            measured,
            cals,
            matrix_format="auto",
            backend_impl="python",
            project_to_probability_simplex=False,
            solver_tol=1e-7,
            d=args.d,
        )

    def cpp_run():
        return _mitigate_probs(
            raw,
            args.q,
            measured,
            cals,
            matrix_format="auto",
            backend_impl="cpp",
            project_to_probability_simplex=False,
            solver_tol=1e-7,
            return_details=True,
            d=args.d,
            threads=args.threads,
        )

    python_timings, python_result = measure(python_run, args.repetitions)
    cpp_timings, cpp_output = measure(cpp_run, args.repetitions)
    cpp_result, _, cpp_details = cpp_output
    keys = list(raw)
    difference = np.asarray([python_result[key] - cpp_result[key] for key in keys])
    result = {
        "q": args.q,
        "k": len(raw),
        "d": args.d,
        "repetitions": args.repetitions,
        "python": python_timings,
        "cpp": cpp_timings,
        "speedup_median": python_timings["median_s"] / cpp_timings["median_s"],
        "max_abs_quasi_difference": float(np.max(np.abs(difference))),
        "l2_quasi_difference": float(np.linalg.norm(difference)),
        "cpp_backend": cpp_details["selected_backend"],
        "cpp_solver": cpp_details["selected_solver"],
        "cpp_edges": cpp_details["retained_edges"],
        "cpp_iterations": cpp_details["iterations"],
        "cpp_residual": cpp_details["residual_norm"],
        "cpp_estimated_memory_bytes": cpp_details["estimated_memory_bytes"],
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
