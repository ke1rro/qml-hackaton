"""Python/C++ equivalence tests without requiring pytest."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = Path(__file__).resolve().parents[1] / "build-python"
sys.path.insert(0, str(BUILD))
sys.path.insert(0, str(ROOT))

import rem_cpp  # noqa: E402
from sparse_rem import _mitigate_probs  # noqa: E402


def random_calibrations(qubits: int, rng: np.random.Generator) -> np.ndarray:
    cals = np.empty((qubits, 2, 2), dtype=np.float64)
    p10 = rng.uniform(0.005, 0.08, size=qubits)
    p01 = rng.uniform(0.005, 0.08, size=qubits)
    cals[:, 0, 0] = 1.0 - p10
    cals[:, 1, 0] = p10
    cals[:, 0, 1] = p01
    cals[:, 1, 1] = 1.0 - p01
    return cals


def bitstrings(states: np.ndarray, qubits: int) -> list[str]:
    return [format(int(value), f"0{qubits}b") for value in states]


class TestPythonEquivalence(unittest.TestCase):
    def test_small_random_dense(self):
        rng = np.random.default_rng(913)
        for qubits in (1, 2, 3, 4, 8):
            universe = 1 << qubits
            count = min(universe, 3 * qubits + 2)
            states = np.sort(rng.choice(universe, size=count, replace=False))
            strings = bitstrings(states, qubits)
            raw_values = rng.random(count)
            raw_values /= raw_values.sum()
            raw = dict(zip(strings, raw_values))
            cals = random_calibrations(qubits, rng)
            python_quasi = _mitigate_probs(
                raw,
                qubits,
                list(range(qubits)),
                cals,
                matrix_format="dense",
                backend_impl="python",
                project_to_probability_simplex=False,
                d=min(3, qubits),
            )
            cpp_quasi, _, details = _mitigate_probs(
                raw,
                qubits,
                list(range(qubits)),
                cals,
                matrix_format="dense",
                backend_impl="cpp",
                project_to_probability_simplex=False,
                return_details=True,
                d=min(3, qubits),
                threads=2,
            )
            self.assertTrue(details["converged"])
            np.testing.assert_allclose(
                [cpp_quasi[state] for state in strings],
                [python_quasi[state] for state in strings],
                rtol=2e-12,
                atol=2e-13,
            )

    def test_sparse_and_matrix_free_match_dense(self):
        rng = np.random.default_rng(1776)
        qubits = 12
        states = np.sort(rng.choice(1 << qubits, size=180, replace=False))
        strings = bitstrings(states, qubits)
        raw_values = rng.random(len(strings))
        raw_values /= raw_values.sum()
        raw = dict(zip(strings, raw_values))
        cals = random_calibrations(qubits, rng)
        dense, _, _ = _mitigate_probs(
            raw,
            qubits,
            list(range(qubits)),
            cals,
            matrix_format="dense",
            backend_impl="cpp",
            project_to_probability_simplex=False,
            return_details=True,
            d=3,
        )
        packed = states.astype(np.uint64)
        raw_array = np.ascontiguousarray(raw_values, dtype=np.float64)
        for backend in ("explicit_sparse", "matrix_free"):
            output = rem_cpp.mitigate_packed(
                packed,
                raw_array,
                np.ascontiguousarray(cals),
                qubits,
                {
                    "backend": backend,
                    "solver": "gmres",
                    "max_hamming_distance": 3,
                    "relative_tolerance": 1e-10,
                    "max_iterations": 100,
                    "threads": 2,
                    "cache_values": True,
                },
            )
            self.assertTrue(output["diagnostics"]["converged"])
            np.testing.assert_allclose(
                output["quasi_probabilities"],
                [dense[state] for state in strings],
                rtol=2e-8,
                atol=2e-10,
            )

    def test_physical_mapping_and_lsb_order(self):
        rng = np.random.default_rng(42)
        all_cals = random_calibrations(7, rng)
        measured = [5, 1, 6]
        strings = ["000", "001", "010", "011", "100", "111"]
        raw_values = np.array([0.05, 0.15, 0.2, 0.1, 0.25, 0.25])
        raw = dict(zip(strings, raw_values))
        python_quasi = _mitigate_probs(
            raw,
            3,
            measured,
            all_cals,
            matrix_format="dense",
            backend_impl="python",
            project_to_probability_simplex=False,
            d=3,
        )
        cpp_quasi = _mitigate_probs(
            raw,
            3,
            measured,
            all_cals,
            matrix_format="dense",
            backend_impl="cpp",
            project_to_probability_simplex=False,
            d=3,
        )
        np.testing.assert_allclose(
            [cpp_quasi[state] for state in strings],
            [python_quasi[state] for state in strings],
            rtol=2e-12,
            atol=2e-13,
        )

    def test_zero_diagonal_fallback_and_projection(self):
        states = np.array([0, 1], dtype=np.uint64)
        raw = np.array([0.1, 0.9], dtype=np.float64)
        cals = np.array([[[0.0, 0.1], [1.0, 0.9]]], dtype=np.float64)
        output = rem_cpp.mitigate_packed(
            states,
            raw,
            cals,
            1,
            {
                "backend": "dense",
                "solver": "direct",
                "max_hamming_distance": 1,
                "project_to_probability_simplex": True,
            },
        )
        self.assertTrue(output["diagnostics"]["used_safe_element_fallback"])
        self.assertTrue(output["diagnostics"]["jacobi_had_zero_diagonal"])
        self.assertAlmostEqual(float(np.sum(output["projected_probabilities"])), 1.0)
        self.assertTrue(np.all(output["projected_probabilities"] >= 0.0))

    def test_no_hidden_forcecast(self):
        states = np.arange(4, dtype=np.int64)
        raw = np.full(4, 0.25, dtype=np.float64)
        cals = random_calibrations(2, np.random.default_rng(3))
        with self.assertRaises(TypeError):
            rem_cpp.mitigate_packed(states, raw, cals, 2, {})
        with self.assertRaises(TypeError):
            rem_cpp.mitigate_packed(
                states.astype(np.uint64), raw.astype(np.float32), cals, 2, {}
            )

    def test_batched_binding(self):
        states = np.arange(4, dtype=np.uint64)
        probabilities = np.ascontiguousarray(
            [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
            dtype=np.float64,
        )
        cals = random_calibrations(2, np.random.default_rng(91))
        output = rem_cpp.mitigate_packed_many(
            states,
            probabilities,
            np.ascontiguousarray(cals),
            2,
            {
                "backend": "dense",
                "solver": "direct",
                "max_hamming_distance": 2,
            },
        )
        self.assertEqual(len(output), 2)
        self.assertTrue(all(item["diagnostics"]["converged"] for item in output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
