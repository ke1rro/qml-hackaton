# High-Performance Readout Error Mitigation

High-performance C++ implementation of scalable **readout-error mitigation (REM)**
for quantum computing.

The project was developed during the **Haiqu Hackathon at the UCU Quantum Machine
Learning School 2026**, where it received **2nd place**.

The implementation is based on the scalable REM approach from:

> P. D. Nation, H. Kang, N. Sundaresan, J. M. Gambetta,
> **Scalable Mitigation of Measurement Errors on Quantum Computers**,
> PRX Quantum 2, 040326 (2021).
> https://doi.org/10.1103/PRXQuantum.2.040326

## Implementation

The mitigation problem is solved in the reduced space of observed bitstrings rather
than the full `2^Q` state space.[^nation2021]

The C++ core provides three backends:

- **Dense** — explicit assignment matrix + direct LAPACK solve
- **Sparse CSR** — Hamming-truncated matrix + iterative solver
- **Matrix-free** — evaluates the assignment operator on demand without storing the matrix

Additional features:

- restarted GMRES with Jacobi preconditioning
- BiCGSTAB support
- Hamming-ball and blocked-pairwise sparse topology construction
- packed bitstrings with support for more than 64 measured qubits
- parallel C++ operator construction and matrix-vector products
- automatic backend selection based on problem structure and memory budget
- Python bindings and Qiskit integration

## Results

### Readout-error mitigation

4-qubit GHZ state:

| | Hellinger fidelity |
|---|---:|
| Noisy | 0.8247 |
| Mitigated | 0.9828 |

Partial measurement of 3 out of 4 qubits:

| | Hellinger fidelity |
|---|---:|
| Noisy | 0.8535 |
| Mitigated | 1.0000 |

### C++ vs Python

For a 16-qubit problem with 1,000 observed states and Hamming cutoff `d=3`:

- Python: **7.985 ms**
- C++: **2.830 ms**
- Speedup: **2.82×**
- Maximum numerical difference: `1.15e-9`

### Matrix-free memory scaling

For `Q=65`, `K=32,768` observed states:

| Backend | Estimated working set |
|---|---:|
| Sparse CSR | 6153.5 MiB |
| Matrix-free | 7.0 MiB |

The matrix-free backend uses approximately **879× less working memory** in this
high-density stress test, at approximately **1.86× higher runtime**.

## Requirements

- C++20
- CMake
- Python
- NumPy / SciPy
- Qiskit
- pybind11
- BLAS / LAPACK

## Authors

- Nikita Lenyk
- Yurii Barchyshyn

## References

[^nation2021]: P. D. Nation, H. Kang, N. Sundaresan, and J. M. Gambetta,
    “Scalable Mitigation of Measurement Errors on Quantum Computers,”
    *PRX Quantum*, vol. 2, 040326, 2021.
    https://doi.org/10.1103/PRXQuantum.2.040326