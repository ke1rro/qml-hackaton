"""Haiqu readout-error mitigation with lightweight local profiling.

The public function in this module mirrors the output shape used by
``sparse_rem.mitigate_readout_errors`` closely enough for the existing demo
helpers (``report`` and ``plot_comparison``).

Important: Haiqu jobs, including ``aer_simulator`` jobs, run in the Haiqu
cloud. The CPU and RSS measurements collected here describe the local Python
client. Server-side execution metrics remain available on the Haiqu dashboard.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

T = TypeVar("T")
_MIB = 1024 * 1024


def _rss_bytes(process: Any) -> int:
    """Return RSS for the current process and its live child processes."""
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def measure_call(
    function: Callable[..., T],
    *args: Any,
    sample_interval_s: float = 0.01,
    **kwargs: Any,
) -> tuple[T, dict[str, float]]:
    """Execute a callable and measure local wall time, CPU time, and peak RSS.

    ``wall_seconds`` includes waiting, so it is the main end-to-end metric for
    a cloud call. ``cpu_seconds`` and memory values describe the local Python
    process, not Haiqu's cloud workers.
    """
    if psutil is None:
        raise ModuleNotFoundError(
            "Memory profiling requires psutil. Install it with: "
            "python -m pip install psutil"
        )
    if sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be greater than zero")

    process = psutil.Process(os.getpid())
    baseline_rss = _rss_bytes(process)
    peak_rss = baseline_rss
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        nonlocal peak_rss
        while not stop_sampling.wait(sample_interval_s):
            peak_rss = max(peak_rss, _rss_bytes(process))

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        result = function(*args, **kwargs)
    finally:
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        stop_sampling.set()
        sampler.join()
        peak_rss = max(peak_rss, _rss_bytes(process))

    metrics = {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "baseline_rss_mib": baseline_rss / _MIB,
        "peak_rss_mib": peak_rss / _MIB,
        "peak_rss_delta_mib": (peak_rss - baseline_rss) / _MIB,
    }
    return result, metrics


def _single_distribution(result: Any) -> dict[str, float]:
    """Extract and normalize one distribution returned by ``haiqu.run``."""
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError(
            "Expected Haiqu to return one distribution for one circuit; "
            f"received {type(result).__name__}: {result!r}"
        )

    distribution = result[0]
    if not isinstance(distribution, Mapping):
        raise TypeError(
            "Expected the first Haiqu result to be a probability mapping; "
            f"received {type(distribution).__name__}"
        )

    cleaned = {
        str(bitstring).replace(" ", ""): float(probability)
        for bitstring, probability in distribution.items()
    }
    total = sum(cleaned.values())
    if total == 0:
        raise ValueError("Haiqu returned a distribution with zero total weight")
    return {bitstring: probability / total for bitstring, probability in cleaned.items()}


def _measured_input_qubits(circuit: Any) -> list[int]:
    """Return measured input-qubit indices ordered by destination classical bit."""
    measured: list[tuple[int, int]] = []
    for instruction in circuit.data:
        if instruction.operation.name != "measure":
            continue
        for qubit, clbit in zip(instruction.qubits, instruction.clbits):
            measured.append(
                (circuit.find_bit(clbit).index, circuit.find_bit(qubit).index)
            )
    return [qubit for _, qubit in sorted(measured)]


def _job_metadata(job: Any) -> dict[str, Any]:
    """Collect documented/useful job fields when exposed by the SDK version."""
    metadata: dict[str, Any] = {}
    for name in (
        "id",
        "job_id",
        "info",
        "time",
        "pre_device_pipeline_time",
        "estimated_qpu_cost",
    ):
        value = getattr(job, name, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value is not None:
            metadata[name] = value
    return metadata


def mitigate_readout_errors_haiqu(
    circuit: Any,
    shots: int = 10_000,
    device_id: str = "aer_simulator",
    noise_model: Any | None = None,
    device_options: Mapping[str, Any] | None = None,
    include_raw: bool = True,
    job_name: str = "readout-mitigation",
    sample_interval_s: float = 0.01,
) -> dict[str, Any]:
    """Run a Qiskit circuit through Haiqu readout-only error mitigation.

    Authentication and experiment initialization must be performed once before
    calling this function:

        from haiqu.sdk import haiqu
        haiqu.login()
        haiqu.init("Readout mitigation benchmark")

    Two independent jobs are submitted when ``include_raw=True``:
    one without mitigation and one with readout mitigation only. Keeping their
    timings separate avoids charging the raw comparison run to mitigation time.

    Args:
        circuit: Measured Qiskit ``QuantumCircuit``.
        shots: Number of circuit shots per Haiqu job.
        device_id: Haiqu device identifier.
        noise_model: Optional Qiskit Aer ``NoiseModel``. Supported with
            ``device_id="aer_simulator"``.
        device_options: Additional Haiqu device options. For example,
            ``{"method": "matrix_product_state"}`` enables MPS simulation.
        include_raw: Also submit an unmitigated run for comparison/plotting.
        job_name: Prefix for job names in the Haiqu dashboard.
        sample_interval_s: Local RSS sampling interval.

    Returns:
        A dictionary containing ``raw``, ``mitigated``, measured input qubits,
        separate local metrics for both jobs, and available Haiqu job metadata.
    """
    if shots <= 0:
        raise ValueError("shots must be greater than zero")
    if noise_model is not None and device_id != "aer_simulator":
        raise ValueError(
            "noise_model is supported here only with device_id='aer_simulator'"
        )

    # Lazy imports let ``measure_call`` remain usable before Haiqu is installed.
    from haiqu.sdk import haiqu

    base_options: dict[str, Any] = dict(device_options or {})
    if noise_model is not None:
        base_options["noise_model"] = noise_model

    mitigation_options = {
        **base_options,
        "error_mitigation_options": {
            "readout_mitigation": True,
            "dynamical_decoupling": False,
            "noise_tailoring": False,
            "advanced_mitigation": False,
        },
    }

    def run_once(*, use_mitigation: bool, suffix: str) -> tuple[Any, Any]:
        job = haiqu.run(
            circuits=circuit,
            shots=shots,
            device_id=device_id,
            options=mitigation_options if use_mitigation else base_options,
            use_mitigation=use_mitigation,
            job_name=f"{job_name}-{suffix}",
        )
        return job, job.result()

    raw_distribution: dict[str, float] = {}
    raw_metrics: dict[str, float] | None = None
    raw_job_metadata: dict[str, Any] | None = None

    if include_raw:
        (raw_job, raw_result), raw_metrics = measure_call(
            run_once,
            use_mitigation=False,
            suffix="raw",
            sample_interval_s=sample_interval_s,
        )
        raw_distribution = _single_distribution(raw_result)
        raw_job_metadata = _job_metadata(raw_job)

    (mitigated_job, mitigated_result), mitigated_metrics = measure_call(
        run_once,
        use_mitigation=True,
        suffix="mitigated",
        sample_interval_s=sample_interval_s,
    )

    return {
        "raw": raw_distribution,
        "mitigated": _single_distribution(mitigated_result),
        # For the simulator these match physical indices. On real hardware,
        # Haiqu owns the final transpilation/layout, so these are input indices.
        "measured_physical_qubits": _measured_input_qubits(circuit),
        "metrics": {
            "raw": raw_metrics,
            "mitigated": mitigated_metrics,
        },
        "haiqu_jobs": {
            "raw": raw_job_metadata,
            "mitigated": _job_metadata(mitigated_job),
        },
    }