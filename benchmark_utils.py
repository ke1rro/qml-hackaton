import os
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

T = TypeVar("T")
_MIB = 1024 * 1024


def _rss_bytes(process: Any) -> int:
    """Return RSS for this process and all live child processes."""
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
    measure_memory: bool = True,
    **kwargs: Any,
) -> tuple[T, dict[str, Any]]:
    """Execute a local callable and measure wall time, CPU time, and RSS."""
    if not measure_memory:
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            result = function(*args, **kwargs)
        finally:
            cpu_seconds = time.process_time() - cpu_start
            wall_seconds = time.perf_counter() - wall_start
        return result, {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
        }

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

    return result, {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "baseline_rss_mib": baseline_rss / _MIB,
        "peak_rss_mib": peak_rss / _MIB,
        "peak_rss_delta_mib": (peak_rss - baseline_rss) / _MIB,
    }
