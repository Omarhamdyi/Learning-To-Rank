from __future__ import annotations

from collections import deque
from pathlib import Path


def safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    w = pos - lo
    return data[lo] * (1.0 - w) + data[hi] * w


def tail_log(path: Path, max_lines: int = 250) -> str:
    if not path.exists():
        return f"[log file missing] {path}"
    lines = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
    except Exception as exc:
        return f"[failed to read log file] {exc}"
    return "\n".join(lines) if lines else "[log file is empty]"


def process_memory_mb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss / 1024.0
    except Exception:
        return 0.0

