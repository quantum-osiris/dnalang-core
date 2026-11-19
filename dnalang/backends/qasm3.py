"""OpenQASM 3 emission (text)."""
from __future__ import annotations

from typing import Optional

from ..qc_ir import Circuit


def emit(
    circ: Circuit,
    dt: Optional[float] = None,
    *,
    timing_policy: str = "exact",
    require_dt: bool = False,
) -> str:
    return circ.to_qasm3(dt, timing_policy=timing_policy, require_dt=require_dt)
