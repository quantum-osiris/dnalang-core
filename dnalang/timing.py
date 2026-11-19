"""Exact rational duration representation and explicit target-clock conversion policy.

Sprint 1 timing contract (`S1-02` / `S1-04`):
- Physical durations (`ns`, `us`, `ms`, `s`) are represented as exact `fractions.Fraction`
  values in seconds throughout parsing, constant folding, and logical `Circuit` IR.
- Equivalent physical durations (`1000ns`, `1us`, `0.001ms`, `0.000001s`) normalize to
  the exact same `Fraction` and canonical IR representation.
- Conversion to target clock ticks (`dt`) requires an explicit positive clock period and
  an explicit timing policy:
    * `"exact"` (default): fails with `TimingError` if any delay is not an exact integer
      multiple of `dt`.
    * `"quantize"`: rounds to the nearest integer tick and records `requested_duration_s`,
      `effective_duration_s`, `ticks`, and `adjustment_s` for every delay operation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union


class CompilerContractError(ValueError):
    """Base error for DNA-Lang logical-IR and timing contract violations."""


class IRValidationError(CompilerContractError):
    """Raised when logical `Circuit` IR violates structural, operand, or cbit invariants."""


class TimingError(CompilerContractError):
    """Raised when duration dimensions, bounds, clock info, or exact tick conversion fail."""


UNIT_FACTORS_S: Dict[str, Fraction] = {
    "ns": Fraction(1, 1_000_000_000),
    "us": Fraction(1, 1_000_000),
    "ms": Fraction(1, 1_000),
    "s": Fraction(1, 1),
}

# Ordered longest-suffix-first so "ns"/"us"/"ms" match before "s"
_UNIT_SUFFIXES: Tuple[str, ...] = ("ns", "us", "ms", "s")


def parse_number_token_exact(raw_text: str) -> Tuple[Fraction, str]:
    """Parse a lexer NUMBER token into `(exact_fraction_in_base_unit, dim)`.

    Returns `dim == "duration"` when a duration unit (`ns`, `us`, `ms`, `s`) is present,
    `"zero"` for a unitless zero literal (`0`, `0.0`), or `"scalar"` for any non-zero
    unitless literal.
    """
    s = raw_text.strip()
    for unit in _UNIT_SUFFIXES:
        if s.endswith(unit):
            stem = s[: -len(unit)]
            # Guard against scientific notation ending in e.g. unsupported suffix (lexer already checks)
            if stem and not stem.endswith(("e", "E", "+", "-")):
                try:
                    dec = Decimal(stem)
                except InvalidOperation as exc:
                    raise TimingError(f"Invalid numeric literal {raw_text!r}") from exc
                exact_s = Fraction(dec) * UNIT_FACTORS_S[unit]
                return exact_s, "duration"

    try:
        dec = Decimal(s)
    except InvalidOperation as exc:
        raise TimingError(f"Invalid numeric literal {raw_text!r}") from exc
    exact_val = Fraction(dec)
    dim = "zero" if exact_val == 0 else "scalar"
    return exact_val, dim


def coerce_exact_seconds(value: Union[Fraction, Decimal, int, float, str]) -> Fraction:
    """Convert a duration or clock `dt` specification into exact `Fraction` seconds."""
    if isinstance(value, bool):
        raise TimingError("Boolean is not a valid duration or clock period")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise TimingError(f"Non-finite duration/clock value {value!r}")
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TimingError(f"Non-finite duration/clock value {value!r}")
        # repr(float) gives shortest decimal string that round-trips; avoids binary mantissa noise
        return Fraction(Decimal(repr(value)))
    if isinstance(value, str):
        exact, _ = parse_number_token_exact(value)
        return exact
    raise TimingError(f"Unsupported duration/clock type {type(value).__name__}")


def canonical_duration_str(duration_s: Fraction) -> str:
    """Format an exact `Fraction` duration in seconds deterministically without loss.

    Terminating decimal rationals (`denominator == 2^a * 5^b`) format as normalized
    fixed-point decimal strings (`"0.000001"` for `1us`, `"1000ns"`, `"0.001ms"`).
    Non-terminating rationals (`1us / 3`) format as reduced `"numerator/denominator"`.
    """
    if duration_s == 0:
        return "0"
    sign = "-" if duration_s < 0 else ""
    num = abs(duration_s.numerator)
    den = duration_s.denominator

    rem = den
    twos = 0
    while rem % 2 == 0:
        rem //= 2
        twos += 1
    fives = 0
    while rem % 5 == 0:
        rem //= 5
        fives += 1

    if rem == 1:
        scale = max(twos, fives)
        scaled_int = num * (2 ** (scale - twos)) * (5 ** (scale - fives))
        if scale == 0:
            return f"{sign}{scaled_int}"
        digits = str(scaled_int).rjust(scale + 1, "0")
        whole = digits[:-scale]
        frac_part = digits[-scale:].rstrip("0")
        return f"{sign}{whole}.{frac_part}" if frac_part else f"{sign}{whole}"

    return f"{sign}{num}/{den}"


def format_exact_ns_qasm(duration_s: Fraction) -> str:
    """Format an exact duration in `ns` for physical-unit OpenQASM 3 emission."""
    ns_frac = duration_s * 1_000_000_000
    text = canonical_duration_str(ns_frac)
    if "/" in text:
        # OpenQASM 3 duration literal requires decimal float/int with unit
        return f"{float(ns_frac)!r}ns"
    return f"{text}ns"


@dataclass(frozen=True)
class QuantizedDelayRecord:
    """Audit record for converting one logical delay into target clock ticks (`dt`)."""

    op_index: int
    qubits: Tuple[int, ...]
    requested_duration_s: Fraction
    effective_duration_s: Fraction
    ticks: int
    adjustment_s: Fraction

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_index": self.op_index,
            "qubits": list(self.qubits),
            "requested_duration_s": canonical_duration_str(self.requested_duration_s),
            "effective_duration_s": canonical_duration_str(self.effective_duration_s),
            "ticks": self.ticks,
            "adjustment_s": canonical_duration_str(self.adjustment_s),
        }


@dataclass(frozen=True)
class TimingResolutionReport:
    """Summary of target-tick timing conversion across an entire `Circuit`."""

    dt_s: Fraction
    policy: str
    records: List[QuantizedDelayRecord] = field(default_factory=list)

    @property
    def adjusted_count(self) -> int:
        return sum(1 for r in self.records if r.adjustment_s != 0)

    @property
    def max_abs_adjustment_s(self) -> Fraction:
        return max((abs(r.adjustment_s) for r in self.records), default=Fraction(0, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dt_s": canonical_duration_str(self.dt_s),
            "policy": self.policy,
            "delay_count": len(self.records),
            "adjusted_count": self.adjusted_count,
            "max_abs_adjustment_s": canonical_duration_str(self.max_abs_adjustment_s),
            "records": [r.to_dict() for r in self.records],
        }


def resolve_delay_ticks(
    duration_exact_s: Fraction,
    *,
    dt: Optional[Union[Fraction, Decimal, int, float, str]],
    policy: str = "exact",
    op_index: int = 0,
    qubits: Tuple[int, ...] = (0,),
) -> QuantizedDelayRecord:
    """Convert an exact delay duration into integer clock `dt` ticks under `policy`."""
    if policy not in ("exact", "quantize"):
        raise TimingError(f"Unsupported timing_policy {policy!r}; expected 'exact' or 'quantize'")
    if dt is None:
        raise TimingError("Conversion to target ticks requires a non-null clock period 'dt'")
    dt_exact = coerce_exact_seconds(dt)
    if dt_exact <= 0:
        raise TimingError(f"Target clock period 'dt' must be > 0 (got {canonical_duration_str(dt_exact)} s)")
    if duration_exact_s < 0:
        raise TimingError(
            f"Delay duration at op #{op_index} must be non-negative "
            f"(got {canonical_duration_str(duration_exact_s)} s)"
        )

    ratio = duration_exact_s / dt_exact
    if policy == "exact":
        if ratio.denominator != 1:
            floor_t = ratio.numerator // ratio.denominator
            ceil_t = floor_t + 1
            raise TimingError(
                f"Delay duration {canonical_duration_str(duration_exact_s)} s at op #{op_index} "
                f"is not an exact integer multiple of dt={canonical_duration_str(dt_exact)} s "
                f"(ratio={ratio.numerator}/{ratio.denominator} ticks, between {floor_t}dt and {ceil_t}dt); "
                f"pass timing_policy='quantize' to allow explicit tick quantization"
            )
        ticks = int(ratio.numerator)
    else:
        # Deterministic nearest-tick rounding (half-up for non-negative ratios)
        ticks = int((2 * ratio.numerator + ratio.denominator) // (2 * ratio.denominator))

    effective_s = Fraction(ticks, 1) * dt_exact
    adjustment_s = effective_s - duration_exact_s
    return QuantizedDelayRecord(
        op_index=op_index,
        qubits=qubits,
        requested_duration_s=duration_exact_s,
        effective_duration_s=effective_s,
        ticks=ticks,
        adjustment_s=adjustment_s,
    )
