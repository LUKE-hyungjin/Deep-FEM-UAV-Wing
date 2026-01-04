from __future__ import annotations

import hashlib
import json

from .types import WingParams


def _round_float(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def normalize_params(params: WingParams) -> dict[str, float]:
    return {
        "span_m": _round_float(params.span_m, 1e-3),
        "chord_m": _round_float(params.chord_m, 1e-3),
        "sweep_deg": _round_float(params.sweep_deg, 0.1),
        "thickness_ratio": _round_float(params.thickness_ratio, 1e-4),
    }


def compute_case_id(*, params: WingParams, pipeline_version: str) -> str:
    payload = {
        "params": normalize_params(params),
        "pipeline_version": pipeline_version,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


