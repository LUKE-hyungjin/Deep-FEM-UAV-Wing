from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_fem_uav_wing.config import PIPELINE_VERSION, get_paths  # noqa: E402
from deep_fem_uav_wing.blender_runner import find_blender_bin  # noqa: E402
from deep_fem_uav_wing.pipeline_geometry import get_or_build_geometry  # noqa: E402
from deep_fem_uav_wing.types import WingParams  # noqa: E402


@dataclass(frozen=True)
class Ranges:
    span_m: tuple[float, float] = (1.0, 2.0)
    chord_m: tuple[float, float] = (0.2, 0.5)
    sweep_deg: tuple[float, float] = (0.0, 30.0)
    thickness_ratio: tuple[float, float] = (0.05, 0.15)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_params(rng: random.Random, ranges: Ranges) -> WingParams:
    return WingParams(
        span_m=rng.uniform(*ranges.span_m),
        chord_m=rng.uniform(*ranges.chord_m),
        sweep_deg=rng.uniform(*ranges.sweep_deg),
        thickness_ratio=rng.uniform(*ranges.thickness_ratio),
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "created_at": now_iso(),
            "pipeline_version": PIPELINE_VERSION,
            "tools": {
                "blender_bin": os.environ.get("BLENDER_BIN"),
            },
            "geometry_index": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_params_csv(
    *,
    csv_path: Path,
    row: dict[str, str],
) -> None:
    ensure_parent(csv_path)
    file_exists = csv_path.exists()
    fieldnames = list(row.keys())

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate wing geometry cases (STL + GLB) with caching.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_blender", action="store_true", help="Fail if Blender is not available.")
    args = parser.parse_args()

    paths = get_paths(PROJECT_ROOT)
    paths.geometry_dir.mkdir(parents=True, exist_ok=True)

    if args.require_blender:
        blender_bin = find_blender_bin()
        if blender_bin is None:
            print("ERROR: require_blender=true but Blender not found. Set BLENDER_BIN.", file=sys.stderr)
            return 2

    params_csv = paths.geometry_dir / "params.csv"
    manifest_json = paths.raw_dir / "manifest.json"
    manifest = load_manifest(manifest_json)
    if manifest.get("pipeline_version") != PIPELINE_VERSION:
        manifest["pipeline_version"] = PIPELINE_VERSION

    rng = random.Random(args.seed)
    ranges = Ranges()

    started_at = time.perf_counter()
    ok_count = 0
    fail_count = 0

    for i in range(args.count):
        params = sample_params(rng, ranges)

        # Force case_id to be sequential (001..200) for better readability
        forced_id = f"{i + 1:03d}"

        case_id, artifacts, report, logs = get_or_build_geometry(
            project_root=PROJECT_ROOT,
            params=params,
            force_rebuild=args.require_blender,
            forced_case_id=forced_id,
        )
        status = report.status
        failure_reason = report.failure_reason or ""

        if args.require_blender:
            # Strong signal: geometry source marker from the generation step
            if "GEOMETRY_SOURCE=blender" not in logs:
                status = "failed"
                failure_reason = "require_blender=true but Blender unavailable or generation failed"

        if status == "success":
            ok_count += 1
        else:
            fail_count += 1

        append_params_csv(
            csv_path=params_csv,
            row={
                "case_id": case_id,
                "seed": str(args.seed),
                "span_m": str(params.span_m),
                "chord_m": str(params.chord_m),
                "sweep_deg": str(params.sweep_deg),
                "thickness_ratio": str(params.thickness_ratio),
                "status": status,
                "failure_reason": failure_reason,
            },
        )

        manifest["geometry_index"].append(
            {
                "case_id": case_id,
                "created_at": now_iso(),
                "params": params.model_dump(),
                "status": status,
                "failure_reason": failure_reason,
                "artifacts": {k: str(v) for k, v in artifacts.items()},
            }
        )

        short = f"[{i+1}/{args.count}] {case_id} {status}"
        if status != "success":
            short += f" ({failure_reason})"
        print(short)

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    manifest["last_run"] = {
        "created_at": now_iso(),
        "seed": args.seed,
        "count": args.count,
        "ok": ok_count,
        "failed": fail_count,
        "elapsed_ms": elapsed_ms,
    }
    write_manifest(manifest_json, manifest)

    print(f"done: ok={ok_count} failed={fail_count} elapsed_ms={elapsed_ms}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


