from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_fem_uav_wing.config import get_paths  # noqa: E402
from deep_fem_uav_wing.meshing import find_gmsh_bin, run_meshing_case  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_geometry_success_case_ids(params_csv: Path) -> list[str]:
    if not params_csv.exists():
        return []
    with params_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    case_ids = [(r.get("case_id") or "").strip() for r in rows if (r.get("status") or "").strip() == "success"]
    # unique preserve order
    seen: set[str] = set()
    out: list[str] = []
    for cid in case_ids:
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate meshing artifacts (wing.msh + boundary sets).")
    parser.add_argument("--limit", type=int, default=0, help="0 means all cases")
    parser.add_argument("--y_tol", type=float, default=1e-4)
    parser.add_argument("--nz_min", type=float, default=0.2)
    args = parser.parse_args()

    paths = get_paths(PROJECT_ROOT)
    gmsh_bin = find_gmsh_bin()
    if gmsh_bin is None:
        print(
            "ERROR: gmsh not found. Install gmsh or set GMSH_BIN.\n"
            "- macOS(Homebrew): `brew install gmsh`\n"
            "- verify: `gmsh -version`",
            file=sys.stderr,
        )
        return 2

    geometry_params_csv = paths.geometry_dir / "params.csv"
    case_ids = read_geometry_success_case_ids(geometry_params_csv)
    # Filter out cases missing wing.stl (avoid noisy failures)
    case_ids = [cid for cid in case_ids if (paths.geometry_dir / cid / "wing.stl").exists()]
    if args.limit and args.limit > 0:
        case_ids = case_ids[: args.limit]

    mesh_dir = paths.raw_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_csv = mesh_dir / "mesh.csv"
    file_exists = mesh_csv.exists()

    with mesh_csv.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "created_at",
            "status",
            "failure_reason",
            "nodes",
            "tets",
            "tris",
            "nroot_count",
            "surf_upper_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        ok_count = 0
        fail_count = 0

        for idx, case_id in enumerate(case_ids, start=1):
            try:
                ok, report, _ = run_meshing_case(
                    case_id=case_id,
                    geometry_dir=paths.geometry_dir,
                    mesh_dir=mesh_dir,
                    y_tol=args.y_tol,
                    nz_min=args.nz_min,
                )
            except Exception as e:  # noqa: BLE001
                ok = False
                report = {"status": "failed", "failure_reason": str(e)}

            status = report.get("status") or "failed"
            failure_reason = report.get("failure_reason") or ""
            stats = report.get("stats") or {}

            if status == "success":
                ok_count += 1
            else:
                fail_count += 1

            writer.writerow(
                {
                    "case_id": case_id,
                    "created_at": now_iso(),
                    "status": status,
                    "failure_reason": failure_reason,
                    "nodes": str(stats.get("nodes", "")),
                    "tets": str(stats.get("tets", "")),
                    "tris": str(stats.get("tris", "")),
                    "nroot_count": str(stats.get("nroot_count", "")),
                    "surf_upper_ratio": str(stats.get("surf_upper_ratio", "")),
                }
            )

            short = f"[{idx}/{len(case_ids)}] {case_id} {status}"
            if status != "success":
                short += f" ({failure_reason})"
            print(short)

    # also append to manifest.json (optional, lightweight)
    manifest_path = paths.raw_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mesh_index = manifest.get("mesh_index") or []
            mesh_index.append({"created_at": now_iso(), "count": len(case_ids)})
            manifest["mesh_index"] = mesh_index
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


