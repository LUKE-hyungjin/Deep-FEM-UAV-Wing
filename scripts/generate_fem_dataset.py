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
from deep_fem_uav_wing.fem import find_ccx_bin, run_fem_case  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_mesh_success_case_ids(mesh_csv: Path) -> list[str]:
    if not mesh_csv.exists():
        return []
    with mesh_csv.open("r", encoding="utf-8", newline="") as f:
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
    parser = argparse.ArgumentParser(description="Batch-generate FEM artifacts (ccx + surface_results + wing_result.glb).")
    parser.add_argument("--limit", type=int, default=0, help="0 means all cases")
    parser.add_argument("--E", type=float, default=69e9, help="Young's modulus (Pa)")
    parser.add_argument("--nu", type=float, default=0.33, help="Poisson ratio")
    parser.add_argument("--pressure", type=float, default=5e3, help="Uniform pressure (Pa)")
    parser.add_argument("--timeout_s", type=int, default=900)
    args = parser.parse_args()

    paths = get_paths(PROJECT_ROOT)

    ccx_bin = find_ccx_bin()
    if ccx_bin is None:
        print(
            "ERROR: ccx not found. Install CalculiX or set CCX_BIN.\n"
            "- Homebrew `calculix-ccx`는 `ccx` 대신 `ccx_2.22` 같은 버전 바이너리를 설치할 수 있습니다.\n"
            "  예: export CCX_BIN=\"$(brew --prefix calculix-ccx)/bin/ccx_2.22\"\n"
            "- verify: `ccx -v` (or `which ccx`)\n",
            file=sys.stderr,
        )
        return 2

    # Postprocess uses ccx2paraview if available; otherwise falls back to python(meshio) FRD reader.

    mesh_dir = paths.raw_dir / "mesh"
    fem_dir = paths.raw_dir / "fem"
    fem_dir.mkdir(parents=True, exist_ok=True)

    mesh_csv = mesh_dir / "mesh.csv"
    case_ids = read_mesh_success_case_ids(mesh_csv)
    if args.limit and args.limit > 0:
        case_ids = case_ids[: args.limit]

    fem_csv = fem_dir / "fem.csv"
    file_exists = fem_csv.exists()

    with fem_csv.open("a", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "created_at",
            "status",
            "failure_reason",
            "elapsed_ms",
            "surface_nodes",
            "min_stress_vm",
            "max_stress_vm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for idx, case_id in enumerate(case_ids, start=1):
            try:
                ok, report, _ = run_fem_case(
                    case_id=case_id,
                    geometry_dir=paths.geometry_dir,
                    mesh_dir=mesh_dir,
                    fem_dir=fem_dir,
                    young_modulus_pa=float(args.E),
                    poisson_ratio=float(args.nu),
                    pressure_pa=float(args.pressure),
                    timeout_s=int(args.timeout_s),
                )
            except Exception as e:  # noqa: BLE001
                ok = False
                report = {"status": "failed", "failure_reason": str(e), "elapsed_ms": 0}

            status = report.get("status") or ("success" if ok else "failed")
            failure_reason = report.get("failure_reason") or ""
            stats = report.get("stats") or {}

            writer.writerow(
                {
                    "case_id": case_id,
                    "created_at": now_iso(),
                    "status": status,
                    "failure_reason": failure_reason,
                    "elapsed_ms": str(report.get("elapsed_ms") or 0),
                    "surface_nodes": str(stats.get("surface_nodes", "")),
                    "min_stress_vm": str(stats.get("min_stress_vm", "")),
                    "max_stress_vm": str(stats.get("max_stress_vm", "")),
                }
            )

            short = f"[{idx}/{len(case_ids)}] {case_id} {status}"
            if status != "success":
                short += f" ({failure_reason})"
            print(short)

    # append to manifest.json
    manifest_path = paths.raw_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fem_index = manifest.get("fem_index") or []
            fem_index.append({"created_at": now_iso(), "count": len(case_ids)})
            manifest["fem_index"] = fem_index
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


