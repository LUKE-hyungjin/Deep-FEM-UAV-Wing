from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_fem_uav_wing.viz import stl_to_glb  # noqa: E402


def is_binary_glb(path: Path) -> bool:
    if not path.exists():
        return False
    return path.read_bytes()[:4] == b"glTF"


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair/normalize wing_viz.glb into real binary GLB files.")
    parser.add_argument("--geometry_dir", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "geometry"))
    args = parser.parse_args()

    geometry_dir = Path(args.geometry_dir)
    if not geometry_dir.exists():
        print(f"geometry_dir not found: {geometry_dir}", file=sys.stderr)
        return 2

    fixed = 0
    skipped = 0
    failed = 0

    for case_dir in sorted([p for p in geometry_dir.iterdir() if p.is_dir()]):
        stl_path = case_dir / "wing.stl"
        glb_path = case_dir / "wing_viz.glb"
        if not stl_path.exists() or not glb_path.exists():
            skipped += 1
            continue

        if is_binary_glb(glb_path):
            skipped += 1
            continue

        ok, message, _ = stl_to_glb(stl_path=stl_path, glb_path=glb_path)
        if ok and is_binary_glb(glb_path):
            fixed += 1
            print(f"fixed {case_dir.name}: {message}")
        else:
            failed += 1
            print(f"FAILED {case_dir.name}: {message}", file=sys.stderr)

    print(f"done: fixed={fixed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


