from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlenderRunResult:
    ok: bool
    message: str
    stdout_tail: str | None
    stderr_tail: str | None


def _tail(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def find_blender_bin() -> Path | None:
    env = os.environ.get("BLENDER_BIN")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None

    candidates = [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("/usr/bin/blender"),
        Path("/usr/local/bin/blender"),
        Path("/opt/homebrew/bin/blender"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_blender_generate_stl(
    *,
    project_root: Path,
    out_stl: Path,
    span_m: float,
    chord_m: float,
    sweep_deg: float,
    thickness_ratio: float,
    timeout_s: int = 120,
) -> BlenderRunResult:
    blender_bin = find_blender_bin()
    if blender_bin is None:
        return BlenderRunResult(
            ok=False,
            message="Blender not found (set BLENDER_BIN to enable bpy generation).",
            stdout_tail=None,
            stderr_tail=None,
        )

    script_path = (project_root / "blender" / "generate_wing.py").resolve()
    if not script_path.exists():
        return BlenderRunResult(
            ok=False,
            message=f"Blender script not found: {script_path}",
            stdout_tail=None,
            stderr_tail=None,
        )

    out_stl.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(blender_bin),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(script_path),
        "--",
        "--span_m",
        str(span_m),
        "--chord_m",
        str(chord_m),
        "--sweep_deg",
        str(sweep_deg),
        "--thickness_ratio",
        str(thickness_ratio),
        "--out_stl",
        str(out_stl.resolve()),
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as e:  # noqa: BLE001
        return BlenderRunResult(
            ok=False,
            message=f"Blender execution failed: {e}",
            stdout_tail=None,
            stderr_tail=None,
        )

    stdout_tail = _tail(completed.stdout or "")
    stderr_tail = _tail(completed.stderr or "")

    if completed.returncode != 0:
        return BlenderRunResult(
            ok=False,
            message=f"Blender exited with code {completed.returncode}.",
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    if not out_stl.exists():
        return BlenderRunResult(
            ok=False,
            message="Blender reported success but wing.stl not found.",
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    return BlenderRunResult(
        ok=True,
        message="wing.stl generated via Blender (bpy).",
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


