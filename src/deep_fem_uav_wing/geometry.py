from __future__ import annotations

import shutil
import time
from pathlib import Path

from .blender_runner import run_blender_generate_stl
from .config import PIPELINE_VERSION, Paths
from .case_id import compute_case_id
from .types import ParamsFile, StepReport, WingParams


def get_geometry_case_dir(*, paths: Paths, case_id: str) -> Path:
    return paths.geometry_dir / case_id


def get_geometry_artifacts(*, case_dir: Path) -> dict[str, Path]:
    return {
        "wing_stl": case_dir / "wing.stl",
        "wing_viz_glb": case_dir / "wing_viz.glb",
        "params_json": case_dir / "params.json",
        "build_report_json": case_dir / "build_report.json",
    }


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tail_text(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def generate_sample_wing_stl(*, out_path: Path, params: WingParams) -> None:
    """
    Blender가 없는 환경에서도 데모가 돌아가도록, 단순 '날개 형태'를 트리메시로 생성해 STL로 저장한다.
    - 좌표계: +Y span, +X chord, +Z thickness (spec 준수)
    - Root는 y=0, Tip은 y=span
    - Sweep은 Tip에서 x 방향으로 이동
    """
    span = float(params.span_m)
    chord = float(params.chord_m)
    sweep = float(params.sweep_deg)
    t_ratio = float(params.thickness_ratio)

    thickness = max(1e-4, chord * t_ratio)
    dx_tip = (span * __import__("math").tan(__import__("math").radians(sweep))) if sweep else 0.0

    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "trimesh is required for the fallback STL generator. "
            "Install deps: `pip install -r requirements.txt`, or set BLENDER_BIN to use Blender generation."
        ) from e

    # 8 corners of a swept rectangular "wing" prism (very simple baseline)
    x0, x1 = 0.0, chord
    z0, z1 = -thickness / 2.0, thickness / 2.0

    # root (y=0)
    r0 = [x0, 0.0, z0]
    r1 = [x1, 0.0, z0]
    r2 = [x1, 0.0, z1]
    r3 = [x0, 0.0, z1]

    # tip (y=span) with sweep
    t0 = [x0 + dx_tip, span, z0]
    t1 = [x1 + dx_tip, span, z0]
    t2 = [x1 + dx_tip, span, z1]
    t3 = [x0 + dx_tip, span, z1]

    vertices = [r0, r1, r2, r3, t0, t1, t2, t3]
    faces = [
        # bottom
        [0, 1, 5],
        [0, 5, 4],
        # top
        [3, 7, 6],
        [3, 6, 2],
        # root
        [0, 3, 2],
        [0, 2, 1],
        # tip
        [4, 5, 6],
        [4, 6, 7],
        # leading edge
        [0, 4, 7],
        [0, 7, 3],
        # trailing edge
        [1, 2, 6],
        [1, 6, 5],
    ]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if mesh.is_empty:
        raise ValueError("sample mesh is empty")
    mesh.export(out_path)


def prepare_geometry_case(
    *,
    paths: Paths,
    params: WingParams,
    input_stl_path: Path | None,
    force_rebuild: bool = False,
) -> tuple[str, dict[str, Path], StepReport, str]:
    start = time.perf_counter()
    case_id = compute_case_id(params=params, pipeline_version=PIPELINE_VERSION)

    case_dir = get_geometry_case_dir(paths=paths, case_id=case_id)
    _ensure_dir(case_dir)

    artifacts = get_geometry_artifacts(case_dir=case_dir)
    wing_stl = artifacts["wing_stl"]
    params_json = artifacts["params_json"]
    report_json = artifacts["build_report_json"]

    logs: list[str] = []

    if force_rebuild:
        for p in artifacts.values():
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    if (not force_rebuild) and wing_stl.exists() and params_json.exists():
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logs.append("cache hit: wing.stl + params.json")
        if input_stl_path is not None:
            logs.append("note: input STL ignored because cache exists for this case_id.")

        report = StepReport(
            status="success",
            failure_reason=None,
            elapsed_ms=elapsed_ms,
            stdout_tail=_tail_text("\n".join(logs)),
            stderr_tail=None,
            artifacts=[str(p) for p in artifacts.values() if p.exists()],
        )
        # Do not overwrite an existing report on cache hit; preserve provenance.
        return case_id, artifacts, report, "\n".join(logs)

    if input_stl_path is not None:
        if not input_stl_path.exists():
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            report = StepReport(
                status="failed",
                failure_reason=f"input STL not found: {input_stl_path}",
                elapsed_ms=elapsed_ms,
                stdout_tail=None,
                stderr_tail=None,
                artifacts=[],
            )
            _write_json(report_json, report.model_dump())
            return case_id, artifacts, report, "\n".join(logs + [report.failure_reason or "failed"])

        shutil.copyfile(input_stl_path, wing_stl)
        logs.append(f"STL copied: {input_stl_path.name} → wing.stl")
    else:
        project_root = paths.project_root
        blender_result = run_blender_generate_stl(
            project_root=project_root,
            out_stl=wing_stl,
            span_m=float(params.span_m),
            chord_m=float(params.chord_m),
            sweep_deg=float(params.sweep_deg),
            thickness_ratio=float(params.thickness_ratio),
        )

        if blender_result.ok:
            logs.append("GEOMETRY_SOURCE=blender")
            logs.append(blender_result.message)
            if blender_result.stdout_tail:
                logs.append(f"[blender stdout]\n{blender_result.stdout_tail}")
            if blender_result.stderr_tail:
                logs.append(f"[blender stderr]\n{blender_result.stderr_tail}")
        else:
            logs.append("GEOMETRY_SOURCE=fallback")
            logs.append(f"Blender unavailable/failed → fallback. ({blender_result.message})")
            if blender_result.stdout_tail:
                logs.append(f"[blender stdout]\n{blender_result.stdout_tail}")
            if blender_result.stderr_tail:
                logs.append(f"[blender stderr]\n{blender_result.stderr_tail}")
            generate_sample_wing_stl(out_path=wing_stl, params=params)
            logs.append("Generated sample wing.stl (fallback).")

    params_file = ParamsFile(
        case_id=case_id,
        span_m=float(params.span_m),
        chord_m=float(params.chord_m),
        sweep_deg=float(params.sweep_deg),
        thickness_ratio=float(params.thickness_ratio),
        created_at=ParamsFile.now_iso(),
        pipeline_version=PIPELINE_VERSION,
    )
    _write_json(params_json, params_file.model_dump())

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    report = StepReport(
        status="success",
        failure_reason=None,
        elapsed_ms=elapsed_ms,
        stdout_tail=_tail_text("\n".join(logs)),
        stderr_tail=None,
        artifacts=[
            str(wing_stl),
            str(params_json),
            str(report_json),
        ],
    )
    _write_json(report_json, report.model_dump())
    return case_id, artifacts, report, "\n".join(logs)


