from __future__ import annotations

import time
from pathlib import Path

from .config import get_paths
from .geometry import get_geometry_artifacts, get_geometry_case_dir, prepare_geometry_case
from .types import StepReport, WingParams
from .viz import stl_to_glb


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_or_build_geometry(
    *,
    project_root: Path,
    params: WingParams,
    input_stl_path: Path | None = None,
    force_rebuild: bool = False,
) -> tuple[str, dict[str, Path], StepReport, str]:
    """
    Returns (case_id, artifacts, report, logs).
    Artifacts include: wing.stl, wing_viz.glb, params.json, build_report.json
    """
    paths = get_paths(project_root)
    start = time.perf_counter()

    # 1) Prepare case + wing.stl + params/report
    case_id, artifacts, report, logs = prepare_geometry_case(
        paths=paths, params=params, input_stl_path=input_stl_path, force_rebuild=force_rebuild
    )
    if report.status != "success":
        return case_id, artifacts, report, logs

    # 2) Cache hit: wing_viz.glb exists
    case_dir = get_geometry_case_dir(paths=paths, case_id=case_id)
    all_artifacts = get_geometry_artifacts(case_dir=case_dir)
    wing_viz_glb = all_artifacts["wing_viz_glb"]
    if (not force_rebuild) and wing_viz_glb.exists():
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        cached_report = StepReport(
            status="success",
            failure_reason=None,
            elapsed_ms=elapsed_ms,
            stdout_tail=(logs + "\ncache hit: wing_viz.glb").strip(),
            stderr_tail=None,
            artifacts=[str(p) for p in all_artifacts.values() if p.exists()],
        )
        return case_id, all_artifacts, cached_report, cached_report.stdout_tail or ""

    # 3) STL -> GLB
    ok, message, viz_elapsed_ms = stl_to_glb(
        stl_path=all_artifacts["wing_stl"], glb_path=wing_viz_glb
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if not ok:
        failed_report = StepReport(
            status="failed",
            failure_reason=f"viz failed: {message}",
            elapsed_ms=elapsed_ms,
            stdout_tail=(logs + "\n" + message).strip(),
            stderr_tail=None,
            artifacts=[str(p) for p in all_artifacts.values() if p.exists()],
        )
        _write_json(all_artifacts["build_report_json"], failed_report.model_dump())
        return case_id, all_artifacts, failed_report, failed_report.stdout_tail or ""

    success_report = StepReport(
        status="success",
        failure_reason=None,
        elapsed_ms=elapsed_ms,
        stdout_tail=(logs + f"\n{message} (viz_elapsed_ms={viz_elapsed_ms})").strip(),
        stderr_tail=None,
        artifacts=[str(p) for p in all_artifacts.values() if p.exists()],
    )
    _write_json(all_artifacts["build_report_json"], success_report.model_dump())
    return case_id, all_artifacts, success_report, success_report.stdout_tail or ""


