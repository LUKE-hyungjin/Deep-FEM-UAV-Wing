"""
Stage 4: Dataset Validation & Quality Assurance

Implements automated validation checklist from PRD:
- Minimum 200 solved cases (status=success)
- Root node count validation
- Upper surface area ratio validation
- nan/inf result detection
- Scale sanity checks
- Tool version tracking for reproducibility
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from .config import PIPELINE_VERSION, Paths


# ---------------------------------------------------------------------------
# Tool version detection
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], timeout: float = 10.0) -> str | None:
    """Run command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def get_gmsh_version() -> str | None:
    """Detect gmsh version."""
    # Try Python API first
    try:
        import gmsh  # type: ignore

        gmsh.initialize()
        ver = gmsh.GMSH_API_VERSION
        gmsh.finalize()
        return str(ver)
    except Exception:
        pass

    # Try CLI
    gmsh_bin = shutil.which("gmsh")
    if gmsh_bin:
        out = _run_cmd([gmsh_bin, "--version"])
        if out:
            return out.split("\n")[0].strip()
    return None


def get_calculix_version(*, paths: Paths | None = None) -> str | None:
    """Detect CalculiX (ccx) version."""
    import re

    ccx_bin = shutil.which("ccx")
    if not ccx_bin:
        # Common macOS Homebrew location
        for candidate in ["/opt/homebrew/bin/ccx", "/usr/local/bin/ccx"]:
            if Path(candidate).exists():
                ccx_bin = candidate
                break

    if ccx_bin:
        # ccx -v outputs version info to stdout
        out = _run_cmd([ccx_bin, "-v"])
        if out:
            # Parse "Version 2.22" from output
            for line in out.split("\n"):
                if "version" in line.lower():
                    return line.strip()
            return out.split("\n")[0].strip()

    # Fallback: parse from existing FEM reports
    if paths is not None:
        fem_dir = paths.fem_dir
        if fem_dir.exists():
            for case_dir in fem_dir.iterdir():
                if case_dir.is_dir():
                    report_path = case_dir / "fem_report.json"
                    if report_path.exists():
                        try:
                            data = json.loads(report_path.read_text(encoding="utf-8"))
                            stdout = data.get("stdout_tail", "")
                            # Look for "CalculiX Version X.XX"
                            match = re.search(r"CalculiX Version\s+([\d.]+)", stdout)
                            if match:
                                return f"CalculiX Version {match.group(1)}"
                        except Exception:
                            pass
                        break  # Only check first case
    return None


def get_blender_version(*, paths: Paths | None = None) -> str | None:
    """Detect Blender version."""
    import os
    import re

    blender_bin = os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if blender_bin and Path(blender_bin).exists():
        out = _run_cmd([blender_bin, "--version"])
        if out:
            # First line: "Blender X.Y.Z"
            return out.split("\n")[0].strip()

    # Fallback: parse from existing geometry reports
    if paths is not None:
        geom_dir = paths.geometry_dir
        if geom_dir.exists():
            for case_dir in geom_dir.iterdir():
                if case_dir.is_dir():
                    report_path = case_dir / "build_report.json"
                    if report_path.exists():
                        try:
                            data = json.loads(report_path.read_text(encoding="utf-8"))
                            stdout = data.get("stdout_tail", "")
                            # Look for "Blender X.Y.Z"
                            match = re.search(r"Blender\s+([\d.]+)", stdout)
                            if match:
                                return f"Blender {match.group(1)}"
                        except Exception:
                            pass
                        break  # Only check first case
    return None


def get_python_packages_versions() -> dict[str, str | None]:
    """Get versions of key Python packages."""
    packages: dict[str, str | None] = {}

    for pkg_name in ["numpy", "pydantic", "trimesh", "gmsh"]:
        try:
            mod = __import__(pkg_name)
            packages[pkg_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg_name] = None

    return packages


def collect_tool_versions(*, paths: Paths | None = None) -> dict[str, Any]:
    """Collect all tool versions for manifest."""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "gmsh": get_gmsh_version(),
        "calculix": get_calculix_version(paths=paths),
        "blender": get_blender_version(paths=paths),
        "python_packages": get_python_packages_versions(),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Validation thresholds & checks
# ---------------------------------------------------------------------------


@dataclass
class ValidationThresholds:
    """Configurable thresholds for dataset validation."""

    min_solved_cases: int = 200
    min_root_nodes: int = 10
    max_root_nodes: int = 500
    min_upper_surface_ratio: float = 0.3
    max_upper_surface_ratio: float = 0.7
    min_mesh_quality_ok_ratio: float = 0.7
    max_stress_mpa: float = 1000.0  # 1 GPa sanity limit for aluminum
    max_displacement_m: float = 1.0  # 1 meter sanity limit
    min_nodes: int = 500
    max_nodes: int = 500000
    min_tets: int = 1000
    max_tets: int = 2500000


class CaseValidationResult(BaseModel):
    """Validation result for a single case."""

    case_id: str
    status: str  # "valid", "warning", "failed", "missing"
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Extracted metrics for summary
    mesh_nodes: int | None = None
    mesh_tets: int | None = None
    mesh_quality_ok_ratio: float | None = None
    root_node_count: int | None = None
    upper_surface_ratio: float | None = None
    stress_max_mpa: float | None = None
    stress_mean_mpa: float | None = None
    disp_max_m: float | None = None
    has_nan_inf: bool | None = None


class DatasetValidationReport(BaseModel):
    """Complete dataset validation report."""

    created_at: str
    pipeline_version: str
    tool_versions: dict[str, Any]
    thresholds: dict[str, Any]

    # Summary counts
    total_cases: int
    valid_cases: int
    warning_cases: int
    failed_cases: int
    missing_cases: int

    # Checklist items (PRD requirements)
    checklist: dict[str, bool]

    # Aggregate statistics
    statistics: dict[str, Any]

    # Per-case details (optional, can be large)
    case_results: list[CaseValidationResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    """Load JSON file, return None if not found or invalid."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def validate_single_case(
    *,
    case_id: str,
    paths: Paths,
    thresholds: ValidationThresholds,
) -> CaseValidationResult:
    """Validate a single case across geometry, mesh, and FEM stages."""
    issues: list[str] = []
    warnings: list[str] = []

    # Paths
    geom_dir = paths.geometry_dir / case_id
    mesh_dir = paths.mesh_dir / case_id
    fem_dir = paths.fem_dir / case_id

    geom_report = _load_json(geom_dir / "build_report.json")
    mesh_report = _load_json(mesh_dir / "mesh_report.json")
    fem_report = _load_json(fem_dir / "fem_report.json")

    # Check if all stages exist
    if geom_report is None:
        return CaseValidationResult(case_id=case_id, status="missing", issues=["geometry report missing"])
    if mesh_report is None:
        return CaseValidationResult(case_id=case_id, status="missing", issues=["mesh report missing"])
    if fem_report is None:
        return CaseValidationResult(case_id=case_id, status="missing", issues=["fem report missing"])

    # Check stage statuses
    if geom_report.get("status") != "success":
        issues.append(f"geometry failed: {geom_report.get('failure_reason', 'unknown')}")
    if mesh_report.get("status") != "success":
        issues.append(f"mesh failed: {mesh_report.get('failure_reason', 'unknown')}")
    if fem_report.get("status") != "success":
        issues.append(f"fem failed: {fem_report.get('failure_reason', 'unknown')}")

    # Extract mesh stats
    mesh_stats = mesh_report.get("stats", {})
    mesh_quality = mesh_report.get("quality", {})
    mesh_nodes = mesh_stats.get("nodes")
    mesh_tets = mesh_stats.get("tets")
    root_node_count = mesh_stats.get("nroot_count")
    upper_surface_ratio = mesh_stats.get("surf_upper_ratio")
    mesh_quality_ok_ratio = mesh_quality.get("quality_ok_ratio")

    # Validate mesh stats
    if mesh_nodes is not None:
        if mesh_nodes < thresholds.min_nodes:
            issues.append(f"mesh nodes too few: {mesh_nodes} < {thresholds.min_nodes}")
        elif mesh_nodes > thresholds.max_nodes:
            issues.append(f"mesh nodes too many: {mesh_nodes} > {thresholds.max_nodes}")

    if mesh_tets is not None:
        if mesh_tets < thresholds.min_tets:
            issues.append(f"mesh tets too few: {mesh_tets} < {thresholds.min_tets}")
        elif mesh_tets > thresholds.max_tets:
            issues.append(f"mesh tets too many: {mesh_tets} > {thresholds.max_tets}")

    if root_node_count is not None:
        if root_node_count < thresholds.min_root_nodes:
            warnings.append(f"few root nodes: {root_node_count} < {thresholds.min_root_nodes}")
        elif root_node_count > thresholds.max_root_nodes:
            warnings.append(f"many root nodes: {root_node_count} > {thresholds.max_root_nodes}")

    if upper_surface_ratio is not None:
        if upper_surface_ratio < thresholds.min_upper_surface_ratio:
            warnings.append(f"low upper surface ratio: {upper_surface_ratio:.3f} < {thresholds.min_upper_surface_ratio}")
        elif upper_surface_ratio > thresholds.max_upper_surface_ratio:
            warnings.append(f"high upper surface ratio: {upper_surface_ratio:.3f} > {thresholds.max_upper_surface_ratio}")

    if mesh_quality_ok_ratio is not None:
        if mesh_quality_ok_ratio < thresholds.min_mesh_quality_ok_ratio:
            warnings.append(f"low mesh quality: {mesh_quality_ok_ratio:.3f} < {thresholds.min_mesh_quality_ok_ratio}")

    # Extract FEM stats
    fem_stats = fem_report.get("stats", {})
    metrics_all = fem_report.get("metrics_all_nodes", {})
    metrics_masked = fem_report.get("metrics_masked_nodes", {})

    stress_max_pa = metrics_all.get("stress_max") or fem_stats.get("max_stress_vm")
    stress_mean_pa = metrics_masked.get("stress_mean") or fem_stats.get("mean_stress_vm")
    disp_max = fem_stats.get("max_disp_mag")

    stress_max_mpa = stress_max_pa / 1e6 if stress_max_pa else None
    stress_mean_mpa = stress_mean_pa / 1e6 if stress_mean_pa else None

    # Check for nan/inf in surface results
    has_nan_inf = False
    surface_npz_path = fem_dir / "surface_results.npz"
    if surface_npz_path.exists():
        try:
            data = np.load(surface_npz_path)
            for key in ["stress_vm", "disp"]:
                if key in data:
                    arr = data[key]
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        has_nan_inf = True
                        issues.append(f"nan/inf detected in {key}")
        except Exception as e:
            warnings.append(f"could not load surface_results.npz: {e}")

    # Sanity checks
    if stress_max_mpa is not None and stress_max_mpa > thresholds.max_stress_mpa:
        warnings.append(f"very high stress: {stress_max_mpa:.1f} MPa > {thresholds.max_stress_mpa} MPa")

    if disp_max is not None and disp_max > thresholds.max_displacement_m:
        warnings.append(f"very large displacement: {disp_max:.3f} m > {thresholds.max_displacement_m} m")

    # Determine overall status
    if issues:
        status = "failed"
    elif warnings:
        status = "warning"
    else:
        status = "valid"

    return CaseValidationResult(
        case_id=case_id,
        status=status,
        issues=issues,
        warnings=warnings,
        mesh_nodes=mesh_nodes,
        mesh_tets=mesh_tets,
        mesh_quality_ok_ratio=mesh_quality_ok_ratio,
        root_node_count=root_node_count,
        upper_surface_ratio=upper_surface_ratio,
        stress_max_mpa=stress_max_mpa,
        stress_mean_mpa=stress_mean_mpa,
        disp_max_m=disp_max,
        has_nan_inf=has_nan_inf,
    )


def validate_dataset(
    *,
    paths: Paths,
    thresholds: ValidationThresholds | None = None,
    include_case_details: bool = True,
) -> DatasetValidationReport:
    """
    Validate the entire dataset against PRD quality checklist.

    Returns a comprehensive report with:
    - Tool versions for reproducibility
    - Checklist pass/fail for each PRD requirement
    - Aggregate statistics
    - Per-case validation results (optional)
    """
    if thresholds is None:
        thresholds = ValidationThresholds()

    tool_versions = collect_tool_versions(paths=paths)

    # Discover all case IDs from FEM directory (the final stage)
    fem_dir = paths.fem_dir
    case_ids: list[str] = []
    if fem_dir.exists():
        case_ids = sorted([d.name for d in fem_dir.iterdir() if d.is_dir()])

    # Validate each case
    case_results: list[CaseValidationResult] = []
    for case_id in case_ids:
        result = validate_single_case(case_id=case_id, paths=paths, thresholds=thresholds)
        case_results.append(result)

    # Count statuses
    valid_cases = sum(1 for r in case_results if r.status == "valid")
    warning_cases = sum(1 for r in case_results if r.status == "warning")
    failed_cases = sum(1 for r in case_results if r.status == "failed")
    missing_cases = sum(1 for r in case_results if r.status == "missing")
    total_cases = len(case_results)

    # Compute aggregate statistics
    stats: dict[str, Any] = {}

    # Mesh stats
    mesh_nodes_list = [r.mesh_nodes for r in case_results if r.mesh_nodes is not None]
    mesh_tets_list = [r.mesh_tets for r in case_results if r.mesh_tets is not None]
    quality_list = [r.mesh_quality_ok_ratio for r in case_results if r.mesh_quality_ok_ratio is not None]
    root_counts = [r.root_node_count for r in case_results if r.root_node_count is not None]
    upper_ratios = [r.upper_surface_ratio for r in case_results if r.upper_surface_ratio is not None]

    if mesh_nodes_list:
        stats["mesh_nodes"] = {
            "min": int(min(mesh_nodes_list)),
            "max": int(max(mesh_nodes_list)),
            "mean": float(np.mean(mesh_nodes_list)),
        }
    if mesh_tets_list:
        stats["mesh_tets"] = {
            "min": int(min(mesh_tets_list)),
            "max": int(max(mesh_tets_list)),
            "mean": float(np.mean(mesh_tets_list)),
        }
    if quality_list:
        stats["mesh_quality_ok_ratio"] = {
            "min": float(min(quality_list)),
            "max": float(max(quality_list)),
            "mean": float(np.mean(quality_list)),
        }
    if root_counts:
        stats["root_node_count"] = {
            "min": int(min(root_counts)),
            "max": int(max(root_counts)),
            "mean": float(np.mean(root_counts)),
        }
    if upper_ratios:
        stats["upper_surface_ratio"] = {
            "min": float(min(upper_ratios)),
            "max": float(max(upper_ratios)),
            "mean": float(np.mean(upper_ratios)),
        }

    # FEM stats
    stress_max_list = [r.stress_max_mpa for r in case_results if r.stress_max_mpa is not None]
    stress_mean_list = [r.stress_mean_mpa for r in case_results if r.stress_mean_mpa is not None]
    disp_list = [r.disp_max_m for r in case_results if r.disp_max_m is not None]

    if stress_max_list:
        stats["stress_max_mpa"] = {
            "min": float(min(stress_max_list)),
            "max": float(max(stress_max_list)),
            "mean": float(np.mean(stress_max_list)),
            "p95": float(np.percentile(stress_max_list, 95)),
        }
    if stress_mean_list:
        stats["stress_mean_mpa"] = {
            "min": float(min(stress_mean_list)),
            "max": float(max(stress_mean_list)),
            "mean": float(np.mean(stress_mean_list)),
        }
    if disp_list:
        stats["disp_max_m"] = {
            "min": float(min(disp_list)),
            "max": float(max(disp_list)),
            "mean": float(np.mean(disp_list)),
        }

    # nan/inf count
    nan_inf_count = sum(1 for r in case_results if r.has_nan_inf)
    stats["nan_inf_cases"] = nan_inf_count

    # Build PRD checklist
    solved_count = valid_cases + warning_cases  # "solved" = not failed/missing
    checklist = {
        "min_200_solved_cases": solved_count >= thresholds.min_solved_cases,
        "no_nan_inf_results": nan_inf_count == 0,
        "root_nodes_in_range": all(
            thresholds.min_root_nodes <= (r.root_node_count or 0) <= thresholds.max_root_nodes
            for r in case_results
            if r.root_node_count is not None
        ),
        "upper_surface_ratio_valid": all(
            thresholds.min_upper_surface_ratio <= (r.upper_surface_ratio or 0) <= thresholds.max_upper_surface_ratio
            for r in case_results
            if r.upper_surface_ratio is not None
        ),
        "stress_sanity_check": all(
            (r.stress_max_mpa or 0) <= thresholds.max_stress_mpa for r in case_results if r.stress_max_mpa is not None
        ),
        "displacement_sanity_check": all(
            (r.disp_max_m or 0) <= thresholds.max_displacement_m for r in case_results if r.disp_max_m is not None
        ),
        "tool_versions_recorded": all(
            v is not None for k, v in tool_versions.items() if k not in ("python_packages", "collected_at")
        ),
    }

    return DatasetValidationReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        pipeline_version=PIPELINE_VERSION,
        tool_versions=tool_versions,
        thresholds={
            "min_solved_cases": thresholds.min_solved_cases,
            "min_root_nodes": thresholds.min_root_nodes,
            "max_root_nodes": thresholds.max_root_nodes,
            "min_upper_surface_ratio": thresholds.min_upper_surface_ratio,
            "max_upper_surface_ratio": thresholds.max_upper_surface_ratio,
            "min_mesh_quality_ok_ratio": thresholds.min_mesh_quality_ok_ratio,
            "max_stress_mpa": thresholds.max_stress_mpa,
            "max_displacement_m": thresholds.max_displacement_m,
        },
        total_cases=total_cases,
        valid_cases=valid_cases,
        warning_cases=warning_cases,
        failed_cases=failed_cases,
        missing_cases=missing_cases,
        checklist=checklist,
        statistics=stats,
        case_results=case_results if include_case_details else [],
    )


def update_manifest_with_validation(
    *,
    paths: Paths,
    report: DatasetValidationReport,
) -> Path:
    """
    Update manifest.json with tool versions and validation summary.
    Returns path to updated manifest.
    """
    manifest_path = paths.raw_dir / "manifest.json"

    # Load existing manifest
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    # Add/update validation section
    manifest["validation"] = {
        "last_validated_at": report.created_at,
        "tool_versions": report.tool_versions,
        "thresholds": report.thresholds,
        "summary": {
            "total_cases": report.total_cases,
            "valid_cases": report.valid_cases,
            "warning_cases": report.warning_cases,
            "failed_cases": report.failed_cases,
            "missing_cases": report.missing_cases,
        },
        "checklist": report.checklist,
        "statistics": report.statistics,
    }

    # Write updated manifest
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return manifest_path


def save_validation_report(
    *,
    paths: Paths,
    report: DatasetValidationReport,
    filename: str = "validation_report.json",
) -> Path:
    """Save full validation report to a separate file."""
    report_path = paths.raw_dir / filename
    report_path.write_text(
        report.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return report_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_validation(
    *,
    project_root: Path,
    include_case_details: bool = True,
    update_manifest: bool = True,
    save_report: bool = True,
) -> DatasetValidationReport:
    """
    Run full dataset validation and optionally update manifest.

    This is the main entry point for Stage 4 validation.
    """
    from .config import get_paths

    paths = get_paths(project_root)
    thresholds = ValidationThresholds()

    print(f"[Stage 4] Validating dataset at: {paths.raw_dir}")
    print(f"[Stage 4] Pipeline version: {PIPELINE_VERSION}")

    # Collect tool versions
    print("[Stage 4] Collecting tool versions...")
    tool_versions = collect_tool_versions(paths=paths)
    print(f"  - gmsh: {tool_versions.get('gmsh', 'not found')}")
    print(f"  - CalculiX: {tool_versions.get('calculix', 'not found')}")
    print(f"  - Blender: {tool_versions.get('blender', 'not found')}")

    # Run validation
    print("[Stage 4] Validating cases...")
    report = validate_dataset(
        paths=paths,
        thresholds=thresholds,
        include_case_details=include_case_details,
    )

    print(f"\n[Stage 4] Validation Summary:")
    print(f"  Total cases:   {report.total_cases}")
    print(f"  Valid:         {report.valid_cases}")
    print(f"  Warnings:      {report.warning_cases}")
    print(f"  Failed:        {report.failed_cases}")
    print(f"  Missing:       {report.missing_cases}")

    print(f"\n[Stage 4] PRD Checklist:")
    for item, passed in report.checklist.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {item}")

    if save_report:
        report_path = save_validation_report(paths=paths, report=report)
        print(f"\n[Stage 4] Full report saved: {report_path}")

    if update_manifest:
        manifest_path = update_manifest_with_validation(paths=paths, report=report)
        print(f"[Stage 4] Manifest updated: {manifest_path}")

    return report


if __name__ == "__main__":
    import sys

    project_root = Path(__file__).resolve().parents[3]
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])

    report = run_validation(project_root=project_root)

    # Exit with error code if checklist fails
    all_passed = all(report.checklist.values())
    sys.exit(0 if all_passed else 1)
