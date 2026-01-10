from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .meshing import _tail, _vec_add, _vec_dot, _vec_scale, _vec_sub, parse_msh2, tri_normal_area  # noqa: PLC2701
from .viz_results import make_pressure_arrows_meshes, surface_results_to_glb, surface_results_to_glb_with_extras

# Type aliases for clarity
Vec3 = tuple[float, float, float]
Vec6 = tuple[float, float, float, float, float, float]
NodeMap = dict[int, Vec3]
StressMap = dict[int, Vec6]


@dataclass(frozen=True)
class FemArtifacts:
    inp_path: Path
    frd_path: Path
    fem_report_json: Path
    surface_results_npz: Path
    wing_result_glb: Path
    wing_result_arrows_glb: Path
    pressure_vectors_glb: Path


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_ccx_bin() -> str | None:
    env = os.environ.get("CCX_BIN")
    if env and Path(env).exists():
        return env
    which = shutil.which("ccx")
    if which:
        return which

    # Homebrew calculix-ccx installs a versioned binary like `ccx_2.22` (no plain `ccx`).
    # Try common brew prefixes.
    candidates: list[Path] = []
    for base in (
        Path("/opt/homebrew/opt/calculix-ccx"),
        Path("/usr/local/opt/calculix-ccx"),
        Path("/opt/homebrew/Cellar/calculix-ccx"),
        Path("/usr/local/Cellar/calculix-ccx"),
    ):
        if base.is_dir():
            candidates.append(base)

    # Search for ccx_* executables (prefer the newest by mtime)
    found: list[Path] = []
    for c in candidates:
        # handle both opt/<formula>/bin and Cellar/<formula>/<ver>/bin
        for p in list(c.glob("bin/ccx_*")) + list(c.glob("*/bin/ccx_*")):
            try:
                if p.is_file() and os.access(str(p), os.X_OK):
                    found.append(p)
            except Exception:
                continue

    if found:
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(found[0])

    return None


def find_ccx2paraview_bin() -> str | None:
    env = os.environ.get("CCX2PARAVIEW_BIN")
    if env and Path(env).exists():
        return env
    which = shutil.which("ccx2paraview")
    return which if which else None


def _format_id_list(ids: list[int], *, per_line: int = 16) -> str:
    if not ids:
        return ""
    lines: list[str] = []
    for i in range(0, len(ids), per_line):
        chunk = ids[i : i + per_line]
        lines.append(", ".join(str(x) for x in chunk))
    return "\n".join(lines)


def _compute_c_vol(nodes: NodeMap) -> Vec3:
    acc: Vec3 = (0.0, 0.0, 0.0)
    for p in nodes.values():
        acc = _vec_add(acc, p)
    return _vec_scale(acc, 1.0 / max(1, len(nodes)))


def compute_equivalent_nodal_loads(
    *,
    nodes: NodeMap,
    surf_upper_faces: list[list[int]],
    pressure_pa: float,
) -> dict[int, Vec3]:
    """
    Equivalent nodal forces for uniform pressure on upper faces.
    - outward normal n_hat is computed per face (robustly oriented using C_vol)
    - face force: F_f = p * A_f * (-n_hat)
    - distribute equally to 3 vertices
    """
    if pressure_pa == 0.0 or not surf_upper_faces:
        return {}

    c_vol = _compute_c_vol(nodes)
    forces: dict[int, tuple[float, float, float]] = {}

    for face in surf_upper_faces:
        if len(face) != 3:
            continue
        n0, n1, n2 = int(face[0]), int(face[1]), int(face[2])
        if n0 not in nodes or n1 not in nodes or n2 not in nodes:
            continue
        p0, p1, p2 = nodes[n0], nodes[n1], nodes[n2]
        n_hat, area = tri_normal_area(p0, p1, p2)
        if area <= 0.0:
            continue

        cf = _vec_scale(_vec_add(_vec_add(p0, p1), p2), 1.0 / 3.0)
        if _vec_dot(n_hat, _vec_sub(cf, c_vol)) < 0.0:
            n_hat = _vec_scale(n_hat, -1.0)

        # Pressure acts "into" the surface
        f_face = _vec_scale(n_hat, -pressure_pa * area)
        f_node = _vec_scale(f_face, 1.0 / 3.0)

        for nid in (n0, n1, n2):
            prev = forces.get(nid, (0.0, 0.0, 0.0))
            forces[nid] = _vec_add(prev, f_node)

    return forces


def write_inp(
    *,
    out_inp_path: Path,
    nodes: dict[int, tuple[float, float, float]],
    tets: list[tuple[int, int, int, int]],
    nroot_node_ids: list[int],
    surf_upper_faces: list[list[int]],
    pressure_pa: float,
    young_modulus_pa: float,
    poisson_ratio: float,
) -> None:
    out_inp_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = out_inp_path.stem

    nodal_forces = compute_equivalent_nodal_loads(
        nodes=nodes, surf_upper_faces=surf_upper_faces, pressure_pa=pressure_pa
    )

    lines: list[str] = []
    lines.append("*HEADING")
    lines.append(f"Deep-FEM-UAV-Wing | {base_name}")
    lines.append("*NODE")
    for nid in sorted(nodes.keys()):
        x, y, z = nodes[nid]
        lines.append(f"{nid}, {x:.12g}, {y:.12g}, {z:.12g}")

    lines.append("*ELEMENT, TYPE=C3D4, ELSET=Eall")
    for eid, (a, b, c, d) in enumerate(tets, start=1):
        lines.append(f"{eid}, {a}, {b}, {c}, {d}")

    lines.append("*NSET, NSET=Nroot")
    lines.append(_format_id_list(sorted(set(int(x) for x in nroot_node_ids))))

    lines.append("*MATERIAL, NAME=AL6061")
    lines.append("*ELASTIC")
    lines.append(f"{young_modulus_pa:.12g}, {poisson_ratio:.12g}")

    lines.append("*SOLID SECTION, ELSET=Eall, MATERIAL=AL6061")
    lines.append("")

    lines.append("*BOUNDARY")
    # Fix x,y,z
    lines.append("Nroot, 1, 3, 0")

    lines.append("*STEP")
    lines.append("*STATIC")
    # Loads must be defined within a STEP in CalculiX
    if nodal_forces:
        lines.append("*CLOAD")
        # dof: 1=x, 2=y, 3=z
        for nid in sorted(nodal_forces.keys()):
            fx, fy, fz = nodal_forces[nid]
            if abs(fx) > 0.0:
                lines.append(f"{nid}, 1, {fx:.12g}")
            if abs(fy) > 0.0:
                lines.append(f"{nid}, 2, {fy:.12g}")
            if abs(fz) > 0.0:
                lines.append(f"{nid}, 3, {fz:.12g}")
    # Output to .frd
    lines.append("*NODE FILE")
    lines.append("U")
    lines.append("*EL FILE")
    lines.append("S")
    lines.append("*END STEP")

    out_inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_ccx2paraview_results(*, work_dir: Path, job_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (points[N,3], disp[N,3], stress[N,K]).
    stress is typically [N,6] or [N,9] depending on converter.
    """
    ccx2pv = find_ccx2paraview_bin()
    if ccx2pv is None:
        raise RuntimeError("ccx2paraview not found (set CCX2PARAVIEW_BIN or install it)")

    cmd = [ccx2pv, "-i", job_name]
    completed = subprocess.run(cmd, cwd=str(work_dir), capture_output=True, text=True, check=False, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(f"ccx2paraview failed: code={completed.returncode} stderr={_tail(completed.stderr or '')}")

    # Common outputs: {job}.vtu (or multiple time steps). Prefer the latest modified .vtu.
    vtu_files = sorted(work_dir.glob("*.vtu"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not vtu_files:
        raise RuntimeError("ccx2paraview succeeded but no .vtu found")
    vtu_path = vtu_files[0]

    import pyvista as pv

    grid = pv.read(str(vtu_path))
    points = np.asarray(grid.points, dtype=np.float64)

    # displacement
    disp = None
    for name in grid.point_data.keys():
        if "disp" in name.lower() or name.lower() in {"u", "displacement"}:
            arr = np.asarray(grid.point_data[name])
            if arr.ndim == 2 and arr.shape[1] == 3:
                disp = arr.astype(np.float64)
                break
    if disp is None:
        # allow missing; treat as zeros
        disp = np.zeros_like(points)

    # stress (try point data first)
    stress = None
    for name in grid.point_data.keys():
        if "stress" in name.lower() or name.lower() in {"s"}:
            arr = np.asarray(grid.point_data[name])
            if arr.ndim == 2 and arr.shape[1] in (6, 9):
                stress = arr.astype(np.float64)
                break
    if stress is None:
        stress = np.zeros((points.shape[0], 6), dtype=np.float64)

    return points, disp, stress


_SCI_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def _parse_frd_section(*, frd_path: Path, section_name: str, ncomp: int) -> dict[int, tuple[float, ...]]:
    """
    Parse a FRD nodal results section, e.g.:
      -4  DISP        4    1
      ...
      -1 <node_id> <v1> <v2> <v3>

    We scan the whole file and keep the LAST occurrence (final step).
    """
    text = frd_path.read_text(encoding="utf-8", errors="ignore")
    if "\x00" in text:
        raise RuntimeError("FRD appears to be binary; ASCII parser cannot read it.")

    out: dict[int, tuple[float, ...]] = {}
    in_section = False

    for raw in text.splitlines():
        line = raw.rstrip("\n")

        if line.startswith("-4") or line.startswith(" -4"):
            # Example: -4  DISP        4    1
            tokens = line.split()
            in_section = len(tokens) >= 2 and tokens[1].strip().upper() == section_name.upper()
            continue

        if in_section and (line.startswith("-3") or line.startswith(" -3") or line.startswith("    1PSTEP")):
            in_section = False
            continue

        if not in_section:
            continue

        if line.startswith("-1") or line.startswith(" -1"):
            # node id + values (note: values may be glued without spaces)
            m = re.match(r"^\s*-1\s+(\d+)\s*(.*)$", line)
            if not m:
                continue
            nid = int(m.group(1))
            rest = m.group(2)
            nums = _SCI_FLOAT_RE.findall(rest)
            if len(nums) < ncomp:
                continue
            vals = tuple(float(nums[i]) for i in range(ncomp))
            out[nid] = vals

    if not out:
        raise RuntimeError(f"FRD section not found or empty: {section_name}")
    return out


def _read_frd_ascii_nodal_results(*, frd_path: Path) -> tuple[NodeMap, StressMap]:
    """Read displacement and stress data from FRD ASCII file."""
    disp = _parse_frd_section(frd_path=frd_path, section_name="DISP", ncomp=3)
    stress = _parse_frd_section(frd_path=frd_path, section_name="STRESS", ncomp=6)
    # type narrowing
    disp3: NodeMap = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in disp.items()}
    s6: StressMap = {k: (float(v[0]), float(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5])) for k, v in stress.items()}
    return disp3, s6


def _stress_to_von_mises(stress: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convert stress components to von Mises.
    Supports:
    - [N,6] with [sxx, syy, szz, sxy, syz, szx]
    - [N,9] with row-major 3x3 tensor (sxx,sxy,sxz, syx,syy,syz, szx,szy,szz)
    """
    if stress.ndim != 2:
        return np.zeros((stress.shape[0],), dtype=np.float64)

    if stress.shape[1] == 6:
        sxx, syy, szz, sxy, syz, szx = (stress[:, 0], stress[:, 1], stress[:, 2], stress[:, 3], stress[:, 4], stress[:, 5])
    elif stress.shape[1] == 9:
        sxx = stress[:, 0]
        syy = stress[:, 4]
        szz = stress[:, 8]
        sxy = 0.5 * (stress[:, 1] + stress[:, 3])
        syz = 0.5 * (stress[:, 5] + stress[:, 7])
        szx = 0.5 * (stress[:, 2] + stress[:, 6])
    else:
        return np.zeros((stress.shape[0],), dtype=np.float64)

    vm2 = 0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * (sxy**2 + syz**2 + szx**2)
    vm2 = np.maximum(vm2, 0.0)
    return np.sqrt(vm2)


def _map_by_quantized_points(
    *, src_points: np.ndarray, dst_points: np.ndarray, values: np.ndarray, decimals: int = 9
) -> np.ndarray:
    """
    Map values defined on src_points to dst_points by exact match after rounding.
    This avoids scipy/kdtree and is usually enough because mesh points are identical.
    """
    if src_points.shape[0] != values.shape[0]:
        raise ValueError("src_points and values length mismatch")

    def key(p: np.ndarray) -> tuple[float, float, float]:
        return (round(float(p[0]), decimals), round(float(p[1]), decimals), round(float(p[2]), decimals))

    lut: dict[tuple[float, float, float], int] = {}
    for i in range(src_points.shape[0]):
        lut[key(src_points[i])] = i

    out = np.zeros((dst_points.shape[0],) + values.shape[1:], dtype=np.float64)
    miss = 0
    for j in range(dst_points.shape[0]):
        idx = lut.get(key(dst_points[j]))
        if idx is None:
            miss += 1
            continue
        out[j] = values[idx]
    # If many misses, try a looser rounding
    if miss > 0 and miss / max(1, dst_points.shape[0]) > 0.01 and decimals > 6:
        return _map_by_quantized_points(src_points=src_points, dst_points=dst_points, values=values, decimals=6)
    return out


def _compute_surface_normals(
    *, nodes: NodeMap, surf_all_faces: list[list[int]]
) -> NodeMap:
    """Compute area-weighted average normals for surface nodes."""
    c_vol = _compute_c_vol(nodes)
    acc: dict[int, Vec3] = {}

    for face in surf_all_faces:
        if len(face) != 3:
            continue
        n0, n1, n2 = int(face[0]), int(face[1]), int(face[2])
        if n0 not in nodes or n1 not in nodes or n2 not in nodes:
            continue
        p0, p1, p2 = nodes[n0], nodes[n1], nodes[n2]
        n_hat, area = tri_normal_area(p0, p1, p2)
        if area <= 0.0:
            continue
        cf = _vec_scale(_vec_add(_vec_add(p0, p1), p2), 1.0 / 3.0)
        if _vec_dot(n_hat, _vec_sub(cf, c_vol)) < 0.0:
            n_hat = _vec_scale(n_hat, -1.0)
        # area-weighted accumulate
        n_w = _vec_scale(n_hat, area)
        for nid in (n0, n1, n2):
            prev = acc.get(nid, (0.0, 0.0, 0.0))
            acc[nid] = _vec_add(prev, n_w)

    # normalize
    out: NodeMap = {}
    for nid, v in acc.items():
        norm = math.sqrt(_vec_dot(v, v))
        if norm <= 0.0:
            out[nid] = (0.0, 0.0, 0.0)
        else:
            out[nid] = _vec_scale(v, 1.0 / norm)
    return out


def run_fem_case(
    *,
    case_id: str,
    geometry_dir: Path,
    mesh_dir: Path,
    fem_dir: Path,
    young_modulus_pa: float = 69e9,
    poisson_ratio: float = 0.33,
    pressure_pa: float = 5e3,
    timeout_s: int = 900,
) -> tuple[bool, dict[str, Any], FemArtifacts | None]:
    """
    Stage 3: Run CalculiX FEM analysis.

    Args:
        case_id: Unique case identifier
        geometry_dir: Path to geometry data directory
        mesh_dir: Path to mesh data directory
        fem_dir: Path to FEM output directory
        young_modulus_pa: Young's modulus in Pascal (default: 69 GPa for AL6061)
        poisson_ratio: Poisson's ratio (default: 0.33 for AL6061)
        pressure_pa: Uniform pressure on upper surface in Pascal
        timeout_s: Solver timeout in seconds

    Returns:
        Tuple of (success, report_dict, artifacts_or_none)
    """
    start = time.perf_counter()

    case_mesh_dir = mesh_dir / case_id
    wing_msh = case_mesh_dir / "wing.msh"
    boundary_sets_json = case_mesh_dir / "boundary_sets.json"
    if not wing_msh.exists() or not boundary_sets_json.exists():
        return False, {"status": "failed", "failure_reason": "missing wing.msh or boundary_sets.json"}, None

    ccx_bin = find_ccx_bin()
    if ccx_bin is None:
        return False, {"status": "failed", "failure_reason": "ccx not found (set CCX_BIN or install CalculiX)"}, None

    # Parse mesh
    try:
        nodes, _triangles, tets = parse_msh2(wing_msh)
    except ValueError as e:
        return False, {"status": "failed", "failure_reason": f"msh parse failed: {e}"}, None
    except (OSError, IOError) as e:
        return False, {"status": "failed", "failure_reason": f"msh file read error: {e}"}, None

    try:
        boundary_sets = json.loads(boundary_sets_json.read_text(encoding="utf-8"))
        nroot_node_ids = list(map(int, boundary_sets.get("nroot_node_ids") or []))
        surf_all_faces = boundary_sets.get("surf_all_faces") or []
        surf_upper_faces = boundary_sets.get("surf_upper_faces") or []
        y_tol = float(boundary_sets.get("y_tol") or 1e-4)
    except json.JSONDecodeError as e:
        return False, {"status": "failed", "failure_reason": f"boundary_sets.json invalid JSON: {e}"}, None
    except (KeyError, TypeError, ValueError) as e:
        return False, {"status": "failed", "failure_reason": f"boundary_sets.json data error: {e}"}, None
    except (OSError, IOError) as e:
        return False, {"status": "failed", "failure_reason": f"boundary_sets.json read error: {e}"}, None

    if not nroot_node_ids:
        return False, {"status": "failed", "failure_reason": "NROOT empty"}, None
    if not surf_all_faces:
        return False, {"status": "failed", "failure_reason": "SURF_ALL empty"}, None
    if not surf_upper_faces:
        return False, {"status": "failed", "failure_reason": "SURF_UPPER empty"}, None

    # FEM output structure
    case_fem_dir = fem_dir / case_id
    case_fem_dir.mkdir(parents=True, exist_ok=True)
    job_name = case_id
    inp_path = case_fem_dir / f"{job_name}.inp"
    frd_path = case_fem_dir / f"{job_name}.frd"
    fem_report_json = case_fem_dir / "fem_report.json"
    surface_results_npz = case_fem_dir / "surface_results.npz"
    wing_result_glb = case_fem_dir / "wing_result.glb"
    wing_result_arrows_glb = case_fem_dir / "wing_result_arrows.glb"
    pressure_vectors_glb = case_fem_dir / "pressure_vectors.glb"

    # Write inp
    try:
        write_inp(
            out_inp_path=inp_path,
            nodes=nodes,
            tets=tets,
            nroot_node_ids=nroot_node_ids,
            surf_upper_faces=surf_upper_faces,
            pressure_pa=pressure_pa,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
        )
    except (OSError, IOError) as e:
        return False, {"status": "failed", "failure_reason": f"inp file write error: {e}"}, None
    except (ValueError, TypeError) as e:
        return False, {"status": "failed", "failure_reason": f"inp generation data error: {e}"}, None

    # Run ccx
    cmd = [ccx_bin, "-i", job_name]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(case_fem_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        report = {
            "status": "failed",
            "failure_reason": f"ccx timeout after {timeout_s}s",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
        }
        _write_json(fem_report_json, report)
        return False, report, None
    except OSError as e:
        report = {
            "status": "failed",
            "failure_reason": f"ccx execution OS error: {e}",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
        }
        _write_json(fem_report_json, report)
        return False, report, None

    stdout_tail = _tail(completed.stdout or "")
    stderr_tail = _tail(completed.stderr or "")

    if completed.returncode != 0:
        report = {
            "status": "failed",
            "failure_reason": f"ccx exited with code {completed.returncode}",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_json(fem_report_json, report)
        return False, report, None

    if not frd_path.exists():
        report = {
            "status": "failed",
            "failure_reason": "ccx succeeded but .frd not found",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_json(fem_report_json, report)
        return False, report, None

    # Postprocess: prefer direct FRD ASCII parsing (no extra tools),
    # fallback to ccx2paraview if available.
    frd_parse_error: str | None = None
    try:
        disp_map, stress_map = _read_frd_ascii_nodal_results(frd_path=frd_path)
    except RuntimeError as e:
        frd_parse_error = str(e)
        # fallback to ccx2paraview (if installed)
        try:
            pts, disp, stress = _read_ccx2paraview_results(work_dir=case_fem_dir, job_name=job_name)
            disp_map = {i + 1: (float(disp[i, 0]), float(disp[i, 1]), float(disp[i, 2])) for i in range(disp.shape[0])}
            stress_vm = _stress_to_von_mises(stress)
            stress_map = {i + 1: (float(stress[i, 0]), float(stress[i, 1]), float(stress[i, 2]), float(stress[i, 3]), float(stress[i, 4]), float(stress[i, 5])) for i in range(stress.shape[0])}
        except RuntimeError as e2:
            report = {
                "status": "failed",
                "failure_reason": f"postprocess failed: frd_ascii={frd_parse_error} ccx2paraview={e2}",
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            _write_json(fem_report_json, report)
            return False, report, None
        except ImportError as e2:
            report = {
                "status": "failed",
                "failure_reason": f"postprocess failed: frd_ascii={frd_parse_error}, pyvista not available: {e2}",
                "elapsed_ms": int((time.perf_counter() - start) * 1000),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
            _write_json(fem_report_json, report)
            return False, report, None

    # Surface node ids
    surface_node_ids = sorted({int(x) for face in surf_all_faces for x in (face or [])})
    surface_pos = np.asarray([nodes[nid] for nid in surface_node_ids], dtype=np.float64)

    # Use FRD node-id mapping (preferred)
    surface_disp = np.asarray([disp_map.get(nid, (0.0, 0.0, 0.0)) for nid in surface_node_ids], dtype=np.float64)
    surface_stress6 = np.asarray([stress_map.get(nid, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) for nid in surface_node_ids], dtype=np.float64)
    surface_vm = _stress_to_von_mises(surface_stress6)

    # Validate results: check for nan/inf
    has_nan_disp = bool(np.any(np.isnan(surface_disp)) or np.any(np.isinf(surface_disp)))
    has_nan_stress = bool(np.any(np.isnan(surface_vm)) or np.any(np.isinf(surface_vm)))
    if has_nan_disp or has_nan_stress:
        report = {
            "status": "failed",
            "failure_reason": f"invalid results: nan/inf detected (disp={has_nan_disp}, stress={has_nan_stress})",
            "elapsed_ms": int((time.perf_counter() - start) * 1000),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
        _write_json(fem_report_json, report)
        return False, report, None

    # Normals on surface
    normals_map = _compute_surface_normals(nodes=nodes, surf_all_faces=surf_all_faces)
    surface_norm = np.asarray([normals_map.get(nid, (0.0, 0.0, 0.0)) for nid in surface_node_ids], dtype=np.float64)

    # loss_mask: root singularity band (y < 0.05*span)
    ys = np.asarray([nodes[nid][1] for nid in surface_node_ids], dtype=np.float64)
    span = float(np.max(ys) - np.min(ys)) if ys.size else 0.0
    loss_mask = ys > (0.05 * span) if span > 0.0 else np.ones_like(ys, dtype=bool)

    np.savez_compressed(
        surface_results_npz,
        node_id=np.asarray(surface_node_ids, dtype=np.int64),
        pos=surface_pos,
        normal=surface_norm,
        stress_vm=surface_vm,
        disp=surface_disp,
        loss_mask=loss_mask.astype(bool),
    )

    # wing_result.glb from surface results
    surface_results_to_glb(
        glb_path=wing_result_glb,
        node_id=np.asarray(surface_node_ids, dtype=np.int64),
        pos=surface_pos,
        faces=surf_all_faces,
        stress_vm=surface_vm,
        mask=loss_mask,
    )

    # Pressure arrows (sampled faces)
    arrows = make_pressure_arrows_meshes(
        nodes=nodes,
        surf_upper_faces=surf_upper_faces,
        sample_n=200,
        seed=int(case_id[:8], 16) if case_id and len(case_id) >= 8 else 0,
    )
    if arrows:
        # arrows-only GLB
        try:
            import trimesh  # type: ignore

            scene = trimesh.Scene()
            for m in arrows:
                scene.add_geometry(m)
            pressure_vectors_glb.write_bytes(scene.export(file_type="glb"))
        except Exception:
            pass

        # combined GLB (wing + arrows)
        try:
            surface_results_to_glb_with_extras(
                glb_path=wing_result_arrows_glb,
                node_id=np.asarray(surface_node_ids, dtype=np.int64),
                pos=surface_pos,
                faces=surf_all_faces,
                stress_vm=surface_vm,
                mask=loss_mask,
                extras=arrows,
            )
        except Exception:
            pass

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    report = {
        "status": "success",
        "failure_reason": "",
        "elapsed_ms": elapsed_ms,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "artifacts": {
            "inp_path": str(inp_path),
            "frd_path": str(frd_path),
            "surface_results_npz": str(surface_results_npz),
            "wing_result_glb": str(wing_result_glb),
            "wing_result_arrows_glb": str(wing_result_arrows_glb) if wing_result_arrows_glb.exists() else "",
            "pressure_vectors_glb": str(pressure_vectors_glb) if pressure_vectors_glb.exists() else "",
        },
        "params": {
            "young_modulus_pa": young_modulus_pa,
            "poisson_ratio": poisson_ratio,
            "pressure_pa": pressure_pa,
        },
        "stats": {
            "surface_nodes": len(surface_node_ids),
            "min_stress_vm": float(np.min(surface_vm)) if surface_vm.size else 0.0,
            "max_stress_vm": float(np.max(surface_vm)) if surface_vm.size else 0.0,
            "mean_stress_vm": float(np.mean(surface_vm)) if surface_vm.size else 0.0,
            "min_disp_mag": float(np.min(np.linalg.norm(surface_disp, axis=1))) if surface_disp.size else 0.0,
            "max_disp_mag": float(np.max(np.linalg.norm(surface_disp, axis=1))) if surface_disp.size else 0.0,
            "mean_disp_z": float(np.mean(surface_disp[:, 2])) if surface_disp.size else 0.0,
        },
        "metrics_all_nodes": {
            "count": int(surface_vm.size),
            "stress_min": float(np.min(surface_vm)) if surface_vm.size else 0.0,
            "stress_max": float(np.max(surface_vm)) if surface_vm.size else 0.0,
            "stress_mean": float(np.mean(surface_vm)) if surface_vm.size else 0.0,
            "stress_std": float(np.std(surface_vm)) if surface_vm.size else 0.0,
            "stress_p50": float(np.percentile(surface_vm, 50)) if surface_vm.size else 0.0,
            "stress_p95": float(np.percentile(surface_vm, 95)) if surface_vm.size else 0.0,
            "stress_p99": float(np.percentile(surface_vm, 99)) if surface_vm.size else 0.0,
        },
        "metrics_masked_nodes": {
            "count": int(np.sum(loss_mask)),
            "stress_min": float(np.min(surface_vm[loss_mask])) if np.any(loss_mask) else 0.0,
            "stress_max": float(np.max(surface_vm[loss_mask])) if np.any(loss_mask) else 0.0,
            "stress_mean": float(np.mean(surface_vm[loss_mask])) if np.any(loss_mask) else 0.0,
            "stress_std": float(np.std(surface_vm[loss_mask])) if np.any(loss_mask) else 0.0,
            "stress_p50": float(np.percentile(surface_vm[loss_mask], 50)) if np.any(loss_mask) else 0.0,
            "stress_p95": float(np.percentile(surface_vm[loss_mask], 95)) if np.any(loss_mask) else 0.0,
            "stress_p99": float(np.percentile(surface_vm[loss_mask], 99)) if np.any(loss_mask) else 0.0,
        },
        "surface_mask": {
            "loss_mask_true": int(np.sum(loss_mask)),
            "loss_mask_false": int(loss_mask.size - int(np.sum(loss_mask))),
            "root_band_y": float(0.05 * span),
        },
        "notes": "postprocess via FRD ASCII parsing (DISP/STRESS) mapped by node_id",
    }
    _write_json(fem_report_json, report)

    artifacts = FemArtifacts(
        inp_path=inp_path,
        frd_path=frd_path,
        fem_report_json=fem_report_json,
        surface_results_npz=surface_results_npz,
        wing_result_glb=wing_result_glb,
        wing_result_arrows_glb=wing_result_arrows_glb,
        pressure_vectors_glb=pressure_vectors_glb,
    )
    return True, report, artifacts


