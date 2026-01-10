from __future__ import annotations

import json
import math
import os
import subprocess
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeshArtifacts:
    wing_msh: Path
    mesh_report_json: Path
    boundary_sets_json: Path
    surf_sets_glb: Path


def _tail(text: str, *, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def find_gmsh_bin() -> str | None:
    env = os.environ.get("GMSH_BIN")
    if env and Path(env).exists():
        return env
    which = shutil.which("gmsh")
    return which if which else None


def run_gmsh_tetra(
    *,
    stl_path: Path,
    out_msh_path: Path,
    mesh_size_factor: float = 1.0,
    timeout_s: int = 600,
) -> tuple[bool, str, str, str]:
    gmsh_bin = find_gmsh_bin()
    if gmsh_bin is None:
        return False, "gmsh not found (set GMSH_BIN or install gmsh)", "", ""

    out_msh_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a generated .geo to reliably create a Volume from STL, then tetra-mesh it.
    # Direct `gmsh wing.stl -3` may create only surface triangles if volume is not defined.
    geo_path = out_msh_path.with_suffix(".geo")
    geo_text = "\n".join(
        [
            f'Merge "{stl_path.resolve()}";',
            # angle (deg) to detect sharp edges. 40 is a reasonable default.
            # NOTE: Gmsh uses `Pi` (not `pi`)
            "ClassifySurfaces{40*Pi/180, 1, 1, 180*Pi/180};",
            "CreateGeometry;",
            "Surface Loop(1) = Surface{:};",
            "Volume(1) = {1};",
            # Ensure MSH2
            "Mesh.MshFileVersion = 2.2;",
            "Mesh 3;",
        ]
    )
    geo_path.write_text(geo_text + "\n", encoding="utf-8")

    cmd = [
        gmsh_bin,
        str(geo_path),
        "-3",
        "-order",
        "1",
        "-format",
        "msh2",
        "-o",
        str(out_msh_path),
        "-clscale",
        str(mesh_size_factor),
        "-v",
        "5",
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"gmsh execution failed: {e}", "", ""

    stdout_tail = _tail(completed.stdout or "")
    stderr_tail = _tail(completed.stderr or "")

    if completed.returncode != 0:
        return False, f"gmsh exited with code {completed.returncode}", stdout_tail, stderr_tail

    if not out_msh_path.exists():
        return False, "gmsh reported success but wing.msh not found", stdout_tail, stderr_tail

    return True, "gmsh meshing success", stdout_tail, stderr_tail


def parse_msh2(path: Path) -> tuple[dict[int, tuple[float, float, float]], list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    """
    Minimal Gmsh MSH v2 parser for nodes + triangle/tetra elements.
    - nodes: {node_id: (x,y,z)}
    - triangles: [(n1,n2,n3)]
    - tets: [(n1,n2,n3,n4)]
    """
    nodes: dict[int, tuple[float, float, float]] = {}
    triangles: list[tuple[int, int, int]] = []
    tets: list[tuple[int, int, int, int]] = []

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "$Nodes":
            i += 1
            n = int(lines[i].strip())
            i += 1
            for _ in range(n):
                parts = lines[i].split()
                node_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes[node_id] = (x, y, z)
                i += 1
            # skip $EndNodes
            while i < len(lines) and lines[i].strip() != "$EndNodes":
                i += 1
        elif line == "$Elements":
            i += 1
            n = int(lines[i].strip())
            i += 1
            for _ in range(n):
                parts = lines[i].split()
                elem_type = int(parts[1])
                ntags = int(parts[2])
                node_start = 3 + ntags
                node_ids = list(map(int, parts[node_start:]))
                # Triangles:
                # - 2: 3-node triangle
                # - 9: 6-node second-order triangle (use corner nodes)
                if elem_type == 2 and len(node_ids) >= 3:
                    triangles.append((node_ids[0], node_ids[1], node_ids[2]))
                elif elem_type == 9 and len(node_ids) >= 3:
                    triangles.append((node_ids[0], node_ids[1], node_ids[2]))
                # Tetrahedra:
                # - 4: 4-node tetra
                # - 11: 10-node second-order tetra (use corner nodes)
                elif elem_type == 4 and len(node_ids) >= 4:
                    tets.append((node_ids[0], node_ids[1], node_ids[2], node_ids[3]))
                elif elem_type == 11 and len(node_ids) >= 4:
                    tets.append((node_ids[0], node_ids[1], node_ids[2], node_ids[3]))
                i += 1
            while i < len(lines) and lines[i].strip() != "$EndElements":
                i += 1
        i += 1

    if not nodes:
        raise ValueError("no nodes parsed from msh")
    if not tets:
        raise ValueError("no tetra elements parsed from msh")
    if not triangles:
        raise ValueError("no boundary triangles parsed from msh")

    return nodes, triangles, tets


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_vec_dot(a, a))


def _vec_scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def tri_normal_area(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> tuple[tuple[float, float, float], float]:
    e1 = _vec_sub(p1, p0)
    e2 = _vec_sub(p2, p0)
    n = _vec_cross(e1, e2)
    area = 0.5 * _vec_norm(n)
    if area <= 0:
        return (0.0, 0.0, 0.0), 0.0
    unit = _vec_scale(n, 1.0 / (2.0 * area))
    return unit, area


def orient_surface_consistently(
    nodes: dict[int, tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> list[tuple[int, int, int]]:
    """
    Enforces consistent winding order for all connected triangles and orients them outward.
    - First, enforce consistent winding by DFS over face adjacency (shared edges).
    - For each connected component, choose a seed face and orient it so that
      dot(n, C_f - C_vol) > 0 (outward w.r.t. volume centroid).
    - Then propagate orientation to neighbors (shared edge must be opposite direction).
    """
    if not triangles:
        return []

    # Approx volume centroid (mean of all node positions)
    acc = (0.0, 0.0, 0.0)
    for p in nodes.values():
        acc = _vec_add(acc, p)
    c_vol = _vec_scale(acc, 1.0 / max(1, len(nodes)))

    # 1. Build adjacency (edge -> list of face_indices)
    edge_to_faces: dict[frozenset[int], list[int]] = {}
    for idx, (a, b, c) in enumerate(triangles):
        for pair in [(a, b), (b, c), (c, a)]:
            edge = frozenset(pair)
            edge_to_faces.setdefault(edge, []).append(idx)

    visited = [False] * len(triangles)
    new_triangles = [None] * len(triangles)

    def has_directed_edge(tri: tuple[int, int, int], u: int, v: int) -> bool:
        return (tri[0] == u and tri[1] == v) or (tri[1] == u and tri[2] == v) or (tri[2] == u and tri[0] == v)

    # Orient all connected components
    for seed_idx in range(len(triangles)):
        if visited[seed_idx]:
            continue

        # Seed: choose outward w.r.t c_vol
        seed = triangles[seed_idx]
        p0, p1, p2 = nodes[seed[0]], nodes[seed[1]], nodes[seed[2]]
        n_seed, area = tri_normal_area(p0, p1, p2)
        if area > 0:
            cf = _vec_scale(_vec_add(_vec_add(p0, p1), p2), 1.0 / 3.0)
            if _vec_dot(n_seed, _vec_sub(cf, c_vol)) < 0:
                seed = (seed[0], seed[2], seed[1])

        new_triangles[seed_idx] = seed
        visited[seed_idx] = True
        stack = [seed_idx]

        while stack:
            current_idx = stack.pop()
            curr_tri = new_triangles[current_idx]
            if curr_tri is None:
                continue

            directed_edges = [
                (curr_tri[0], curr_tri[1]),
                (curr_tri[1], curr_tri[2]),
                (curr_tri[2], curr_tri[0]),
            ]

            for u, v in directed_edges:
                neighbors = edge_to_faces.get(frozenset((u, v)), [])
                for n_idx in neighbors:
                    if visited[n_idx]:
                        continue
                    nbr = triangles[n_idx]
                    # Neighbor should have opposite direction along the shared edge.
                    # If it has u->v (same), flip.
                    if has_directed_edge(nbr, u, v):
                        nbr = (nbr[0], nbr[2], nbr[1])
                    new_triangles[n_idx] = nbr
                    visited[n_idx] = True
                    stack.append(n_idx)

    final_triangles: list[tuple[int, int, int]] = []
    for i in range(len(triangles)):
        if new_triangles[i] is not None:
            final_triangles.append(new_triangles[i])  # type: ignore
        else:
            final_triangles.append(triangles[i])
            
    return final_triangles


def build_boundary_sets(
    *,
    nodes: dict[int, tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    y_tol: float = 1e-4,
    nz_min: float = 0.2,
) -> tuple[dict[str, Any], str]:
    # Robustly orient triangles to point outward
    consistent_triangles = orient_surface_consistently(nodes, triangles)

    nroot = [nid for nid, p in nodes.items() if p[1] <= y_tol]

    def classify(nz_threshold: float) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
        surf_all_faces: list[list[int]] = []
        surf_upper_faces: list[list[int]] = []
        surf_all_area = 0.0
        surf_upper_area = 0.0

        face_areas: list[float] = []

        for (n0, n1, n2) in consistent_triangles:
            p0, p1, p2 = nodes[n0], nodes[n1], nodes[n2]
            n_unit, area = tri_normal_area(p0, p1, p2)
            if area <= 0:
                continue

            surf_all_faces.append([n0, n1, n2])
            face_areas.append(area)
            surf_all_area += area
            
            cf = _vec_scale(_vec_add(_vec_add(p0, p1), p2), 1.0 / 3.0)

            # Root 근방 제외는 face의 "중심" 기준으로 제한한다.
            # min_y 기준은 루트 경계와 연결된 표면 strip까지 과도하게 제외될 수 있다.
            if cf[1] <= y_tol * 5.0:
                continue
            if n_unit[2] >= nz_threshold:
                surf_upper_faces.append([n0, n1, n2])
                surf_upper_area += area

        # Postprocess: keep only the largest connected component among candidate upper faces.
        # This removes "holes/speckles" caused by threshold noise.
        if surf_upper_faces:
            # Build adjacency of ALL surface faces by shared edges
            # Use surf_all_faces indexing (consistent with face_areas)
            edge_to_faces: dict[frozenset[int], list[int]] = {}
            for fi, (a, b, c) in enumerate(surf_all_faces):
                for e in (frozenset((a, b)), frozenset((b, c)), frozenset((c, a))):
                    edge_to_faces.setdefault(e, []).append(fi)

            adj: list[list[int]] = [[] for _ in range(len(surf_all_faces))]
            for faces in edge_to_faces.values():
                if len(faces) < 2:
                    continue
                # connect all faces sharing this edge (usually 2)
                for i in range(len(faces)):
                    for j in range(i + 1, len(faces)):
                        a = faces[i]
                        b = faces[j]
                        adj[a].append(b)
                        adj[b].append(a)

            upper_set = {frozenset(face) for face in surf_upper_faces}
            is_upper = [frozenset(face) in upper_set for face in surf_all_faces]

            visited = [False] * len(surf_all_faces)
            best_component: list[int] = []
            best_area = -1.0

            for start in range(len(surf_all_faces)):
                if visited[start] or not is_upper[start]:
                    continue
                stack = [start]
                visited[start] = True
                comp: list[int] = []
                comp_area = 0.0
                while stack:
                    cur = stack.pop()
                    comp.append(cur)
                    comp_area += face_areas[cur]
                    for nb in adj[cur]:
                        if visited[nb] or not is_upper[nb]:
                            continue
                        visited[nb] = True
                        stack.append(nb)
                if comp_area > best_area:
                    best_area = comp_area
                    best_component = comp

            # Rebuild surf_upper_faces with the best component only
            if best_component:
                surf_upper_faces = [surf_all_faces[i] for i in best_component]
                surf_upper_area = sum(face_areas[i] for i in best_component)

        ratio = (surf_upper_area / surf_all_area) if surf_all_area > 0 else 0.0
        stats = {
            "nroot_count": len(nroot),
            "surf_all_area": surf_all_area,
            "surf_upper_area": surf_upper_area,
            "surf_upper_ratio": ratio,
        }
        return surf_all_faces, surf_upper_faces, stats

    # Auto-tuning (lower-bound only):
    # - If SURF_UPPER ratio is too small (<0.2), relax nz_min step-by-step until it reaches the lower bound.
    # - No upper bound: if ratio is large, we accept it.
    tuning_history: list[dict[str, float]] = []
    used_nz_min = nz_min
    note_parts: list[str] = []

    surf_all_faces, surf_upper_faces, stats = classify(used_nz_min)
    tuning_history.append({"nz_min": float(used_nz_min), "surf_upper_ratio": float(stats["surf_upper_ratio"])})

    lower_bound = -0.2
    relax_step = 0.05
    max_iters = 12

    iters = 0
    while stats["surf_upper_ratio"] < 0.2 and iters < max_iters and used_nz_min > lower_bound:
        iters += 1
        used_nz_min = max(lower_bound, used_nz_min - relax_step)
        surf_all_faces, surf_upper_faces, stats = classify(used_nz_min)
        tuning_history.append({"nz_min": float(used_nz_min), "surf_upper_ratio": float(stats["surf_upper_ratio"])})

    if iters > 0:
        note_parts.append(
            f"auto-tune applied: nz_min {nz_min} → {used_nz_min} (iters={iters}, step={relax_step}, lower_bound={lower_bound})"
        )

    if stats["nroot_count"] <= 0:
        raise ValueError("NROOT is empty (check root alignment y=0 and y_tol)")
    if stats["surf_upper_ratio"] < 0.2:
        raise ValueError("SURF_UPPER ratio < 0.2 (upper separation failed)")

    boundary_sets = {
        "y_tol": y_tol,
        "nz_min": used_nz_min,
        "nroot_node_ids": sorted(nroot),
        "surf_all_faces": surf_all_faces,
        "surf_upper_faces": surf_upper_faces,
        "stats": stats,
        "tuning": {
            "mode": "lower_bound_only",
            "history": tuning_history,
        },
    }
    return boundary_sets, "\n".join(note_parts).strip()


def build_surf_sets_glb(
    *,
    nodes: dict[int, tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    boundary_sets: dict[str, Any],
    out_glb: Path,
) -> None:
    """
    Debug visualization:
    - surface mesh only
    - vertex colors:
        - default: light gray
        - NROOT nodes: red
        - SURF_UPPER face nodes: blue (overrides default, not root)
    """
    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("trimesh is required to export debug GLB") from e

    # IMPORTANT:
    # Use per-face (flat) coloring by duplicating vertices per triangle.
    # Vertex-color interpolation can make Root(red) "bleed" into adjacent faces, which is confusing for debugging.
    y_tol = float(boundary_sets.get("y_tol", 1e-4))
    nroot_set = set(boundary_sets.get("nroot_node_ids", []))
    # Face membership must be order-independent (node ordering can differ between writers/readers).
    upper_faces_set = {frozenset(face) for face in boundary_sets.get("surf_upper_faces", [])}

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    colors: list[list[int]] = []

    for (a, b, c) in triangles:
        p0, p1, p2 = nodes[a], nodes[b], nodes[c]
        max_y = max(p0[1], p1[1], p2[1])

        is_root_face = (a in nroot_set and b in nroot_set and c in nroot_set) or (max_y <= y_tol * 5.0)
        is_upper_face = frozenset((a, b, c)) in upper_faces_set

        if is_root_face:
            rgba = [230, 57, 70, 255]  # red
        elif is_upper_face:
            rgba = [29, 78, 216, 255]  # blue
        else:
            rgba = [200, 200, 200, 255]  # gray

        base = len(vertices)
        vertices.extend([p0, p1, p2])
        faces.append((base, base + 1, base + 2))
        colors.extend([rgba, rgba, rgba])

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.vertex_colors = colors
    scene = trimesh.Scene(mesh)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_glb.write_bytes(scene.export(file_type="glb"))


def compute_tet_quality(
    nodes: dict[int, tuple[float, float, float]],
    tets: list[tuple[int, int, int, int]],
) -> dict[str, float]:
    """
    Compute tetrahedron mesh quality metrics.
    Returns min/mean/max of aspect ratio (ideal=1.0, higher=worse).
    Aspect ratio = longest_edge / (shortest_altitude * sqrt(6))
    For a regular tetrahedron, aspect ratio = 1.0.
    """
    if not tets:
        return {"min_aspect_ratio": 0.0, "mean_aspect_ratio": 0.0, "max_aspect_ratio": 0.0, "quality_ok_ratio": 0.0}

    aspect_ratios: list[float] = []

    for (n0, n1, n2, n3) in tets:
        if n0 not in nodes or n1 not in nodes or n2 not in nodes or n3 not in nodes:
            continue
        p0, p1, p2, p3 = nodes[n0], nodes[n1], nodes[n2], nodes[n3]

        # Compute all 6 edge lengths
        edges = [
            _vec_norm(_vec_sub(p1, p0)),
            _vec_norm(_vec_sub(p2, p0)),
            _vec_norm(_vec_sub(p3, p0)),
            _vec_norm(_vec_sub(p2, p1)),
            _vec_norm(_vec_sub(p3, p1)),
            _vec_norm(_vec_sub(p3, p2)),
        ]
        max_edge = max(edges)
        min_edge = min(edges)

        if min_edge <= 1e-15:
            aspect_ratios.append(1e6)  # degenerate
            continue

        # Compute volume using scalar triple product
        v01 = _vec_sub(p1, p0)
        v02 = _vec_sub(p2, p0)
        v03 = _vec_sub(p3, p0)
        cross = _vec_cross(v01, v02)
        volume = abs(_vec_dot(cross, v03)) / 6.0

        if volume <= 1e-20:
            aspect_ratios.append(1e6)  # degenerate
            continue

        # Compute inscribed sphere radius (inradius)
        # For tetrahedron: r = 3V / A (total surface area)
        # Surface area = sum of 4 triangle areas
        def tri_area(a: tuple[float, float, float], b: tuple[float, float, float], c: tuple[float, float, float]) -> float:
            _, area = tri_normal_area(a, b, c)
            return area

        total_area = tri_area(p0, p1, p2) + tri_area(p0, p1, p3) + tri_area(p0, p2, p3) + tri_area(p1, p2, p3)
        if total_area <= 1e-20:
            aspect_ratios.append(1e6)
            continue

        inradius = 3.0 * volume / total_area

        # Aspect ratio: max_edge / (inradius * 2 * sqrt(6)) for regular tet normalization
        # For a regular tetrahedron: aspect_ratio ≈ 1.0
        ar = max_edge / (inradius * 2.0 * math.sqrt(6.0)) if inradius > 1e-15 else 1e6
        aspect_ratios.append(ar)

    if not aspect_ratios:
        return {"min_aspect_ratio": 0.0, "mean_aspect_ratio": 0.0, "max_aspect_ratio": 0.0, "quality_ok_ratio": 0.0}

    # Quality threshold: aspect ratio < 3.0 is considered "good"
    quality_ok = sum(1 for ar in aspect_ratios if ar < 3.0)

    return {
        "min_aspect_ratio": float(min(aspect_ratios)),
        "mean_aspect_ratio": float(sum(aspect_ratios) / len(aspect_ratios)),
        "max_aspect_ratio": float(max(aspect_ratios)),
        "quality_ok_ratio": float(quality_ok / len(aspect_ratios)),
    }


# Mesh guardrail constants
MAX_NODES = 500_000
MAX_TETS = 2_500_000
MIN_QUALITY_OK_RATIO = 0.8  # At least 80% of tets should have aspect ratio < 3.0


def run_meshing_case(
    *,
    case_id: str,
    geometry_dir: Path,
    mesh_dir: Path,
    y_tol: float = 1e-4,
    nz_min: float = 0.2,
    mesh_size_factor: float = 0.1,
    max_nodes: int = MAX_NODES,
    max_tets: int = MAX_TETS,
    min_quality_ratio: float = MIN_QUALITY_OK_RATIO,
) -> tuple[bool, dict, str]:
    """
    Creates:
    - wing.msh
    - boundary_sets.json
    - mesh_report.json
    - surf_sets.glb (debug)
    """
    start = time.perf_counter()

    case_geom_dir = geometry_dir / case_id
    stl_path = case_geom_dir / "wing.stl"
    if not stl_path.exists():
        raise FileNotFoundError(f"wing.stl not found: {stl_path}")

    case_mesh_dir = mesh_dir / case_id
    artifacts = MeshArtifacts(
        wing_msh=case_mesh_dir / "wing.msh",
        mesh_report_json=case_mesh_dir / "mesh_report.json",
        boundary_sets_json=case_mesh_dir / "boundary_sets.json",
        surf_sets_glb=case_mesh_dir / "surf_sets.glb",
    )

    ok, msg, stdout_tail, stderr_tail = run_gmsh_tetra(
        stl_path=stl_path,
        out_msh_path=artifacts.wing_msh,
        mesh_size_factor=mesh_size_factor,
    )
    if not ok:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        report = {
            "status": "failed",
            "failure_reason": msg,
            "elapsed_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "artifacts": [str(p) for p in artifacts.__dict__.values() if Path(p).exists()],
        }
        _write_json(artifacts.mesh_report_json, report)
        return False, report, msg

    try:
        nodes, triangles, tets = parse_msh2(artifacts.wing_msh)
    except ValueError as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        report = {
            "status": "failed",
            "failure_reason": f"mesh parse failed: {e}",
            "elapsed_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "artifacts": [str(p) for p in artifacts.__dict__.values() if Path(p).exists()],
        }
        _write_json(artifacts.mesh_report_json, report)
        return False, report, str(e)

    # Mesh guardrails: check node/element counts
    if len(nodes) > max_nodes:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        report = {
            "status": "failed",
            "failure_reason": f"mesh too large: {len(nodes)} nodes > {max_nodes} limit",
            "elapsed_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "stats": {"nodes": len(nodes), "tets": len(tets), "tris": len(triangles)},
            "artifacts": [str(p) for p in artifacts.__dict__.values() if Path(p).exists()],
        }
        _write_json(artifacts.mesh_report_json, report)
        return False, report, report["failure_reason"]

    if len(tets) > max_tets:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        report = {
            "status": "failed",
            "failure_reason": f"mesh too large: {len(tets)} tets > {max_tets} limit",
            "elapsed_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "stats": {"nodes": len(nodes), "tets": len(tets), "tris": len(triangles)},
            "artifacts": [str(p) for p in artifacts.__dict__.values() if Path(p).exists()],
        }
        _write_json(artifacts.mesh_report_json, report)
        return False, report, report["failure_reason"]

    # Compute mesh quality metrics
    quality_metrics = compute_tet_quality(nodes, tets)

    # Check quality guardrail (warning only, don't fail)
    quality_warning = ""
    if quality_metrics["quality_ok_ratio"] < min_quality_ratio:
        quality_warning = (
            f"mesh quality warning: only {quality_metrics['quality_ok_ratio']*100:.1f}% of tets have "
            f"aspect ratio < 3.0 (threshold: {min_quality_ratio*100:.0f}%)"
        )

    try:
        boundary_sets, note = build_boundary_sets(nodes=nodes, triangles=triangles, y_tol=y_tol, nz_min=nz_min)
        _write_json(artifacts.boundary_sets_json, boundary_sets)
        build_surf_sets_glb(nodes=nodes, triangles=triangles, boundary_sets=boundary_sets, out_glb=artifacts.surf_sets_glb)
    except ValueError as e:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        report = {
            "status": "failed",
            "failure_reason": f"boundary sets failed: {e}",
            "elapsed_ms": elapsed_ms,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "stats": {"nodes": len(nodes), "tets": len(tets), "tris": len(triangles)},
            "quality": quality_metrics,
            "artifacts": [str(p) for p in artifacts.__dict__.values() if Path(p).exists()],
        }
        _write_json(artifacts.mesh_report_json, report)
        return False, report, str(e)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    notes_combined = "\n".join(filter(None, [note, quality_warning])).strip()
    report = {
        "status": "success",
        "failure_reason": None,
        "elapsed_ms": elapsed_ms,
        "stdout_tail": (stdout_tail + ("\n" + notes_combined if notes_combined else "")).strip() or None,
        "stderr_tail": stderr_tail.strip() or None,
        "stats": {
            "nodes": len(nodes),
            "tets": len(tets),
            "tris": len(triangles),
            "nroot_count": boundary_sets["stats"]["nroot_count"],
            "surf_upper_ratio": boundary_sets["stats"]["surf_upper_ratio"],
        },
        "quality": quality_metrics,
        "guardrails": {
            "max_nodes": max_nodes,
            "max_tets": max_tets,
            "min_quality_ratio": min_quality_ratio,
            "quality_warning": quality_warning or None,
        },
        "artifacts": [str(p) for p in artifacts.__dict__.values() if Path(p).exists()],
    }
    _write_json(artifacts.mesh_report_json, report)
    return True, report, msg


