from __future__ import annotations

from pathlib import Path

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        return v
    return v / n


def _rotation_matrix_from_z(to_dir: np.ndarray) -> np.ndarray:
    """
    Return 3x3 rotation matrix that maps +Z axis to `to_dir` (unit).
    """
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    d = _unit(to_dir.astype(np.float64))
    c = float(np.clip(np.dot(z, d), -1.0, 1.0))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        # 180deg: rotate around X
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    axis = np.cross(z, d)
    axis = _unit(axis)
    s = float(np.linalg.norm(np.cross(z, d)))
    # Rodrigues
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)
    return R


def _make_arrow_trimesh(*, origin: np.ndarray, direction: np.ndarray, length: float, radius: float):
    """
    Build a simple arrow mesh (cylinder + cone), oriented along `direction`.
    """
    import trimesh  # type: ignore

    d = _unit(direction)
    if np.linalg.norm(d) <= 0:
        d = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    shaft_len = max(1e-6, length * 0.75)
    head_len = max(1e-6, length * 0.25)
    head_radius = radius * 2.2

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_len, sections=12)
    head = trimesh.creation.cone(radius=head_radius, height=head_len, sections=12)

    # Local arrow: +Z oriented, origin at base of shaft
    # cylinder is centered at origin → translate +Z by shaft_len/2
    shaft.apply_translation([0.0, 0.0, shaft_len / 2.0])
    # cone is centered at origin → translate to sit on top of shaft
    head.apply_translation([0.0, 0.0, shaft_len + head_len / 2.0])

    arrow = trimesh.util.concatenate([shaft, head])

    R = _rotation_matrix_from_z(d)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = origin.astype(np.float64)
    arrow.apply_transform(T)

    # Red arrows
    arrow.visual.vertex_colors = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (arrow.vertices.shape[0], 1))
    return arrow


def _viridis_rgb(x: np.ndarray) -> np.ndarray:
    """
    Tiny viridis-like colormap (piecewise linear stops), returns uint8 RGB.
    """
    # fmt: off
    stops = np.array(
        [
            [0.0000, 68, 1, 84],
            [0.2500, 59, 82, 139],
            [0.5000, 33, 145, 140],
            [0.7500, 94, 201, 98],
            [1.0000, 253, 231, 37],
        ],
        dtype=np.float64,
    )
    # fmt: on
    t = np.clip(x.astype(np.float64), 0.0, 1.0)
    out = np.zeros((t.shape[0], 3), dtype=np.float64)

    for i in range(len(stops) - 1):
        t0, r0, g0, b0 = stops[i]
        t1, r1, g1, b1 = stops[i + 1]
        mask = (t >= t0) & (t <= t1)
        if not np.any(mask):
            continue
        alpha = (t[mask] - t0) / max(1e-12, (t1 - t0))
        out[mask, 0] = r0 + (r1 - r0) * alpha
        out[mask, 1] = g0 + (g1 - g0) * alpha
        out[mask, 2] = b0 + (b1 - b0) * alpha

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def surface_results_to_glb(
    *,
    glb_path: Path,
    node_id: np.ndarray,
    pos: np.ndarray,
    faces: list[list[int]],
    stress_vm: np.ndarray,
    clim: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """
    Create a GLB from surface mesh (pos + faces) with per-vertex colors mapped from stress_vm.
    faces are in original node_id space, so we remap them to 0..N-1 indices.

    Args:
        clim: Optional (vmin, vmax) tuple. If None, auto-calculated from data (using mask if provided).
        mask: Optional boolean array (True=valid, False=ignored). Used for auto-clim calculation only.
    """
    glb_path.parent.mkdir(parents=True, exist_ok=True)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be [N,3]")
    if stress_vm.ndim != 1 or stress_vm.shape[0] != pos.shape[0]:
        raise ValueError("stress_vm must be [N]")

    node_id_int = node_id.astype(np.int64)
    lut = {int(node_id_int[i]): int(i) for i in range(node_id_int.shape[0])}

    tri: list[list[int]] = []
    for f in faces:
        if not f or len(f) != 3:
            continue
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a not in lut or b not in lut or c not in lut:
            continue
        tri.append([lut[a], lut[b], lut[c]])

    if not tri:
        raise ValueError("no faces after remap")

    # Normalize colors
    if clim is not None:
        vmin, vmax = clim
    else:
        # Auto-range: use mask to exclude singularities if provided
        # mask is True for VALID nodes, False for IGNORED nodes.
        valid_stress = stress_vm[mask] if mask is not None else stress_vm
        if valid_stress.size > 0:
            vmin = float(np.min(valid_stress))
            # Use 98th percentile instead of max to avoid outliers compressing the dynamic range
            vmax = float(np.percentile(valid_stress, 98))
        else:
            vmin = float(np.min(stress_vm)) if stress_vm.size > 0 else 0.0
            vmax = float(np.percentile(stress_vm, 98)) if stress_vm.size > 0 else 1.0

    denom = max(1e-12, (vmax - vmin))
    # Clip values to range before normalizing
    t = (np.clip(stress_vm, vmin, vmax) - vmin) / denom
    rgb = _viridis_rgb(t)
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)

    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("trimesh is required to export GLB. Install deps: `pip install -r requirements.txt`.") from e

    mesh = trimesh.Trimesh(vertices=pos.astype(np.float64), faces=np.asarray(tri, dtype=np.int64), process=False)
    mesh.visual.vertex_colors = rgba
    scene = trimesh.Scene(mesh)
    glb_path.write_bytes(scene.export(file_type="glb"))


def surface_results_to_glb_with_extras(
    *,
    glb_path: Path,
    node_id: np.ndarray,
    pos: np.ndarray,
    faces: list[list[int]],
    stress_vm: np.ndarray,
    extras: list,
    clim: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> None:
    """
    Same as surface_results_to_glb but adds extra trimesh meshes into the scene (e.g., pressure arrows).
    """
    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError("trimesh is required to export GLB. Install deps: `pip install -r requirements.txt`.") from e

    # Build the base wing mesh using the existing function logic, but keep the mesh object
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("pos must be [N,3]")
    if stress_vm.ndim != 1 or stress_vm.shape[0] != pos.shape[0]:
        raise ValueError("stress_vm must be [N]")

    node_id_int = node_id.astype(np.int64)
    lut = {int(node_id_int[i]): int(i) for i in range(node_id_int.shape[0])}

    tri: list[list[int]] = []
    for f in faces:
        if not f or len(f) != 3:
            continue
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a not in lut or b not in lut or c not in lut:
            continue
        tri.append([lut[a], lut[b], lut[c]])
    if not tri:
        raise ValueError("no faces after remap")

    if clim is not None:
        vmin, vmax = clim
    else:
        valid_stress = stress_vm[mask] if mask is not None else stress_vm
        if valid_stress.size > 0:
            vmin = float(np.min(valid_stress))
            vmax = float(np.percentile(valid_stress, 98))
        else:
            vmin = float(np.min(stress_vm)) if stress_vm.size > 0 else 0.0
            vmax = float(np.percentile(stress_vm, 98)) if stress_vm.size > 0 else 1.0

    denom = max(1e-12, (vmax - vmin))
    t = (np.clip(stress_vm, vmin, vmax) - vmin) / denom
    rgb = _viridis_rgb(t)
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1)

    wing_mesh = trimesh.Trimesh(vertices=pos.astype(np.float64), faces=np.asarray(tri, dtype=np.int64), process=False)
    wing_mesh.visual.vertex_colors = rgba

    scene = trimesh.Scene(wing_mesh)
    for m in extras:
        scene.add_geometry(m)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(scene.export(file_type="glb"))


def make_pressure_arrows_meshes(
    *,
    nodes: dict[int, tuple[float, float, float]],
    surf_upper_faces: list[list[int]],
    sample_n: int = 200,
    seed: int = 0,
) -> list:
    """
    Create arrow meshes sampled from SURF_UPPER faces.
    """
    import random

    if not surf_upper_faces:
        return []

    # Scale arrows by span
    ys = [p[1] for p in nodes.values()]
    span = float(max(ys) - min(ys)) if ys else 1.0
    arrow_len = max(0.01, 0.05 * span)
    radius = arrow_len * 0.04

    rnd = random.Random(int(seed) & 0xFFFFFFFF)
    faces = list(surf_upper_faces)
    if len(faces) > sample_n:
        faces = rnd.sample(faces, sample_n)

    # Outward test centroid
    c_vol = np.mean(np.asarray(list(nodes.values()), dtype=np.float64), axis=0)

    arrows = []
    for f in faces:
        if not f or len(f) != 3:
            continue
        n0, n1, n2 = int(f[0]), int(f[1]), int(f[2])
        if n0 not in nodes or n1 not in nodes or n2 not in nodes:
            continue
        p0 = np.asarray(nodes[n0], dtype=np.float64)
        p1 = np.asarray(nodes[n1], dtype=np.float64)
        p2 = np.asarray(nodes[n2], dtype=np.float64)
        cf = (p0 + p1 + p2) / 3.0

        # face normal (unnormalized)
        e1 = p1 - p0
        e2 = p2 - p0
        n = np.cross(e1, e2)
        n_norm = float(np.linalg.norm(n))
        if n_norm <= 0:
            continue
        n_hat = n / n_norm
        # outward correction
        if float(np.dot(n_hat, cf - c_vol)) < 0:
            n_hat = -n_hat

        # pressure direction is inward
        d = -n_hat
        arrows.append(_make_arrow_trimesh(origin=cf, direction=d, length=arrow_len, radius=radius))

    return arrows


