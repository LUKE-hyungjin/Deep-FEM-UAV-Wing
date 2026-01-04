"""
Blender bpy script: generate a simple parametric wing volume and export as STL.

Coordinate system (spec):
- +Y: span direction (root -> tip)
- +X: chord direction (LE -> TE)
- +Z: thickness direction (lower -> upper)
- Root section lies on y=0 plane

Usage (headless):
  blender --background --python blender/generate_wing.py -- \
    --span_m 1.2 --chord_m 0.35 --sweep_deg 10 --thickness_ratio 0.1 \
    --out_stl /abs/path/wing.stl
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy


def _parse_args(argv: list[str]) -> dict[str, str]:
    if "--" not in argv:
        raise ValueError("Missing '--' separator for script args.")

    args = argv[argv.index("--") + 1 :]
    result: dict[str, str] = {}
    i = 0
    while i < len(args):
        key = args[i]
        if not key.startswith("--"):
            raise ValueError(f"Invalid arg: {key}")
        if i + 1 >= len(args):
            raise ValueError(f"Missing value for {key}")
        result[key[2:]] = args[i + 1]
        i += 2
    return result


def _naca_00xx_thickness_z(*, x: float, chord: float, thickness_ratio: float) -> float:
    # x is normalized 0..1
    t = thickness_ratio
    # Classic NACA thickness distribution (00xx), scaled by chord
    yt = 5.0 * t * (
        0.2969 * math.sqrt(max(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    return yt * chord


def _build_airfoil_loop(*, chord: float, thickness_ratio: float, n: int = 60) -> list[tuple[float, float]]:
    # Returns list of (x, z) points (closed loop) for symmetric airfoil.
    # Lower surface: TE -> LE, Upper: LE -> TE
    xs = [i / (n - 1) for i in range(n)]
    lower = [(x * chord, -_naca_00xx_thickness_z(x=x, chord=chord, thickness_ratio=thickness_ratio)) for x in xs]
    upper = [(x * chord, +_naca_00xx_thickness_z(x=x, chord=chord, thickness_ratio=thickness_ratio)) for x in reversed(xs)]
    loop = lower + upper[1:-1]  # avoid duplicating endpoints
    return loop


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)


def _set_units() -> None:
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def generate_wing_mesh(*, span_m: float, chord_m: float, sweep_deg: float, thickness_ratio: float) -> bpy.types.Object:
    loop_xz = _build_airfoil_loop(chord=chord_m, thickness_ratio=thickness_ratio, n=70)
    m = len(loop_xz)

    dx_tip = math.tan(math.radians(sweep_deg)) * span_m if sweep_deg else 0.0

    # Create vertices: root loop (y=0) + tip loop (y=span)
    verts: list[tuple[float, float, float]] = []
    for (x, z) in loop_xz:
        verts.append((x, 0.0, z))
    for (x, z) in loop_xz:
        verts.append((x + dx_tip, span_m, z))

    faces: list[tuple[int, int, int, int]] = []

    # Side faces (quads) connecting root to tip
    for i in range(m):
        i0 = i
        i1 = (i + 1) % m
        j0 = i + m
        j1 = ((i + 1) % m) + m
        faces.append((i0, i1, j1, j0))

    # Caps: add explicit root/tip faces (as triangles) later via bmesh for correctness.

    mesh = bpy.data.meshes.new("wing_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new("wing", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Clean + normals
    # Cap open boundaries (root/tip) using bmesh, then triangulate
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        # Fill all boundary loops (root & tip openings)
        bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)

    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-9)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()

    return obj


def export_stl(*, obj: bpy.types.Object, out_stl: Path) -> None:
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Ensure STL exporter is available (addon/operator name differs by Blender version)
    try:
        bpy.ops.preferences.addon_enable(module="io_mesh_stl")
    except Exception:
        pass

    # 1) Blender <=4.x typically provides export_mesh.stl
    try:
        bpy.ops.export_mesh.stl(
            filepath=str(out_stl),
            use_selection=True,
            global_scale=1.0,
            ascii=False,
        )
        return
    except Exception:
        pass

    # 2) Blender 5.x: new operator name (try common signature)
    try:
        bpy.ops.wm.stl_export(
            filepath=str(out_stl),
            export_selected_objects=True,
            global_scale=1.0,
            ascii_format=False,
        )
        return
    except Exception:
        pass

    # 3) Last resort: try minimal call (signature varies)
    bpy.ops.wm.stl_export(filepath=str(out_stl))


def main() -> None:
    args = _parse_args(sys.argv)
    span_m = float(args["span_m"])
    chord_m = float(args["chord_m"])
    sweep_deg = float(args["sweep_deg"])
    thickness_ratio = float(args["thickness_ratio"])
    out_stl = Path(args["out_stl"]).resolve()

    _set_units()
    _clear_scene()

    obj = generate_wing_mesh(
        span_m=span_m,
        chord_m=chord_m,
        sweep_deg=sweep_deg,
        thickness_ratio=thickness_ratio,
    )
    export_stl(obj=obj, out_stl=out_stl)
    print(f"OK: exported STL → {out_stl}")


if __name__ == "__main__":
    main()


