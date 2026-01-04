from __future__ import annotations

import time
from pathlib import Path


def _is_binary_glb(path: Path) -> bool:
    """
    Binary GLB files start with ASCII 'glTF' magic.
    If it starts with '{', it's likely JSON .gltf content misnamed as .glb.
    """
    if not path.exists():
        return False
    head = path.read_bytes()[:4]
    return head == b"glTF"


def stl_to_glb_with_pyvista(*, stl_path: Path, glb_path: Path) -> None:
    import pyvista as pv

    mesh = pv.read(str(stl_path))
    if mesh is None or mesh.n_points <= 0:
        raise ValueError("pyvista read returned empty mesh")

    # PyVista export_gltf may write JSON glTF even if extension is .glb.
    # We export to .gltf explicitly, then convert to a real binary .glb via trimesh.
    tmp_gltf = glb_path.with_suffix(".gltf")
    if tmp_gltf.exists():
        tmp_gltf.unlink()

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="lightgray")
    plotter.export_gltf(str(tmp_gltf))

    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "trimesh is required to convert PyVista-exported glTF to binary GLB. "
            "Install deps: `pip install -r requirements.txt`."
        ) from e

    scene = trimesh.load(tmp_gltf, force="scene")
    glb_bytes = scene.export(file_type="glb")
    glb_path.write_bytes(glb_bytes)

    try:
        tmp_gltf.unlink()
    except Exception:
        pass

    if not _is_binary_glb(glb_path):
        raise ValueError("PyVista→trimesh conversion did not produce a valid binary GLB")


def stl_to_glb_with_trimesh(*, stl_path: Path, glb_path: Path) -> None:
    try:
        import trimesh  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "trimesh is required for the GLB fallback exporter. Install deps: `pip install -r requirements.txt`."
        ) from e

    mesh = trimesh.load_mesh(stl_path, process=False)
    if mesh is None or mesh.is_empty:
        raise ValueError("trimesh read returned empty mesh")

    scene = trimesh.Scene(mesh)
    glb_bytes = scene.export(file_type="glb")
    glb_path.write_bytes(glb_bytes)


def stl_to_glb(*, stl_path: Path, glb_path: Path) -> tuple[bool, str, int]:
    """
    Returns (success, message, elapsed_ms).
    """
    start = time.perf_counter()

    if not stl_path.exists():
        return False, f"STL not found: {stl_path}", 0

    glb_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stl_to_glb_with_pyvista(stl_path=stl_path, glb_path=glb_path)
        if not _is_binary_glb(glb_path):
            raise ValueError("GLB output is not binary GLB")
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return True, "GLB exported via PyVista.", elapsed_ms
    except Exception as pyvista_error:  # noqa: BLE001
        try:
            stl_to_glb_with_trimesh(stl_path=stl_path, glb_path=glb_path)
            if not _is_binary_glb(glb_path):
                raise ValueError("GLB output is not binary GLB")
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return True, f"PyVista failed → GLB exported via trimesh. ({pyvista_error})", elapsed_ms
        except Exception as trimesh_error:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return False, f"GLB export failed. pyvista={pyvista_error} trimesh={trimesh_error}", elapsed_ms


