#!/usr/bin/env python3
"""Run GNN inference and generate prediction/error GLBs.

Usage:
    # Single case
    python scripts/inference_gnn.py --case-id 001

    # All cases
    python scripts/inference_gnn.py --all

The script will:
1. Load trained model from checkpoints/
2. Run inference on specified case(s)
3. Generate wing_pred.glb (AI stress) and wing_error.glb (|FEM-AI|)
4. Report metrics (MAE/RMSE/Max for all_nodes and masked_nodes)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import numpy as np
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch is required.")
    sys.exit(1)

from deep_fem_uav_wing.gnn.dataset import build_graph_data
from deep_fem_uav_wing.gnn.model import GraphSAGEModel, compute_metrics


def load_model(checkpoint_path: Path, device: torch.device) -> GraphSAGEModel:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config
    config = checkpoint.get("model_config", {
        "in_channels": 10,
        "hidden_channels": 128,
        "out_channels": 1,
        "num_layers": 4,
        "dropout": 0.1,
    })

    model = GraphSAGEModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def generate_prediction_glb(
    case_id: str,
    pred_stress: np.ndarray,
    surface_npz_path: Path,
    boundary_sets_path: Path,
    out_glb_path: Path,
    *,
    deform_scale: float = 10.0,
):
    """Generate wing_pred.glb with AI-predicted stress colors.

    Uses same visualization approach as FEM results.
    """
    try:
        import trimesh
        from matplotlib import cm
    except ImportError:
        print("ERROR: trimesh and matplotlib required for GLB generation")
        return False

    # Load surface data
    npz = np.load(surface_npz_path)
    pos = npz["pos"]
    disp = npz["disp"]
    boundary_sets = json.loads(boundary_sets_path.read_text(encoding="utf-8"))

    # Apply deformation
    deformed_pos = pos + disp * deform_scale

    # Build surface mesh from faces
    node_ids = npz["node_id"]
    node_id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    faces = []
    for face in boundary_sets["surf_all_faces"]:
        idxs = []
        for nid in face:
            if nid in node_id_to_idx:
                idxs.append(node_id_to_idx[nid])
        if len(idxs) == 3:
            faces.append(idxs)

    if not faces:
        print(f"  WARNING: No valid faces for {case_id}")
        return False

    faces = np.array(faces, dtype=np.int32)

    # Color mapping (same as FEM viz)
    stress_min = pred_stress.min()
    stress_max = pred_stress.max()
    if stress_max - stress_min < 1e-10:
        normalized = np.zeros_like(pred_stress)
    else:
        normalized = (pred_stress - stress_min) / (stress_max - stress_min)

    # Use jet colormap
    cmap = cm.get_cmap("jet")
    colors_float = cmap(normalized)[:, :3]  # RGB only
    colors_uint8 = (colors_float * 255).astype(np.uint8)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=deformed_pos, faces=faces, process=False)
    mesh.visual.vertex_colors = np.hstack([colors_uint8, np.full((len(colors_uint8), 1), 255, dtype=np.uint8)])

    # Export
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb_path), file_type="glb")

    return True


def generate_error_glb(
    case_id: str,
    error: np.ndarray,
    surface_npz_path: Path,
    boundary_sets_path: Path,
    out_glb_path: Path,
    *,
    deform_scale: float = 10.0,
):
    """Generate wing_error.glb with |FEM - AI| error colors."""
    try:
        import trimesh
        from matplotlib import cm
    except ImportError:
        print("ERROR: trimesh and matplotlib required for GLB generation")
        return False

    # Load surface data
    npz = np.load(surface_npz_path)
    pos = npz["pos"]
    disp = npz["disp"]
    boundary_sets = json.loads(boundary_sets_path.read_text(encoding="utf-8"))

    # Apply deformation
    deformed_pos = pos + disp * deform_scale

    # Build surface mesh from faces
    node_ids = npz["node_id"]
    node_id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    faces = []
    for face in boundary_sets["surf_all_faces"]:
        idxs = []
        for nid in face:
            if nid in node_id_to_idx:
                idxs.append(node_id_to_idx[nid])
        if len(idxs) == 3:
            faces.append(idxs)

    if not faces:
        print(f"  WARNING: No valid faces for {case_id}")
        return False

    faces = np.array(faces, dtype=np.int32)

    # Color mapping for error (use "hot" colormap - low error = dark, high = bright)
    error_min = 0.0
    error_max = error.max()
    if error_max < 1e-10:
        normalized = np.zeros_like(error)
    else:
        normalized = error / error_max

    cmap = cm.get_cmap("hot")
    colors_float = cmap(normalized)[:, :3]
    colors_uint8 = (colors_float * 255).astype(np.uint8)

    # Create mesh
    mesh = trimesh.Trimesh(vertices=deformed_pos, faces=faces, process=False)
    mesh.visual.vertex_colors = np.hstack([colors_uint8, np.full((len(colors_uint8), 1), 255, dtype=np.uint8)])

    # Export
    out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb_path), file_type="glb")

    return True


def run_inference(
    model: torch.nn.Module,
    case_id: str,
    device: torch.device,
    paths: dict[str, Path],
    *,
    log_scale_stress: bool = True,
    deform_scale: float = 10.0,
) -> dict:
    """Run inference on a single case.

    Returns:
        Dictionary with metrics and output paths
    """
    fem_dir = paths["fem_dir"]
    mesh_dir = paths["mesh_dir"]
    geometry_dir = paths["geometry_dir"]

    surface_npz_path = fem_dir / case_id / "surface_results.npz"
    boundary_sets_path = mesh_dir / case_id / "boundary_sets.json"
    params_path = geometry_dir / case_id / "params.json"

    # Check files exist
    if not surface_npz_path.exists():
        return {"status": "failed", "reason": "surface_results.npz not found"}
    if not boundary_sets_path.exists():
        return {"status": "failed", "reason": "boundary_sets.json not found"}
    if not params_path.exists():
        return {"status": "failed", "reason": "params.json not found"}

    # Build graph data
    graph_data = build_graph_data(
        surface_npz_path=surface_npz_path,
        boundary_sets_path=boundary_sets_path,
        params_path=params_path,
        log_scale_stress=log_scale_stress,
        normalize_pos=True,
    )

    # Convert to tensors
    x = torch.from_numpy(graph_data["x"]).to(device)
    edge_index = torch.from_numpy(graph_data["edge_index"]).to(device)
    y = torch.from_numpy(graph_data["y"]).to(device)
    loss_mask = torch.from_numpy(graph_data["loss_mask"]).to(device)

    # Run inference
    with torch.no_grad():
        pred_log = model(x, edge_index)

    # Compute metrics
    metrics = compute_metrics(pred_log, y, loss_mask, log_scale=log_scale_stress)

    # Convert predictions to original scale
    pred_log_np = pred_log.cpu().numpy().flatten()
    if log_scale_stress:
        pred_stress = np.expm1(pred_log_np)  # inverse of log1p
    else:
        pred_stress = pred_log_np

    # Get ground truth stress
    gt_stress = graph_data["stress_vm_raw"]

    # Compute error
    error = np.abs(gt_stress - pred_stress)

    # Generate GLBs
    pred_glb_path = fem_dir / case_id / "wing_pred.glb"
    error_glb_path = fem_dir / case_id / "wing_error.glb"

    pred_ok = generate_prediction_glb(
        case_id=case_id,
        pred_stress=pred_stress,
        surface_npz_path=surface_npz_path,
        boundary_sets_path=boundary_sets_path,
        out_glb_path=pred_glb_path,
        deform_scale=deform_scale,
    )

    error_ok = generate_error_glb(
        case_id=case_id,
        error=error,
        surface_npz_path=surface_npz_path,
        boundary_sets_path=boundary_sets_path,
        out_glb_path=error_glb_path,
        deform_scale=deform_scale,
    )

    # Save inference report
    report = {
        "case_id": case_id,
        "status": "success" if (pred_ok and error_ok) else "partial",
        "metrics": metrics,
        "pred_glb": str(pred_glb_path) if pred_ok else None,
        "error_glb": str(error_glb_path) if error_ok else None,
        "pred_stress_range": [float(pred_stress.min()), float(pred_stress.max())],
        "gt_stress_range": [float(gt_stress.min()), float(gt_stress.max())],
        "error_range": [float(error.min()), float(error.max())],
    }

    report_path = fem_dir / case_id / "inference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run GNN inference for Wing Stress Prediction")
    parser.add_argument("--case-id", type=str, help="Single case ID to process")
    parser.add_argument("--all", action="store_true", help="Process all cases")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--deform-scale", type=float, default=10.0, help="Deformation scale for visualization")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    args = parser.parse_args()

    if not args.case_id and not args.all:
        parser.error("Either --case-id or --all is required")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Inference] Using device: {device}")

    # Paths
    paths = {
        "fem_dir": PROJECT_ROOT / "data" / "raw" / "fem",
        "mesh_dir": PROJECT_ROOT / "data" / "raw" / "mesh",
        "geometry_dir": PROJECT_ROOT / "data" / "raw" / "geometry",
    }

    # Load model
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Train a model first with: python scripts/train_gnn.py")
        sys.exit(1)

    print(f"[Inference] Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)

    # Get case IDs to process
    if args.all:
        case_ids = sorted([
            d.name for d in paths["fem_dir"].iterdir()
            if d.is_dir() and (d / "surface_results.npz").exists()
        ])
    else:
        case_ids = [args.case_id]

    print(f"[Inference] Processing {len(case_ids)} case(s)...")

    # Run inference
    results = []
    for i, case_id in enumerate(case_ids, 1):
        print(f"[{i}/{len(case_ids)}] Processing {case_id}...", end=" ")

        result = run_inference(
            model=model,
            case_id=case_id,
            device=device,
            paths=paths,
            deform_scale=args.deform_scale,
        )

        if result["status"] == "success":
            mae_all = result["metrics"]["all_nodes"]["mae"]
            mae_masked = result["metrics"]["masked_nodes"]["mae"]
            print(f"OK - MAE(all/masked): {mae_all:.2e}/{mae_masked:.2e} Pa")
        else:
            print(f"FAILED - {result.get('reason', 'unknown')}")

        results.append(result)

    # Summary
    successful = [r for r in results if r["status"] in ("success", "partial")]
    print(f"\n[Inference] Completed: {len(successful)}/{len(results)} successful")

    if successful:
        avg_mae_all = sum(r["metrics"]["all_nodes"]["mae"] for r in successful) / len(successful)
        avg_mae_masked = sum(r["metrics"]["masked_nodes"]["mae"] for r in successful) / len(successful)
        max_error_all = max(r["metrics"]["all_nodes"]["max_error"] for r in successful)
        max_error_masked = max(r["metrics"]["masked_nodes"]["max_error"] for r in successful)

        print(f"  Avg MAE (all nodes): {avg_mae_all:.2e} Pa")
        print(f"  Avg MAE (masked): {avg_mae_masked:.2e} Pa")
        print(f"  Max Error (all nodes): {max_error_all:.2e} Pa")
        print(f"  Max Error (masked): {max_error_masked:.2e} Pa")

    # Save summary
    summary_path = PROJECT_ROOT / "checkpoints" / "inference_summary.json"
    summary = {
        "n_processed": len(results),
        "n_successful": len(successful),
        "results": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[Inference] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
