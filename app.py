"""Deep-FEM-UAV-Wing: Gradio Demo Application.

Stage 6: FEM vs AI Comparison Dashboard
- Side-by-Side 3D View (FEM vs AI prediction)
- Error Map Toggle (|FEM - AI|)
- Metrics Display (MAE/RMSE/Max for all_nodes and masked_nodes)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "geometry"
MESH_DIR = PROJECT_ROOT / "data" / "raw" / "mesh"
FEM_DIR = PROJECT_ROOT / "data" / "raw" / "fem"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARAMS_CSV = GEOMETRY_DIR / "params.csv"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Material constants (AL6061-T6)
YIELD_STRENGTH_PA = 276e6  # 276 MPa
YIELD_STRENGTH_MPA = 276.0


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------


def _load_fem_report(case_id: str) -> dict | None:
    """Load FEM report for a specific case."""
    report_path = FEM_DIR / case_id / "fem_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


def _load_inference_report(case_id: str) -> dict | None:
    """Load AI inference report for a specific case."""
    report_path = FEM_DIR / case_id / "inference_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


def _load_geometry_params(case_id: str) -> dict | None:
    """Load geometry parameters for a specific case."""
    params_path = GEOMETRY_DIR / case_id / "params.json"
    if params_path.exists():
        return json.loads(params_path.read_text(encoding="utf-8"))
    return None


def _load_training_log() -> dict | None:
    """Load GNN training log."""
    log_path = CHECKPOINTS_DIR / "training_log.json"
    if log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8"))
    return None


def _load_surface_results(case_id: str) -> dict | None:
    """Load surface results NPZ for stress histogram."""
    npz_path = FEM_DIR / case_id / "surface_results.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        return {
            "stress_vm": data.get("stress_vm"),
            "disp": data.get("disp"),
            "pos": data.get("pos"),
            "loss_mask": data.get("loss_mask"),
        }
    return None


def _list_success_cases() -> list[dict[str, str]]:
    if not PARAMS_CSV.exists():
        return []

    with PARAMS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    success_rows = [r for r in rows if (r.get("status") or "").strip() == "success"]
    return list(reversed(success_rows))


def _case_label(row: dict[str, str]) -> str:
    case_id = (row.get("case_id") or "").strip()
    span_m = (row.get("span_m") or "").strip()
    chord_m = (row.get("chord_m") or "").strip()
    sweep_deg = (row.get("sweep_deg") or "").strip()
    thickness_ratio = (row.get("thickness_ratio") or "").strip()
    return f"{case_id} | span={span_m} chord={chord_m} sweep={sweep_deg} t/c={thickness_ratio}"


# ---------------------------------------------------------------------------
# Report Generation Functions
# ---------------------------------------------------------------------------


def get_model_summary() -> str:
    """Generate GNN model training summary."""
    training_log = _load_training_log()
    if not training_log:
        return "**Model not trained yet.** Run `python scripts/train_gnn.py`"

    test_metrics = training_log.get("test_metrics", {})
    all_nodes = test_metrics.get("all_nodes", {})
    masked_nodes = test_metrics.get("masked_nodes", {})

    lines = [
        "## GNN Model Performance",
        "",
        f"**Best Epoch:** {training_log.get('best_epoch', 'N/A')}",
        f"**Training Time:** {training_log.get('total_time_s', 0):.1f}s",
        "",
        "### Test Set Metrics",
        "",
        "| Metric | All Nodes | Masked (excl. root) |",
        "|--------|-----------|---------------------|",
        f"| MAE | {all_nodes.get('mae', 0)/1e6:.2f} MPa | {masked_nodes.get('mae', 0)/1e6:.2f} MPa |",
        f"| RMSE | {all_nodes.get('rmse', 0)/1e6:.2f} MPa | {masked_nodes.get('rmse', 0)/1e6:.2f} MPa |",
        f"| Max Error | {all_nodes.get('max_error', 0)/1e6:.2f} MPa | {masked_nodes.get('max_error', 0)/1e6:.2f} MPa |",
    ]

    return "\n".join(lines)


def _load_mesh_report(case_id: str) -> dict | None:
    """Load mesh report for a specific case."""
    mesh_dir = PROJECT_ROOT / "data" / "raw" / "mesh"
    report_path = mesh_dir / case_id / "mesh_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


def get_case_comparison_report(case_id: str | None) -> str:
    """Generate comprehensive engineering report with FEM vs AI comparison."""
    if not case_id:
        return "Select a case to view engineering report."

    fem_report = _load_fem_report(case_id)
    inference_report = _load_inference_report(case_id)
    params = _load_geometry_params(case_id)
    mesh_report = _load_mesh_report(case_id)

    if not fem_report:
        return f"FEM report not found for case {case_id}."

    # Extract FEM data
    fem_params = fem_report.get("params", {})
    fem_stats = fem_report.get("stats", {})
    fem_metrics_all = fem_report.get("metrics_all_nodes", {})
    fem_metrics_masked = fem_report.get("metrics_masked_nodes", {})
    surface_mask = fem_report.get("surface_mask", {})

    # Calculate values
    fem_stress_max_all = fem_metrics_all.get("stress_max", 0)
    fem_stress_max_masked = fem_metrics_masked.get("stress_max", 0)
    sf_all = YIELD_STRENGTH_PA / fem_stress_max_all if fem_stress_max_all > 0 else float("inf")
    sf_masked = YIELD_STRENGTH_PA / fem_stress_max_masked if fem_stress_max_masked > 0 else float("inf")

    lines = [f"# Case {case_id} Engineering Report"]

    # === 1. Geometry Section ===
    lines.append("")
    lines.append("## 1. Geometry")
    if params:
        span = params.get("span_m", 0)
        chord = params.get("chord_m", 0)
        ar = span / chord if chord > 0 else 0
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Span | {span*1000:.1f} mm |")
        lines.append(f"| Chord | {chord*1000:.1f} mm |")
        lines.append(f"| Aspect Ratio | {ar:.2f} |")
        lines.append(f"| Sweep Angle | {params.get('sweep_deg', 0):.1f}° |")
        lines.append(f"| Thickness Ratio | {params.get('thickness_ratio', 0)*100:.1f}% |")

    # === 2. Material & Loading ===
    lines.append("")
    lines.append("## 2. Material & Loading")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Material | AL6061-T6 |")
    lines.append(f"| Young's Modulus | {fem_params.get('young_modulus_pa', 0)/1e9:.0f} GPa |")
    lines.append(f"| Poisson's Ratio | {fem_params.get('poisson_ratio', 0):.2f} |")
    lines.append(f"| Yield Strength | {YIELD_STRENGTH_MPA:.0f} MPa |")
    lines.append(f"| Applied Pressure | {fem_params.get('pressure_pa', 0)/1000:.1f} kPa |")
    lines.append("| Boundary Condition | Root Fixed (Cantilever) |")

    # === 3. Stress Results - FEM vs AI Comparison ===
    lines.append("")
    lines.append("## 3. Stress Analysis (von Mises)")
    lines.append("")

    if inference_report:
        ai_metrics = inference_report.get("metrics", {})
        ai_all = ai_metrics.get("all_nodes", {})
        ai_masked = ai_metrics.get("masked_nodes", {})
        pred_range = inference_report.get("pred_stress_range", [0, 0])

        lines.append("### FEM vs AI Comparison")
        lines.append("")
        lines.append("| Metric | FEM (Ground Truth) | AI (Predicted) | Error |")
        lines.append("|--------|-------------------|----------------|-------|")

        # Max stress comparison
        fem_max = fem_stress_max_masked / 1e6
        ai_max = pred_range[1] / 1e6 if pred_range else 0
        lines.append(f"| Max Stress (masked) | {fem_max:.2f} MPa | {ai_max:.2f} MPa | - |")

        # Mean stress
        fem_mean = fem_metrics_masked.get("stress_mean", 0) / 1e6
        lines.append(f"| Mean Stress (masked) | {fem_mean:.2f} MPa | - | - |")

        # P95/P99
        fem_p95 = fem_metrics_masked.get("stress_p95", 0) / 1e6
        fem_p99 = fem_metrics_masked.get("stress_p99", 0) / 1e6
        lines.append(f"| P95 Stress | {fem_p95:.2f} MPa | - | - |")
        lines.append(f"| P99 Stress | {fem_p99:.2f} MPa | - | - |")

        lines.append("")
        lines.append("### AI Prediction Accuracy")
        lines.append("")
        lines.append("| Metric | All Nodes | Masked (excl. root) |")
        lines.append("|--------|-----------|---------------------|")
        lines.append(f"| MAE | {ai_all.get('mae', 0)/1e6:.3f} MPa | {ai_masked.get('mae', 0)/1e6:.3f} MPa |")
        lines.append(f"| RMSE | {ai_all.get('rmse', 0)/1e6:.3f} MPa | {ai_masked.get('rmse', 0)/1e6:.3f} MPa |")
        lines.append(f"| Max Error | {ai_all.get('max_error', 0)/1e6:.2f} MPa | {ai_masked.get('max_error', 0)/1e6:.2f} MPa |")

        # Relative error percentage
        if fem_max > 0:
            rel_mae = (ai_masked.get('mae', 0) / 1e6) / fem_max * 100
            lines.append("")
            lines.append(f"**Relative MAE:** {rel_mae:.2f}% of max stress")
    else:
        lines.append("| Metric | All Nodes | Masked (excl. root) |")
        lines.append("|--------|-----------|---------------------|")
        lines.append(f"| Max | {fem_stress_max_all/1e6:.2f} MPa | {fem_stress_max_masked/1e6:.2f} MPa |")
        lines.append(f"| Mean | {fem_metrics_all.get('stress_mean', 0)/1e6:.2f} MPa | {fem_metrics_masked.get('stress_mean', 0)/1e6:.2f} MPa |")
        lines.append(f"| P95 | {fem_metrics_all.get('stress_p95', 0)/1e6:.2f} MPa | {fem_metrics_masked.get('stress_p95', 0)/1e6:.2f} MPa |")
        lines.append(f"| P99 | {fem_metrics_all.get('stress_p99', 0)/1e6:.2f} MPa | {fem_metrics_masked.get('stress_p99', 0)/1e6:.2f} MPa |")
        lines.append("")
        lines.append("_AI prediction not available._")

    # Root singularity note
    lines.append("")
    lines.append(f"**Root Singularity Band:** y < {surface_mask.get('root_band_y', 0)*1000:.1f} mm ({surface_mask.get('loss_mask_false', 0)} nodes excluded)")

    # === 4. Safety Factor ===
    lines.append("")
    lines.append("## 4. Safety Factor")
    lines.append("")

    sf_status_all = "SAFE" if sf_all >= 1.5 else ("MARGINAL" if sf_all >= 1.0 else "YIELD EXCEEDED")
    sf_status_masked = "SAFE" if sf_masked >= 1.5 else ("MARGINAL" if sf_masked >= 1.0 else "YIELD EXCEEDED")

    lines.append("| Region | Safety Factor | Status |")
    lines.append("|--------|---------------|--------|")
    lines.append(f"| All Nodes | {sf_all:.2f} | {sf_status_all} |")
    lines.append(f"| Masked (Design) | {sf_masked:.2f} | {sf_status_masked} |")
    lines.append("")
    lines.append("_Note: Use masked value for design decisions (excludes root singularity)._")

    # === 5. Displacement ===
    lines.append("")
    lines.append("## 5. Displacement")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Max Magnitude | {fem_stats.get('max_disp_mag', 0)*1000:.2f} mm |")
    lines.append(f"| Mean Z (downward) | {fem_stats.get('mean_disp_z', 0)*1000:.2f} mm |")

    # === 6. Mesh Quality ===
    if mesh_report:
        mesh_stats = mesh_report.get("stats", {})
        quality = mesh_report.get("quality", {})
        lines.append("")
        lines.append("## 6. Mesh Quality")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Nodes | {mesh_stats.get('nodes', 0):,} |")
        lines.append(f"| Tetrahedra | {mesh_stats.get('tets', 0):,} |")
        lines.append(f"| Quality OK Ratio | {quality.get('quality_ok_ratio', 0)*100:.1f}% |")
        lines.append(f"| Mean Aspect Ratio | {quality.get('mean_aspect_ratio', 0):.2f} |")

    return "\n".join(lines)


def create_comparison_histogram(case_id: str | None):
    """Create FEM vs AI stress comparison histogram."""
    if not case_id:
        return None

    surface_data = _load_surface_results(case_id)
    inference_report = _load_inference_report(case_id)

    if not surface_data or surface_data.get("stress_vm") is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fem_stress = surface_data["stress_vm"] / 1e6  # Convert to MPa
        loss_mask = surface_data.get("loss_mask")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: FEM stress distribution
        ax1 = axes[0]
        ax1.hist(fem_stress, bins=50, alpha=0.7, label="All nodes", color="steelblue", edgecolor="white")
        if loss_mask is not None and np.any(loss_mask):
            ax1.hist(fem_stress[loss_mask], bins=50, alpha=0.5, label="Masked", color="orange", edgecolor="white")
        ax1.axvline(x=YIELD_STRENGTH_MPA, color="red", linestyle="--", linewidth=2, label=f"Yield ({YIELD_STRENGTH_MPA:.0f} MPa)")
        ax1.set_xlabel("von Mises Stress (MPa)")
        ax1.set_ylabel("Node Count")
        ax1.set_title(f"FEM Ground Truth - Case {case_id}")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Right: Error distribution (if inference available)
        ax2 = axes[1]
        if inference_report:
            # Load pred stress from inference report ranges
            error_range = inference_report.get("error_range", [0, 0])
            ai_metrics = inference_report.get("metrics", {})
            ai_masked = ai_metrics.get("masked_nodes", {})

            # Create error info text
            ax2.text(0.5, 0.6, f"Case {case_id} AI Performance",
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax2.transAxes)
            ax2.text(0.5, 0.45,
                    f"MAE (masked): {ai_masked.get('mae', 0)/1e6:.3f} MPa",
                    ha='center', va='center', fontsize=12,
                    transform=ax2.transAxes)
            ax2.text(0.5, 0.35,
                    f"RMSE (masked): {ai_masked.get('rmse', 0)/1e6:.3f} MPa",
                    ha='center', va='center', fontsize=12,
                    transform=ax2.transAxes)
            ax2.text(0.5, 0.25,
                    f"Max Error (masked): {ai_masked.get('max_error', 0)/1e6:.2f} MPa",
                    ha='center', va='center', fontsize=12,
                    transform=ax2.transAxes)

            # Color based on performance
            mae_mpa = ai_masked.get('mae', 0) / 1e6
            if mae_mpa < 1.0:
                color = "green"
                status = "Excellent"
            elif mae_mpa < 3.0:
                color = "orange"
                status = "Good"
            else:
                color = "red"
                status = "Needs Improvement"

            ax2.text(0.5, 0.1, f"Status: {status}",
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    color=color, transform=ax2.transAxes)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, "AI inference not available\n\nRun inference script first",
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.axis('off')

        plt.tight_layout()
        return fig

    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Gradio Handlers
# ---------------------------------------------------------------------------


def handle_refresh():
    """Refresh case dropdown."""
    rows = _list_success_cases()
    rows.sort(key=lambda r: (r.get("case_id") or ""))

    choices = []
    for idx, r in enumerate(rows, start=1):
        label = f"{idx:03d} | {_case_label(r)}"
        value = (r.get("case_id") or "").strip()
        if value:
            choices.append((label, value))

    first_case_id = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first_case_id)


def handle_view_change(case_id: str | None, view_mode: str):
    """Handle view mode change - return appropriate GLB paths."""
    if not case_id:
        return None, None, "Select a case."

    fem_glb = FEM_DIR / case_id / "wing_result.glb"
    pred_glb = FEM_DIR / case_id / "wing_pred.glb"
    error_glb = FEM_DIR / case_id / "wing_error.glb"

    # Get comparison report
    report = get_case_comparison_report(case_id)

    if view_mode == "FEM Only":
        if fem_glb.exists():
            return str(fem_glb), None, report
        return None, None, report + "\n\n_FEM result not found._"

    elif view_mode == "AI Only":
        if pred_glb.exists():
            return str(pred_glb), None, report
        return None, None, report + "\n\n_AI prediction not found. Run inference first._"

    elif view_mode == "Error Map":
        if error_glb.exists():
            return str(error_glb), None, report
        return None, None, report + "\n\n_Error map not found. Run inference first._"

    else:  # Side-by-Side
        left = str(fem_glb) if fem_glb.exists() else None
        right = str(pred_glb) if pred_glb.exists() else None
        return left, right, report


def handle_case_change(case_id: str | None, view_mode: str):
    """Handle full case change - update all outputs."""
    left_model, right_model, report = handle_view_change(case_id, view_mode)
    histogram = create_comparison_histogram(case_id)

    return left_model, right_model, report, histogram


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


with gr.Blocks(title="Deep-FEM-UAV-Wing") as demo:
    gr.Markdown("# Deep-FEM-UAV-Wing")
    gr.Markdown("**FEM vs AI Comparison Dashboard** - GraphSAGE Surrogate Model for Wing Stress Prediction")

    # Model Summary (collapsible)
    with gr.Accordion("GNN Model Summary", open=False):
        model_summary = gr.Markdown(value=get_model_summary())

    # Controls
    with gr.Row():
        case_dropdown = gr.Dropdown(
            label="Select Case",
            choices=[],
            value=None,
            interactive=True,
            scale=2,
        )
        view_mode = gr.Radio(
            choices=["Side-by-Side", "FEM Only", "AI Only", "Error Map"],
            value="Side-by-Side",
            label="View Mode",
            interactive=True,
            scale=2,
        )

    # 3D Viewers - Side by Side
    with gr.Row():
        with gr.Column():
            gr.Markdown("### FEM (Ground Truth)")
            model_fem = gr.Model3D(label="FEM Result", height=350)
        with gr.Column():
            gr.Markdown("### AI Prediction")
            model_ai = gr.Model3D(label="AI Prediction", height=350)

    # Comparison Report and Histogram
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            comparison_report = gr.Markdown(value="Select a case to view comparison.")
        with gr.Column(scale=2):
            comparison_hist = gr.Plot(label="Stress Analysis")

    # Color Legend
    with gr.Accordion("Color Scale Reference", open=False):
        gr.Markdown("""
        ### Stress Visualization (jet colormap)
        - **Blue** → Low stress
        - **Green/Yellow** → Medium stress
        - **Red** → High stress

        ### Error Map (hot colormap)
        - **Black** → Low error (good prediction)
        - **Red/Yellow** → Medium error
        - **White** → High error (poor prediction)

        ### Root Singularity
        - Nodes near wing root (y < 5% span) are excluded from masked metrics
        - High stress at root is expected due to fixed boundary condition
        """)

    # Event handlers
    case_dropdown.change(
        fn=handle_case_change,
        inputs=[case_dropdown, view_mode],
        outputs=[model_fem, model_ai, comparison_report, comparison_hist]
    )
    view_mode.change(
        fn=handle_view_change,
        inputs=[case_dropdown, view_mode],
        outputs=[model_fem, model_ai, comparison_report]
    )

    # Load event
    demo.load(fn=handle_refresh, inputs=None, outputs=[case_dropdown]).then(
        fn=handle_case_change,
        inputs=[case_dropdown, view_mode],
        outputs=[model_fem, model_ai, comparison_report, comparison_hist]
    )


if __name__ == "__main__":
    demo.launch()
