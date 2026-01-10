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

# Material constants (AL6061-T6)
YIELD_STRENGTH_PA = 276e6  # 276 MPa
YIELD_STRENGTH_MPA = 276.0


# ---------------------------------------------------------------------------
# Engineering Dashboard Functions
# ---------------------------------------------------------------------------


def _load_validation_report() -> dict | None:
    """Load the validation report if it exists."""
    report_path = RAW_DIR / "validation_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


def _load_fem_report(case_id: str) -> dict | None:
    """Load FEM report for a specific case."""
    report_path = FEM_DIR / case_id / "fem_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
    return None


def _load_geometry_params(case_id: str) -> dict | None:
    """Load geometry parameters for a specific case."""
    params_path = GEOMETRY_DIR / case_id / "params.json"
    if params_path.exists():
        return json.loads(params_path.read_text(encoding="utf-8"))
    return None


def _load_mesh_report(case_id: str) -> dict | None:
    """Load mesh report for a specific case."""
    report_path = MESH_DIR / case_id / "mesh_report.json"
    if report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8"))
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


def get_dataset_summary() -> str:
    """Generate dataset summary for the dashboard."""
    report = _load_validation_report()
    if not report:
        return "Validation report not found. Run: `python -m src.deep_fem_uav_wing.validate_dataset`"

    lines = [
        "## Dataset Summary",
        "",
        f"**Total Cases:** {report.get('total_cases', 0)}",
        f"**Valid:** {report.get('valid_cases', 0)} | **Warning:** {report.get('warning_cases', 0)} | **Failed:** {report.get('failed_cases', 0)}",
        "",
    ]

    # Statistics
    stats = report.get("statistics", {})
    if stats:
        stress_stats = stats.get("stress_max_mpa", {})
        disp_stats = stats.get("disp_max_m", {})
        mesh_stats = stats.get("mesh_nodes", {})

        lines.extend([
            "### Stress (von Mises, MPa)",
            f"- Min: {stress_stats.get('min', 0):.1f} | Mean: {stress_stats.get('mean', 0):.1f} | Max: {stress_stats.get('max', 0):.1f} | P95: {stress_stats.get('p95', 0):.1f}",
            "",
            "### Displacement (m)",
            f"- Min: {disp_stats.get('min', 0)*1000:.1f} mm | Mean: {disp_stats.get('mean', 0)*1000:.1f} mm | Max: {disp_stats.get('max', 0)*1000:.1f} mm",
            "",
            "### Mesh",
            f"- Nodes: {mesh_stats.get('min', 0):,} ~ {mesh_stats.get('max', 0):,} (mean: {mesh_stats.get('mean', 0):,.0f})",
        ])

    return "\n".join(lines)


def get_case_engineering_report(case_id: str | None) -> tuple[str, dict | None]:
    """Generate engineering report for a specific case."""
    if not case_id:
        return "Select a case to view engineering report.", None

    fem_report = _load_fem_report(case_id)
    if not fem_report:
        return f"FEM report not found for case {case_id}.", None

    params = _load_geometry_params(case_id)
    mesh_report = _load_mesh_report(case_id)

    # Extract data
    fem_params = fem_report.get("params", {})
    stats = fem_report.get("stats", {})
    metrics_all = fem_report.get("metrics_all_nodes", {})
    metrics_masked = fem_report.get("metrics_masked_nodes", {})
    surface_mask = fem_report.get("surface_mask", {})

    # Calculate safety factors
    stress_max_all = metrics_all.get("stress_max", 0)
    stress_max_masked = metrics_masked.get("stress_max", 0)
    sf_all = YIELD_STRENGTH_PA / stress_max_all if stress_max_all > 0 else float("inf")
    sf_masked = YIELD_STRENGTH_PA / stress_max_masked if stress_max_masked > 0 else float("inf")

    # Geometry info
    geom_lines = ["## Geometry"]
    if params:
        span = params.get("span_m", 0)
        chord = params.get("chord_m", 0)
        ar = span / chord if chord > 0 else 0
        geom_lines.extend([
            f"- **Span:** {span*1000:.1f} mm",
            f"- **Chord:** {chord*1000:.1f} mm",
            f"- **Aspect Ratio:** {ar:.2f}",
            f"- **Sweep:** {params.get('sweep_deg', 0):.1f} deg",
            f"- **Thickness Ratio:** {params.get('thickness_ratio', 0)*100:.1f} %",
        ])

    # Material & Loading
    mat_lines = [
        "",
        "## Material & Loading",
        f"- **Material:** AL6061-T6",
        f"- **Young's Modulus:** {fem_params.get('young_modulus_pa', 0)/1e9:.0f} GPa",
        f"- **Poisson's Ratio:** {fem_params.get('poisson_ratio', 0):.2f}",
        f"- **Yield Strength:** {YIELD_STRENGTH_MPA:.0f} MPa",
        f"- **Applied Pressure:** {fem_params.get('pressure_pa', 0)/1000:.1f} kPa",
        f"- **Boundary Condition:** Root fixed (Cantilever)",
    ]

    # Stress Results
    stress_lines = [
        "",
        "## Stress Results (von Mises)",
        "",
        "| Metric | All Nodes | Masked (excl. root) |",
        "|--------|-----------|---------------------|",
        f"| Max | {stress_max_all/1e6:.1f} MPa | {stress_max_masked/1e6:.1f} MPa |",
        f"| Mean | {metrics_all.get('stress_mean', 0)/1e6:.1f} MPa | {metrics_masked.get('stress_mean', 0)/1e6:.1f} MPa |",
        f"| P95 | {metrics_all.get('stress_p95', 0)/1e6:.1f} MPa | {metrics_masked.get('stress_p95', 0)/1e6:.1f} MPa |",
        f"| P99 | {metrics_all.get('stress_p99', 0)/1e6:.1f} MPa | {metrics_masked.get('stress_p99', 0)/1e6:.1f} MPa |",
        "",
        f"**Root Singularity Band:** y < {surface_mask.get('root_band_y', 0)*1000:.1f} mm ({surface_mask.get('loss_mask_false', 0)} nodes excluded)",
    ]

    # Safety Factor
    sf_status_all = "SAFE" if sf_all >= 1.5 else ("MARGINAL" if sf_all >= 1.0 else "YIELD EXCEEDED")
    sf_status_masked = "SAFE" if sf_masked >= 1.5 else ("MARGINAL" if sf_masked >= 1.0 else "YIELD EXCEEDED")

    sf_lines = [
        "",
        "## Safety Factor",
        f"- **SF (all nodes):** {sf_all:.2f} - {sf_status_all}",
        f"- **SF (masked):** {sf_masked:.2f} - {sf_status_masked}",
        "",
        "_Note: Root singularity causes artificially high stress. Use masked value for design._",
    ]

    # Displacement
    disp_lines = [
        "",
        "## Displacement",
        f"- **Max Magnitude:** {stats.get('max_disp_mag', 0)*1000:.2f} mm",
        f"- **Mean Z (downward):** {stats.get('mean_disp_z', 0)*1000:.2f} mm",
    ]

    # Mesh Quality
    mesh_lines = ["", "## Mesh Quality"]
    if mesh_report:
        mesh_stats = mesh_report.get("stats", {})
        quality = mesh_report.get("quality", {})
        mesh_lines.extend([
            f"- **Nodes:** {mesh_stats.get('nodes', 0):,}",
            f"- **Tetrahedra:** {mesh_stats.get('tets', 0):,}",
            f"- **Quality OK Ratio:** {quality.get('quality_ok_ratio', 0)*100:.1f}%",
            f"- **Mean Aspect Ratio:** {quality.get('mean_aspect_ratio', 0):.2f}",
        ])

    full_report = "\n".join(geom_lines + mat_lines + stress_lines + sf_lines + disp_lines + mesh_lines)

    # Return plot data for histogram
    plot_data = {
        "stress_max_all_mpa": stress_max_all / 1e6,
        "stress_max_masked_mpa": stress_max_masked / 1e6,
        "sf_all": sf_all,
        "sf_masked": sf_masked,
    }

    return full_report, plot_data


def create_stress_histogram(case_id: str | None):
    """Create stress distribution histogram using matplotlib."""
    if not case_id:
        return None

    surface_data = _load_surface_results(case_id)
    if not surface_data or surface_data.get("stress_vm") is None:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stress_vm = surface_data["stress_vm"]
        loss_mask = surface_data.get("loss_mask")

        stress_mpa = stress_vm / 1e6

        fig, ax = plt.subplots(figsize=(8, 4))

        # Histogram of all nodes
        ax.hist(stress_mpa, bins=50, alpha=0.7, label="All nodes", color="steelblue", edgecolor="white")

        # Histogram of masked nodes (if available)
        if loss_mask is not None and np.any(loss_mask):
            stress_masked = stress_mpa[loss_mask]
            ax.hist(stress_masked, bins=50, alpha=0.5, label="Masked (excl. root)", color="orange", edgecolor="white")

        # Yield line
        ax.axvline(x=YIELD_STRENGTH_MPA, color="red", linestyle="--", linewidth=2, label=f"Yield ({YIELD_STRENGTH_MPA:.0f} MPa)")

        ax.set_xlabel("von Mises Stress (MPa)")
        ax.set_ylabel("Node Count")
        ax.set_title(f"Stress Distribution - Case {case_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except ImportError:
        return None


def create_safety_factor_gauge(sf_value: float, title: str = "Safety Factor"):
    """Create a simple safety factor visualization."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 2))

        # Color based on SF value
        if sf_value >= 2.0:
            color = "green"
            status = "SAFE"
        elif sf_value >= 1.5:
            color = "limegreen"
            status = "SAFE"
        elif sf_value >= 1.0:
            color = "orange"
            status = "MARGINAL"
        else:
            color = "red"
            status = "YIELD"

        # Simple bar
        ax.barh([0], [min(sf_value, 3.0)], color=color, height=0.5)
        ax.axvline(x=1.0, color="red", linestyle="--", linewidth=2)
        ax.axvline(x=1.5, color="orange", linestyle="--", linewidth=1)
        ax.axvline(x=2.0, color="green", linestyle="--", linewidth=1)

        ax.set_xlim(0, 3.0)
        ax.set_yticks([])
        ax.set_xlabel("Safety Factor")
        ax.set_title(f"{title}: {sf_value:.2f} ({status})")

        plt.tight_layout()
        return fig

    except ImportError:
        return None


def _list_success_cases() -> list[dict[str, str]]:
    if not PARAMS_CSV.exists():
        return []

    with PARAMS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    success_rows = [r for r in rows if (r.get("status") or "").strip() == "success"]
    # newest first (append-only CSV)
    return list(reversed(success_rows))


def _case_label(row: dict[str, str]) -> str:
    case_id = (row.get("case_id") or "").strip()
    span_m = (row.get("span_m") or "").strip()
    chord_m = (row.get("chord_m") or "").strip()
    sweep_deg = (row.get("sweep_deg") or "").strip()
    thickness_ratio = (row.get("thickness_ratio") or "").strip()
    return f"{case_id} | span={span_m} chord={chord_m} sweep={sweep_deg} t/c={thickness_ratio}"


def _case_dir(case_id: str) -> Path:
    return GEOMETRY_DIR / case_id


def handle_refresh():
    rows = _list_success_cases()
    # Sort by case_id for consistent ordering
    rows.sort(key=lambda r: (r.get("case_id") or ""))
    
    choices = []
    for idx, r in enumerate(rows, start=1):
        label = f"{idx:03d} | {_case_label(r)}"
        value = (r.get("case_id") or "").strip()
        if value:
            choices.append((label, value))
            
    first_case_id = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first_case_id)


def handle_select_case(case_id: str | None, preview_mode: str, show_pressure_arrows: bool):
    """Handle case selection and return 3D model path and engineering report."""
    if not case_id:
        return None, "Select a case to view results."

    case_dir = _case_dir(case_id)
    glb_path = case_dir / "wing_viz.glb"
    stl_path = case_dir / "wing.stl"
    surf_sets_glb = MESH_DIR / case_id / "surf_sets.glb"
    fem_result_glb = FEM_DIR / case_id / "wing_result.glb"
    fem_result_arrows_glb = FEM_DIR / case_id / "wing_result_arrows.glb"

    if not glb_path.exists() or not stl_path.exists():
        missing: list[str] = []
        if not glb_path.exists():
            missing.append("wing_viz.glb")
        if not stl_path.exists():
            missing.append("wing.stl")
        return None, f"필수 산출물이 없습니다: {', '.join(missing)}"

    # Preview selection
    if preview_mode == "Meshing Debug (surf_sets.glb)":
        if not surf_sets_glb.exists():
            return None, "Meshing Debug를 선택했지만 surf_sets.glb가 없습니다."
        preview_path = surf_sets_glb
    elif preview_mode == "FEM Result (wing_result.glb)":
        target = fem_result_arrows_glb if show_pressure_arrows else fem_result_glb
        if not target.exists():
            return None, "FEM Result를 선택했지만 wing_result.glb가 없습니다."
        preview_path = target
    else:
        preview_path = glb_path

    head = preview_path.read_bytes()[:4]
    if head != b"glTF":
        return None, "GLB 파일이 올바르지 않습니다."

    # Generate engineering report
    report_text, _ = get_case_engineering_report(case_id)
    return str(preview_path), report_text


with gr.Blocks(title="Deep-FEM-UAV-Wing") as demo:
    gr.Markdown("# Deep-FEM-UAV-Wing")
    gr.Markdown("**UAV Wing FEM Analysis Dashboard** - 3D Viewer + Engineering Report")

    # Controls Row
    with gr.Row():
        case_dropdown = gr.Dropdown(
            label="Select Case",
            choices=[],
            value=None,
            interactive=True,
            scale=2,
        )
        preview_mode = gr.Radio(
            choices=["Geometry (wing_viz.glb)", "Meshing Debug (surf_sets.glb)", "FEM Result (wing_result.glb)"],
            value="FEM Result (wing_result.glb)",
            label="Preview Mode",
            interactive=True,
            scale=2,
        )
        show_pressure_arrows = gr.Checkbox(
            label="Show Pressure Arrows",
            value=False,
            interactive=True,
            scale=1,
        )

    # 3D Model Viewer (Top)
    model = gr.Model3D(label="3D Preview", height=400)

    # Engineering Report (Below)
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=2):
            eng_report = gr.Markdown(value="Select a case to view engineering report.")
        with gr.Column(scale=1):
            sf_plot = gr.Plot(label="Safety Factor")

    with gr.Row():
        stress_hist = gr.Plot(label="Stress Distribution")

    # -----------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------

    def handle_case_change(case_id: str | None, preview_mode: str, show_pressure_arrows: bool):
        """Handle case selection - update 3D model and engineering report."""
        model_path, report_text = handle_select_case(case_id, preview_mode, show_pressure_arrows)

        # Generate plots
        _, plot_data = get_case_engineering_report(case_id)
        stress_fig = create_stress_histogram(case_id)
        sf_fig = None
        if plot_data and plot_data.get("sf_masked"):
            sf_fig = create_safety_factor_gauge(plot_data["sf_masked"], "Safety Factor (Masked)")

        return model_path, report_text, sf_fig, stress_fig

    # Event bindings
    case_dropdown.change(
        fn=handle_case_change,
        inputs=[case_dropdown, preview_mode, show_pressure_arrows],
        outputs=[model, eng_report, sf_plot, stress_hist]
    )
    preview_mode.change(
        fn=handle_case_change,
        inputs=[case_dropdown, preview_mode, show_pressure_arrows],
        outputs=[model, eng_report, sf_plot, stress_hist]
    )
    show_pressure_arrows.change(
        fn=handle_case_change,
        inputs=[case_dropdown, preview_mode, show_pressure_arrows],
        outputs=[model, eng_report, sf_plot, stress_hist]
    )

    # Load event - auto-refresh on page load
    demo.load(fn=handle_refresh, inputs=None, outputs=[case_dropdown]).then(
        fn=handle_case_change,
        inputs=[case_dropdown, preview_mode, show_pressure_arrows],
        outputs=[model, eng_report, sf_plot, stress_hist]
    )


if __name__ == "__main__":
    demo.launch()


