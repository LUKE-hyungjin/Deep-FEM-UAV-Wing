from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

GEOMETRY_DIR = PROJECT_ROOT / "data" / "raw" / "geometry"
MESH_DIR = PROJECT_ROOT / "data" / "raw" / "mesh"
FEM_DIR = PROJECT_ROOT / "data" / "raw" / "fem"
PARAMS_CSV = GEOMETRY_DIR / "params.csv"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def _format_case_log(case_id: str) -> str:
    report_path = _case_dir(case_id) / "build_report.json"
    params_path = _case_dir(case_id) / "params.json"

    parts: list[str] = [f"case_id={case_id}"]

    if params_path.exists():
        parts.append(f"[params.json]\n{_read_text(params_path).strip()}")
    else:
        parts.append("[params.json]\nMISSING")

    if report_path.exists():
        report = json.loads(_read_text(report_path))
        stdout_tail = (report.get("stdout_tail") or "").strip()
        stderr_tail = (report.get("stderr_tail") or "").strip()
        parts.append(f"[build_report.json]\nstatus={report.get('status')} elapsed_ms={report.get('elapsed_ms')}")
        if report.get("failure_reason"):
            parts.append(f"failure_reason={report.get('failure_reason')}")
        if stdout_tail:
            parts.append(f"[stdout_tail]\n{stdout_tail}")
        if stderr_tail:
            parts.append(f"[stderr_tail]\n{stderr_tail}")
    else:
        parts.append("[build_report.json]\nMISSING")

    return "\n\n".join(parts).strip()


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


def handle_slider_change(case_num: int, choices):
    target_id = f"{case_num:03d}"
    # Check if target_id exists in loaded choices
    # choices is list of (label, value) or just values. Gradio Dropdown choices usually internal.
    # We will rely on the dropdown update to validate.
    return target_id


def handle_select_case(case_id: str | None, preview_mode: str, show_pressure_arrows: bool):
    if not case_id:
        return None, None, "선택된 case_id가 없습니다."

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
        return None, None, f"필수 산출물이 없습니다: {', '.join(missing)}\n{_format_case_log(case_id)}"

    # Preview selection
    if preview_mode == "Meshing Debug (surf_sets.glb)":
        if not surf_sets_glb.exists():
            return (
                None,
                str(stl_path),
                "Meshing Debug를 선택했지만 `data/raw/mesh/{case_id}/surf_sets.glb`가 없습니다.\n"
                "해결: `python scripts/generate_mesh_dataset.py --limit 0` 실행 후 Refresh list.\n\n"
                + _format_case_log(case_id),
            )
        preview_path = surf_sets_glb
    elif preview_mode == "FEM Result (wing_result.glb)":
        target = fem_result_arrows_glb if show_pressure_arrows else fem_result_glb
        if not target.exists():
            return (
                None,
                str(stl_path),
                "FEM Result를 선택했지만 `data/raw/fem/{case_id}/wing_result.glb`가 없습니다.\n"
                "해결: `python scripts/generate_fem_dataset.py --limit 1`로 테스트하거나, 전체는 `--limit 0`.\n"
                "- pressure arrows 토글이 켜져있다면 `wing_result_arrows.glb`가 생성돼야 합니다.\n\n"
                + _format_case_log(case_id),
            )
        preview_path = target
    else:
        preview_path = glb_path

    head = preview_path.read_bytes()[:4]
    if head != b"glTF":
        fix_hint = (
            "wing_viz.glb가 바이너리 GLB가 아닙니다(렌더링이 빈 화면일 수 있음).\n"
            "해결: `python scripts/repair_geometry_glb.py` 실행 후 다시 새로고침하세요.\n\n"
        )
        if preview_path == surf_sets_glb:
            fix_hint = (
                "surf_sets.glb가 바이너리 GLB가 아닙니다(렌더링이 빈 화면일 수 있음).\n"
                "해결: Meshing을 다시 생성하거나 GLB export 환경을 점검하세요.\n\n"
            )
        return None, str(stl_path), fix_hint + _format_case_log(case_id)

    base_log = _format_case_log(case_id)
    header = f"[preview]\nmode={preview_mode}\nfile={preview_path}\n"
    return str(preview_path), str(stl_path), (header + "\n" + base_log).strip()


with gr.Blocks(title="Deep-FEM-UAV-Wing (Week 1)") as demo:
    gr.Markdown(
        """
### Deep-FEM-UAV-Wing — Geometry Demo
- **목표(Week 1)**: Blender 배치로 생성된 케이스를 **Gradio에서 3D 미리보기 + STL 다운로드 + 로그 확인**
        """.strip()
    )

    with gr.Row():
        with gr.Column(scale=1):
            case_dropdown = gr.Dropdown(
                label="Select Case (Success only)",
                choices=[],
                value=None,
                interactive=True,
            )
            preview_mode = gr.Radio(
                choices=["Geometry (wing_viz.glb)", "Meshing Debug (surf_sets.glb)", "FEM Result (wing_result.glb)"],
                value="FEM Result (wing_result.glb)",
                label="Preview Mode",
                interactive=True,
            )
            show_pressure_arrows = gr.Checkbox(
                label="Show Pressure Arrows (sampled)",
                value=False,
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh list", variant="secondary")

        with gr.Column(scale=1):
            model = gr.Model3D(label="3D Preview (GLB)")
            stl_file = gr.File(label="Download STL")
            log = gr.Textbox(label="Log", lines=16)

    def _get_case_id_by_index(index_1based: int) -> str | None:
        rows = _list_success_cases()
        # Sort by case_id to match the sorted dropdown list
        rows.sort(key=lambda r: (r.get("case_id") or ""))
        
        idx = int(index_1based) - 1
        if 0 <= idx < len(rows):
            return (rows[idx].get("case_id") or "").strip()
        return None

    refresh_btn.click(fn=handle_refresh, inputs=None, outputs=[case_dropdown])
    case_dropdown.change(fn=handle_select_case, inputs=[case_dropdown, preview_mode, show_pressure_arrows], outputs=[model, stl_file, log])
    preview_mode.change(fn=handle_select_case, inputs=[case_dropdown, preview_mode, show_pressure_arrows], outputs=[model, stl_file, log])
    show_pressure_arrows.change(fn=handle_select_case, inputs=[case_dropdown, preview_mode, show_pressure_arrows], outputs=[model, stl_file, log])

    demo.load(fn=handle_refresh, inputs=None, outputs=[case_dropdown]).then(
        fn=handle_select_case, inputs=[case_dropdown, preview_mode, show_pressure_arrows], outputs=[model, stl_file, log]
    )


if __name__ == "__main__":
    demo.launch()


