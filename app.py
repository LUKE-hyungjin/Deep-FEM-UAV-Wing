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
    choices = [(_case_label(r), (r.get("case_id") or "").strip()) for r in rows if (r.get("case_id") or "").strip()]
    first_case_id = choices[0][1] if choices else None
    return gr.update(choices=choices, value=first_case_id)


def handle_select_case(case_id: str | None):
    if not case_id:
        return None, None, "선택된 case_id가 없습니다."

    case_dir = _case_dir(case_id)
    glb_path = case_dir / "wing_viz.glb"
    stl_path = case_dir / "wing.stl"

    if not glb_path.exists() or not stl_path.exists():
        missing: list[str] = []
        if not glb_path.exists():
            missing.append("wing_viz.glb")
        if not stl_path.exists():
            missing.append("wing.stl")
        return None, None, f"필수 산출물이 없습니다: {', '.join(missing)}\n{_format_case_log(case_id)}"

    # Validate GLB magic ('glTF'). If not, Model3D may render blank.
    head = glb_path.read_bytes()[:4]
    if head != b"glTF":
        return (
            None,
            str(stl_path),
            "wing_viz.glb가 바이너리 GLB가 아닙니다(렌더링이 빈 화면일 수 있음).\n"
            "해결: `python scripts/repair_geometry_glb.py` 실행 후 다시 새로고침하세요.\n\n"
            + _format_case_log(case_id),
        )

    return str(glb_path), str(stl_path), _format_case_log(case_id)


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
                label="case_id (success only)",
                choices=[],
                value=None,
                interactive=True,
            )
            refresh_btn = gr.Button("Refresh list", variant="secondary")

        with gr.Column(scale=1):
            model = gr.Model3D(label="3D Preview (GLB)")
            stl_file = gr.File(label="Download STL")
            log = gr.Textbox(label="Log", lines=16)

    refresh_btn.click(fn=handle_refresh, inputs=None, outputs=[case_dropdown])
    case_dropdown.change(fn=handle_select_case, inputs=[case_dropdown], outputs=[model, stl_file, log])

    demo.load(fn=handle_refresh, inputs=None, outputs=[case_dropdown]).then(
        fn=handle_select_case, inputs=[case_dropdown], outputs=[model, stl_file, log]
    )


if __name__ == "__main__":
    demo.launch()


