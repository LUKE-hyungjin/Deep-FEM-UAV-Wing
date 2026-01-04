# Deep-FEM-UAV-Wing

**AI 기반 UAV 날개 구조 해석 예측(Surrogate Modeling)**을 위한 End-to-End 파이프라인(형상 생성 → 해석 → 학습 → 예측/시각화) 프로젝트입니다.

- **스펙/로드맵 문서**: `docs/spec.md`
- **Blender API로 3D 날개 모델 생성**: `blender/generate_wing.py`
- **Python 웹 데모(Gradio)**: `app.py` (3D 미리보기 + 다운로드)

Gradio는 Python만으로 ML/3D 데모를 빠르게 만들 수 있는 웹 UI 프레임워크입니다: [Gradio](https://www.gradio.app/)

## 빠른 시작 (로컬)

### 1) Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (선택) Blender 설치

- 로컬에 Blender가 설치되어 있으면 **실제 Blender API(`bpy`)**로 `.glb` 모델을 생성합니다.
- Blender 실행 파일 경로를 환경변수로 지정할 수 있습니다.

```bash
export BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
```

### 3) Gradio 앱 실행

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:7860`로 접속하면, 슬라이더로 파라미터를 바꾸고 3D 모델을 확인/다운로드할 수 있습니다.

## 배포 (Hugging Face Spaces 등)

- 이 레포는 `app.py`가 Gradio 엔트리포인트이고, `requirements.txt`를 포함하므로 Spaces에 그대로 올려 배포할 수 있습니다.
- **주의**: Spaces 런타임에 Blender가 기본 포함되지 않는 경우가 많습니다. 이 경우 앱은 Blender 대신 **파이썬 폴백 메쉬 생성**으로 동작합니다(3D 미리보기는 가능).
