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

### 3) Gradio 앱 실행 (뷰어)

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:7860`로 접속하면, Blender 배치로 생성된 케이스 목록을 선택해 **3D 미리보기(GLB)** / **STL 다운로드** / **로그 확인**을 할 수 있습니다.

## 배치 생성 (데이터셋용: wing 여러 개 만들기)

실제 데이터셋을 쌓으려면 **Blender + 파이썬 스크립트로 케이스를 여러 개 생성**하고, Gradio는 생성된 결과를 확인하는 뷰어로 사용합니다.

### 1) (권장) Blender 경로 지정

```bash
export BLENDER_BIN="/Applications/Blender.app/Contents/MacOS/Blender"
```

### 2) N개 케이스 생성(STL + GLB, 캐시 재사용)

```bash
python scripts/generate_geometry_dataset.py --count 200 --seed 42
```

- 산출물: `data/raw/geometry/{case_id}/wing.stl`, `wing_viz.glb`, `params.json`, `build_report.json`
- 인덱스: `data/raw/geometry/params.csv`, `data/raw/manifest.json`

### 2-1) (중요) GLB 정규화/복구(한 번만)

환경에 따라 `wing_viz.glb`가 **바이너리 GLB**가 아니라 **JSON glTF가 `.glb`로 저장**되는 케이스가 있습니다.  
이 경우 Gradio `Model3D`가 **빈 화면**이 될 수 있으므로, 아래 스크립트를 **한 번 실행**해 기존 산출물을 바이너리 GLB로 복구하세요.

```bash
python scripts/repair_geometry_glb.py
```

### 2-2) 2단계 Meshing(Gmsh) 배치 생성(선택)

`wing.stl`에서 테트라 볼륨 메쉬 및 경계 셋(`boundary_sets.json`)을 생성합니다.

> 사전 준비: 로컬에 `gmsh`가 설치되어 있어야 합니다.
> - macOS(Homebrew): `brew install gmsh`
> - 확인: `gmsh -version`

```bash
python scripts/generate_mesh_dataset.py --limit 0
```

- 산출물: `data/raw/mesh/{case_id}/wing.msh`, `boundary_sets.json`, `mesh_report.json`, `surf_sets.glb`(Upper/Root 디버그 시각화)

### 2-4) 3단계 FEM(CalculiX) + Postprocess 배치 생성(선택)

`wing.msh`와 `boundary_sets.json`을 이용해 CalculiX로 선형 정적 해석을 실행하고, 표면 결과(`surface_results.npz`) 및 결과 GLB(`wing_result.glb`)를 생성합니다.

> 사전 준비(권장: Homebrew):
> - `ccx` (CalculiX)
> - (선택) `ccx2paraview`가 있으면 후처리가 더 안정적입니다. 없으면 `.frd`를 직접 파싱하는 폴백을 사용합니다.

```bash
brew tap costerwi/homebrew-calculix
brew install calculix-ccx calculix-cgx
# Homebrew는 `ccx` 대신 버전이 붙은 바이너리(`ccx_2.22` 등)를 설치할 수 있습니다.
ls -la "$(brew --prefix calculix-ccx)/bin/" | grep ccx
"$(brew --prefix calculix-ccx)/bin/ccx_"* -v
```

```bash
python scripts/generate_fem_dataset.py --limit 0 --pressure 5000
```

- 산출물: `data/raw/fem/{case_id}/{case_id}.inp`, `{case_id}.frd`, `surface_results.npz`, `wing_result.glb`, `fem_report.json`

### 2-3) 생성 결과 확인(Gradio 뷰어)

```bash
python app.py
```

- `Preview Mode`에서 **FEM Result (wing_result.glb)** 를 선택하면 `data/raw/fem/{case_id}/wing_result.glb`를 바로 확인할 수 있습니다.
- **Show Pressure Arrows**를 켜면 `SURF_UPPER`에서 샘플링한 압력 방향 화살표(기본 200개)가 함께 표시됩니다(`wing_result_arrows.glb`).

### 3) Blender 필수 모드(폴백 금지)

```bash
python scripts/generate_geometry_dataset.py --count 50 --seed 7 --require_blender
```

## 배포 (Hugging Face Spaces 등)

- 이 레포는 `app.py`가 Gradio 엔트리포인트이고, `requirements.txt`를 포함하므로 Spaces에 그대로 올려 배포할 수 있습니다.
- **주의**: Spaces 런타임에 Blender가 기본 포함되지 않는 경우가 많습니다. 이 경우 앱은 Blender 대신 **파이썬 폴백 메쉬 생성**으로 동작합니다(3D 미리보기는 가능).
