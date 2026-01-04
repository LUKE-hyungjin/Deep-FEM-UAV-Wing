## Deep-FEM-UAV-Wing 개발 로그(1/2): Geometry → Meshing → FEM

  

> 목표: **Blender로 파라메트릭 UAV 날개를 대량 생성(200 케이스)하고**, Gmsh로 **볼륨 테트라 메싱**, CalculiX로 **선형 정적 해석**까지 자동화한 뒤, Gradio에서 **pre-computed 결과를 뷰어로 확인**한다.

  

---

  

## 0-0) 3분 요약 — “이번에 뭘 만들었나?”

PRD/Spec 문서는 “요구사항”이 중심이라 일반인이 보기엔 어렵습니다. 이번 글에서는 한 줄로 정리하면 이거예요:

> **날개 200개를 자동으로 만들고(3D), 자동으로 쪼개서(FEM용 메쉬), 자동으로 눌러보고(하중), 결과를 웹에서 바로 보는 파이프라인**을 만들었습니다.

### 이번에 구현한 것(사람 기준으로 풀어쓰기)

- **(대량 생성)**: Blender를 “백그라운드 모드”로 돌려서 날개를 **200개 자동 생성**했습니다.  
  사람이 200번 클릭할 일을 스크립트가 대신합니다.
- **(해석 준비)**: 3D 표면(STL)을 FEM이 먹을 수 있는 **볼륨 메쉬(테트라)로 자동 변환**했습니다.  
  쉽게 말해 “얇은 껍질”을 “속이 찬 레고 블록”으로 바꿔서 구조해석이 가능해집니다.
- **(하중/고정)**: Root는 고정하고, Upper 면은 압력으로 누르는 **경계조건을 자동으로 정의**했습니다.
- **(실제 FEM 실행)**: CalculiX로 **선형 정적 해석을 자동 실행**하고, 결과(변위/응력)를 파일로 남겼습니다.
- **(보이는 결과)**: Gradio에서 케이스를 고르면 **바로 3D로 확인**할 수 있게 했습니다.  
  “생성 버튼” 대신 **미리 계산된 결과를 보는 뷰어**로 UX를 고정했습니다.

### 결과는 이렇게 보인다(스크린샷 포인트)

- **Stage 1 결과(형상 미리보기)**: `wing_viz.glb`가 Gradio에서 3D로 렌더됨  
  ![[Pasted image 20260104170019.png]]
- **Stage 2 결과(Upper/Root 디버그 색상)**: `surf_sets.glb`에서 Upper/Root가 분리된 상태가 보임  
  ![[Pasted image 20260104220000.png]]
  ![[Pasted image 20260104220042.png]]
- **Stage 3 결과(응력 + 압력 방향 화살표)**: 응력 컬러 + 압력 방향(샘플 200개)을 토글로 확인  
  ![[Pasted image 20260104_pressure_arrows.png]]

---

## 0) 전체 파이프라인 개요

여기서는 “이상적인 계획”이 아니라, **지금 실제로 구현되어 돌아가는 것**만 단계별로 정리합니다.

### Stage 1 — Geometry(형상 생성) + Viz(미리보기) + Viewer(Gradio)

- **무엇을 자동화했나**: 날개 형상을 **200개 자동 생성**하고, 바로 웹에서 볼 수 있게 **미리보기 파일까지 생성**
- **입력**: 랜덤/샘플링된 파라미터(Span/Chord/Sweep/Thickness)
- **출력(케이스별 폴더)**: `data/raw/geometry/{case_id}/`
  - `wing.stl` : 해석/메싱용 “표면 껍질” 파일
  - `wing_viz.glb` : 브라우저 미리보기용 3D 파일
  - `params.json`, `build_report.json` : 어떤 파라미터로 만들었는지/생성 로그
- **사용자 입장에서 보이는 결과(Gradio)**:
  - 케이스를 고르고 **3D로 즉시 확인**
  - `wing.stl` 다운로드
  - 생성 로그 확인

### Stage 2 — Meshing(Gmsh) + Boundary Sets(고정/하중 면 지정) + Debug Viz

- **무엇을 자동화했나**: STL을 구조해석이 가능한 “속이 찬 메쉬(테트라)”로 바꾸고, **고정할 부분/누를 부분을 자동으로 태깅**
- **입력**: `wing.stl`
- **출력(케이스별 폴더)**: `data/raw/mesh/{case_id}/`
  - `wing.msh` : 볼륨 테트라 메쉬
  - `boundary_sets.json` :  
    - `NROOT`(고정할 Root)  
    - `SURF_UPPER`(압력 줄 Upper 면)  
    - `SURF_ALL`(전체 외피)
  - `mesh_report.json` : 메싱 결과 요약
  - `surf_sets.glb` : 위 태깅이 맞는지 “색으로” 확인하는 디버그 3D
- **사용자 입장에서 보이는 결과(Gradio)**:
  - `Meshing Debug` 모드에서 **Upper/Root 분리가 눈으로 보임**

### Stage 3 — FEM(CalculiX) + Postprocess(결과 추출) + Result Viz(응력 컬러)

- **무엇을 자동화했나**: 각 케이스에 대해 **해석 입력 파일을 만들고(ccx 실행), 결과를 표면 기준 데이터/3D로 변환**
- **입력**: `wing.msh` + `boundary_sets.json` + 재료물성(E, ν) + 압력(p)
- **출력(케이스별 폴더)**: `data/raw/fem/{case_id}/`
  - `{case_id}.inp` : CalculiX 입력(고정 + 압력)
  - `{case_id}.frd` : CalculiX 결과 원본
  - `surface_results.npz` : 표면 노드 기준 (좌표/법선/변위/응력/마스크)
  - `wing_result.glb` : 응력을 컬러로 입힌 3D 결과
  - (옵션) `pressure_vectors.glb`, `wing_result_arrows.glb` : 압력 방향 화살표(샘플 200개)
- **사용자 입장에서 보이는 결과(Gradio)**:
  - `FEM Result`에서 **응력 컬러 결과를 3D로 확인**
  - `Show Pressure Arrows` 토글로 **압력 방향(샘플)을 켜고/끄며 확인**

  

  

---

  

  

## 1) Stage 1 — Blender로 날개 200개 생성 + STL→GLB + Gradio Viewer

### 1-1) 배치 생성 (STL + GLB)

아래와 같은 에어포일 형상을 랜덤으로 200개 생성

  

- 실행코드

```bash

  

python scripts/generate_geometry_dataset.py --count 200 --seed 42

  

```

  

![[Pasted image 20260104170019.png]]

- 산출물(케이스별):

- `data/raw/geometry/{case_id}/wing.stl`

- `data/raw/geometry/{case_id}/wing_viz.glb`

- `data/raw/geometry/{case_id}/params.json`

- `data/raw/geometry/{case_id}/build_report.json`

  

- 인덱스:

- `data/raw/geometry/params.csv`

- `data/raw/manifest.json`

  

### 1-2) Gradio에서 GLB가 “빈 화면”으로 보이는 이슈 (JSON glTF 문제)

  

증상: `wing_viz.glb`가 실제로는 **바이너리 GLB가 아니라 JSON glTF가 .glb 확장자로 저장**되면, Gradio `Model3D`에서 렌더가 빈 화면이 될 수 있다.

  

기존 산출물을 한 번에 복구하는 스크립트를 실행한다.

```bash

  

python scripts/repair_geometry_glb.py

  

```

  
  

![[Pasted image 20260104170318.png]]

  

---

  

## 2) Stage 2 — Meshing(Gmsh) + Boundary Sets + surf_sets.glb 디버그

  

### 2-1) 배치 메싱 실행

  

```bash

  

python scripts/generate_mesh_dataset.py --limit 0

  

```

  

  

케이스별 산출물:

- `data/raw/mesh/{case_id}/wing.msh` (볼륨 테트라)

- `data/raw/mesh/{case_id}/mesh_report.json`

- `data/raw/mesh/{case_id}/boundary_sets.json`

- `data/raw/mesh/{case_id}/surf_sets.glb` (디버그: Root/Upper 시각화)

  

  

### 2-2) Boundary Sets 정의(핵심)

  

- `NROOT`: Root 고정용 노드 집합 (기본 `y <= y_tol`)

- `SURF_ALL`: 외피(face) 전체

- `SURF_UPPER`: 외피 중 Upper 면(압력 하중 적용)

- 기본 규칙: `n_z >= nz_min`

- Root 근방 제외(경계조건 영역 혼입 방지)

  

  

### 2-3) surf_sets.glb 디버그에서 “색이 깨지는” 문제와 해결

  

문제: 메쉬 삼각형의 winding/법선 방향이 뒤섞이면, Upper 분류가 들쑥날쑥해지면서 파랑/회색이 이상하게 보인다.

  

  

해결(요지):

  

- 삼각형 인접 그래프(공유 엣지)로 DFS를 돌려 **와인딩을 일관화**

- `dot(n, C_f - C_vol)` 기준으로 outward 방향을 맞추는 방식으로 안정화

- Upper 후보 면에서 “작은 조각”이 끼는 문제는 **연결 컴포넌트(면적 최대 1개만 유지)**로 정리

  

![[Pasted image 20260104220000.png]]

![[Pasted image 20260104220042.png]]

  

  

---

  

  

## 3) Stage 3 — FEM(CalculiX) + Postprocess + 결과 GLB(응력) + 압력 화살표 디버그

  

  

### 3-1) macOS(Homebrew)에서 CalculiX 설치 팁 (ccx 이름 주의)

  

  

Homebrew의 `calculix-ccx`는 `ccx`라는 이름 대신 **버전이 붙은 `ccx_2.22`**로 설치될 수 있다.

  

  

```bash

  

brew tap costerwi/homebrew-calculix

  

brew install calculix-ccx calculix-cgx

  

ls -la "$(brew --prefix calculix-ccx)/bin/" | grep ccx

  

```

  

  

### 3-2) Python venv 필수 (PEP 668 회피)

  

  

시스템 파이썬에서 `pip install`이 막히는 경우가 많아서, 반드시 venv를 사용한다.

  

  

```bash

  

python3 -m venv .venv

  

source .venv/bin/activate

  

python -m pip install -r requirements.txt

  

```

  

  

### 3-3) FEM 배치 실행

  

  

```bash

  

python scripts/generate_fem_dataset.py --limit 0 --pressure 5000

  

```

  

  

케이스별 산출물:

  

- `data/raw/fem/{case_id}/{case_id}.inp`

  

- `data/raw/fem/{case_id}/{case_id}.frd`

  

- `data/raw/fem/{case_id}/surface_results.npz`

  

- `data/raw/fem/{case_id}/wing_result.glb`

  

- `data/raw/fem/{case_id}/fem_report.json`

  

  

### 3-4) 하중(Pressure) 적용 방식 — 등가 절점 하중(*CLOAD)

  

  

압력은 face 기준으로 다음처럼 노드 힘으로 바꿔 `*CLOAD`로 적용한다.

  

  

- face 힘: \(\mathbf{F}_f = p \cdot A_f \cdot (-\hat n)\)

  

- 노드 분배: 삼각형 3개 정점에 \(\mathbf{F}_f/3\)씩 누적

  

  

※ `*CLOAD`는 **반드시 `*STEP` 블록 안**에 있어야 한다(밖에 두면 ccx 에러).

  

  

### 3-5) 후처리(Postprocess): ccx2paraview 없이도 FRD ASCII 파싱으로 처리

  

  

`ccx2paraview`는 설치가 번거로울 수 있어, `.frd`가 ASCII인 경우엔

  

- `-4 DISP` 섹션 (변위)

  

- `-4 STRESS` 섹션 (응력)

  

을 직접 파싱해서 `surface_results.npz`와 `wing_result.glb`를 만든다.

  

  

### 3-6) “압력 방향 화살표” 디버그 (샘플 200개)

  

  

Upper 분리/하중 방향이 맞는지 확인하기 위해, `SURF_UPPER` face에서 **200개만 샘플링**해 face 중심에 화살표를 그린다.

  

  

생성 파일:

  

- `pressure_vectors.glb` (화살표만)

  

- `wing_result_arrows.glb` (응력 + 화살표 합성)

  

  

Gradio에서:

  

- `Preview Mode = FEM Result`

  

- `Show Pressure Arrows (sampled)` 체크 ON/OFF로 토글

  

  

**이미지 위치(필수)**: 응력 결과 + 압력 화살표(샘플) 스크린샷

  

![[Pasted image 20260104_pressure_arrows.png]]

  

  

---

  

  

## 4) 다음 단계(예고)

  

  

- **Stage 4**: FEM solved 데이터 200 케이스 누적/검증 자동화(`nan/inf`, 스케일 sanity, 실패 원인 집계)

  

- **Stage 5**: 표면 그래프 데이터셋 빌드 → GraphSAGE baseline 학습/추론

  

- **Stage 6**: Gradio에서 FEM vs AI side-by-side + error map + metrics