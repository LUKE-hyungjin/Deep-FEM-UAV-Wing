# AI 기반 UAV 날개 구조 해석 예측 모델 (Surrogate Modeling) — 상세 Spec & 로드맵

## 목표(핵심 메시지)
- **End-to-End 파이프라인**(데이터 생성 → 해석 → 학습 → 예측/시각화)을 **혼자서 재현 가능하게 구축**한다.
- 입력은 **형상(3D 메쉬/그래프)**, 출력은 **노드별 응력(von Mises) 및 변위**로 정의한다.
- 데모는 **FEM(느림/정확)** vs **AI(빠름/근사)**를 **시각적으로 한 화면**에서 비교한다.

---

## 이번 버전 “결정사항”(작동 보장용, 필수 고정)
> 아래 항목은 구현 난이도/실패율을 좌우하므로 이번 버전에서 고정한다.

### 좌표계/단위(전 파이프라인 공통)
- **단위**: meter (m), Newton (N), Pascal (Pa)
- **좌표계**:
  - `+Y`: span 방향 (Root → Tip)
  - `+X`: chord 방향(LE → TE)
  - `+Z`: thickness 방향(아래 → 위)
- **Root 위치 규칙**: Root 단면은 `y=0` 평면에 위치해야 한다.
- **Upper/Lower 정의**: Upper는 바깥 표면 중 “법선의 z성분이 양수인 면(또는 그 근방)”이다.

### FEM 해석/학습/시각화의 “노드 집합” 정의
- **FEM은 반드시 볼륨 테트라 메쉬(C3D4)로 수행**한다.
- **시각화/학습/평가는 ‘외피(외부 표면) 노드’만 사용**한다.
  - 이유: 표면 노드에는 법선(normal)을 안정적으로 정의할 수 있고, 웹에서 보는 결과도 표면이 핵심이다.
  - 구현 규칙: 볼륨 메쉬의 “외부 삼각형(face)”을 추출하여 표면 PolyData를 만들고, 그 표면 노드는 볼륨 노드의 부분집합이 되도록 구성한다(가능하면 동일 Node ID 매핑).

### 하중/경계조건 정의(압력의 구현 방식 고정)
- **Upper surface 압력(균일 pressure)은 “등가 절점 하중(Equivalent Nodal Force)”로 변환하여 `*CLOAD`로 적용**한다.
  - 이유: STL 기반 표면 태깅 + CalculiX surface load 문법 차이로 인한 실패를 원천 차단한다.
  - 구현 규칙:
    - Upper boundary face(삼각형)마다 면적 \(A_f\), 단위 법선 \(\hat{n}\) 계산
    - 압력 \(p\)에 대해 face에 작용하는 힘 \(\vec{F}_f = p \cdot A_f \cdot (-\hat{n})\)
    - \(\vec{F}_f\)를 face의 3개 절점에 1/3씩 분배(또는 면적가중 barycentric 분배)하여 node별 힘을 누적
    - 누적된 node 힘을 `*CLOAD`로 입력

### GLB 결과의 색상 보존(컬러 매핑 방식 고정)
- `pyvista.plot()`의 “보이는 스타일”을 그대로 GLB로 보존하는 것은 보장하지 않는다.
- 대신 **버텍스 컬러(Per-vertex RGB)를 직접 생성해 GLB로 내보내는 것을 표준**으로 한다.
  - 구현 규칙: `stress` 스칼라를 colormap으로 `uint8 RGB`로 변환 → mesh의 `point_data["RGB"]`에 저장 → GLB export 시 이 RGB를 사용
  - 필요 시 `pyvista` export가 RGB를 누락하면 `trimesh`로 변환 후 GLB export를 폴백으로 사용한다.

---

## 범위(이번 버전에서 반드시 포함)
### 포함(필수)
- **형상 생성**
  - Blender API(`bpy`)로 파라메트릭 날개 메쉬 생성
  - **해석 파이프라인용 표면 메쉬**: `.stl` 내보내기 (Gmsh 볼륨 메싱 입력)
- **시각화 및 후처리 (Updated with PyVista)**
  - **PyVista**를 활용하여 메쉬 데이터 핸들링 및 시각화 파일 생성
  - 형상 확인용 및 해석 결과(Stress Contour) 매핑된 `.glb` 생성
- **웹 데모**
  - Python + [Gradio](https://www.gradio.app/)로 3D 미리보기/다운로드 제공
  - 슬라이더로 파라미터 변경 → 모델 재생성 → PyVista 후처리 → `Model3D`로 표시
- **데이터 파이프라인 설계 문서화**
  - 데이터 생성/메싱/FEM/학습/검증/실패처리 정책을 스펙으로 명시

### 포함(이번 버전에서 구현)
- **Gmsh 볼륨 메싱 자동화**: STL 입력 → 테트라 볼륨 메쉬 생성 + 경계(face)/노드 셋 생성
- **CalculiX 해석 자동화**: `.inp` 생성 → `ccx` 실행 → 결과 산출/파싱/검증
- **Baseline GNN 학습/추론**: 표면 노드 기준 `stress` 예측 모델(GraphSAGE baseline) 학습/추론
- **FEM vs AI 비교 데모**: Side-by-Side + Error map + Metrics(케이스 단위)

---

## 파라미터 스펙(Geometry)
### 입력 파라미터(슬라이더/실험 변수)
- **Span (m)**: 1.0 ~ 2.0
- **Chord (m)**: 0.2 ~ 0.5
- **Sweep Angle (deg)**: 0 ~ 30
- **Thickness Ratio (t/c)**: 0.05 ~ 0.15

### 파생 규칙(기본)
- Root 기준으로 Chord를 적용하고, Tip도 동일 chord(단순 baseline)로 시작
- Sweep 적용: Tip 섹션의 x를 \( \Delta x = \tan(\text{sweep}) \times \text{span} \) 만큼 이동
- Airfoil은 **대칭 NACA 00xx**(baseline)로 생성:
  - \(t = (t/c)\)
  - 두께 분포(간단 근사)로 상/하면 좌표 생성 후 폐곡선으로 연결

---

## 파일/데이터 포맷 스펙
### 1) 형상 산출물
- `data/raw/geometry/{case_id}/wing.stl` (**필수**, Gmsh 볼륨 메싱 입력용 표면 메쉬)
- `data/raw/geometry/{case_id}/params.json` (재현성)
- `data/raw/geometry/{case_id}/build_report.json` (형상 생성 단계 로그/검증 결과)

### 1-1) 시각화용 산출물 (PyVista)
- `data/raw/geometry/{case_id}/wing_viz.glb`:
  - 단순 형상 확인용 (PyVista로 STL 로드 후 GLB export)
- `data/raw/geometry/{case_id}/wing_result.glb` (해석/예측 후):
  - **Scalar Color 적용**: 노드별 von Mises Stress 값을 Colormap(Jet/Viridis)으로 매핑
  - **표준은 버텍스 컬러 기반(RGB) export**이며, “plot 스타일 유지”는 목표로만 둔다.

### 1-2) 해석용 메쉬 산출물(권장)
- `data/raw/mesh/{case_id}/wing.msh` (**필수**, Gmsh tetra volume mesh)
- `data/raw/mesh/{case_id}/mesh_report.json` (요소 수/품질/실패 사유)
- `data/raw/mesh/{case_id}/boundary_sets.json` (Root/Upper/Surface 노드/face 셋 정의 및 통계)

### 1-3) FEM 산출물
- `data/raw/fem/{case_id}/{case_id}.inp` (CalculiX 입력)
- `data/raw/fem/{case_id}/{case_id}.frd` (CalculiX 결과)
- `data/raw/fem/{case_id}/fem_report.json` (실행 로그/시간/검증/실패 사유)
- `data/raw/fem/{case_id}/surface_results.npz`
  - `node_id`(표면 노드), `pos`(표면 좌표), `stress_vm`(von Mises), `disp`(dx,dy,dz)

### 2) 케이스 메타데이터(권장)
- `data/raw/geometry/params.csv`
  - `case_id, seed, span_m, chord_m, sweep_deg, thickness_ratio, status, failure_reason`
- `data/raw/manifest.json`
  - 버전/의존성/실행환경(gmsh, ccx, blender) 및 전체 케이스 인덱스

---

## Blender API 모델 생성 스펙
### 실행 방식(표준: A안 — Blender GUI에서 직접 실행/검수)
- Blender를 **직접 띄워서(GUI)** 생성 과정을 눈으로 확인한다.
- 실행 절차:
  - Blender → **Scripting** 탭 → **Text Editor**
  - 생성 스크립트(파라미터 포함)를 붙여넣기
  - **Run Script**로 메쉬 생성
  - 뷰포트에서 형상/메쉬 품질을 검토한 뒤 `.stl`(해석용)로 Export
- 장점:
  - 파라미터 변경 시 즉시 결과를 눈으로 확인 가능
  - 해석 실패(비매니폴드/교차/노멀 뒤집힘 등) 원인을 조기에 발견 가능

### Blender에서 반드시 체크할 검토 포인트(해석 성공률에 직결)
- **스케일/단위**: span 1~2m가 실제 크기감으로 보이는지(meter 기준)
- **폐곡면(watertight)**: 구멍/틈/겹침이 없는지(볼륨 메싱 실패 원인)
- **노멀 방향**: 면 노멀이 일관적인지(상/하면 식별 및 표면 셋 생성 안정성)
- **자기 교차(Self-intersection)**: sweep/두께가 커질 때 프로파일이 겹치지 않는지
- **Root/Upper surface 식별 가능성**: Root 단면 명확성, Upper surface 선택 안정성

### 모델 생성 요구사항
- **단위는 meter** 기준(해석/실제 스케일 호환)
- **매니폴드(가능한 한 watertight)** 메쉬 생성(후속 볼륨 메싱에 유리)
- 삼각형/사각형 혼합 가능(내보내기 시 자동 삼각화 가능)
- **좌표계 고정**: Root가 `y=0`에 오고, span은 `+Y`로 향하도록 최종 오브젝트를 정렬(apply transforms)
- 실패 시:
  - 오류 메시지/스택을 stdout로 남기고 non-zero exit
  - caller가 `failure_reason` 기록 가능

---

## Gradio 웹 데모 스펙
### UI 요구사항
- 슬라이더: span, chord, sweep, thickness_ratio
- 버튼: `Generate`
- 출력:
  - `Model3D`: PyVista로 변환된 `.glb` 미리보기 (초기엔 형상, 추후엔 Stress Contour 포함)
  - `File`: **해석용 `.stl` 다운로드(기본)**
  - `Textbox`: 생성 로그(Blender 사용 여부, 소요시간, 에러)

> v0(Week 1) 구현 메모(뷰어 모드):
> - 이번 단계에서는 `Generate`(형상 생성)는 Gradio에서 직접 수행하지 않고, **Blender 배치 스크립트로 사전 생성(pre-computed)** 한다.
> - Gradio는 `data/raw/geometry/params.csv(status=success)`에서 케이스 목록을 로드하여, 선택된 `case_id`의 `wing_viz.glb`/`wing.stl`/로그를 확인하는 **뷰어**로 동작한다.

### 동작 규칙 (PyVista 기반)
- **Geometry Pipeline**:
  1. Blender 스크립트로 `.stl` 생성
  2. Python(Gradio 백엔드)에서 `pyvista.read('wing.stl')`
  3. `plotter.export_gltf('wing_viz.glb')` 로 변환
  4. 프론트엔드(`Model3D`)에 `wing_viz.glb` 전달
- **Result Pipeline (이번 버전 포함)**:
  1. `wing.stl` → Gmsh 볼륨 메싱(`wing.msh`)
  2. 경계 셋 생성(Root/Upper/Surface)
  3. CalculiX `.inp` 생성(등가 절점 하중 방식)
  4. `ccx` 실행 후 `.frd` 산출
  5. 결과에서 표면 노드 응력/변위 추출(`surface_results.npz`)
  6. 표면 mesh에 스칼라(`stress_vm`) 할당 → 버텍스 컬러 생성 → `wing_result.glb` 생성
  7. (학습 완료 후) GNN 추론 결과도 동일 방식으로 `wing_pred.glb`, `wing_error.glb` 생성 가능

---

## FEM 시뮬레이션 자동화 스펙(이번 버전 필수)
### 1) 솔버 및 실행 환경
- **Solver**: CalculiX (`ccx` 2.15 이상)
- **Wrapper**: Python `subprocess` 모듈로 실행 제어
- **실행 커맨드**: `ccx -i {case_id} -n {cpu_count}` (멀티코어 활용)

### 2) 해석 물성치 (Material Properties)
- **표준 소재**: 알루미늄 합금 (Aluminium 6061-T6)
- **Elastic Modulus (E)**: 69 GPa (69e9 Pa)
- **Poisson's Ratio (ν)**: 0.33
- **Density**: 2700 kg/m³ (필요 시)

### 3) 메싱/경계 셋 생성 규칙(Gmsh 기준, STL 입력)
> STL에서 “Upper/Root”를 안정적으로 뽑기 위한 최소 규칙.

- **경계 face 법선의 방향(Outward) 고정 규칙**:
  - STL/메싱 과정에서 삼각형 정점 순서가 뒤집히면 법선 방향이 반대로 나올 수 있다.
  - 구현 규칙:
    - 볼륨 메쉬의 대략적 중심 `C_vol`(예: 모든 노드 좌표 평균)을 계산
    - 경계 face 중심 `C_f`와 face 법선 `n`에 대해 `dot(n, C_f - C_vol) < 0`이면 `n = -n`으로 뒤집어 **항상 바깥 방향(outward)**으로 맞춘다.

- **Root 노드 셋(`NROOT`)**:
  - 조건: `y <= y_tol` (기본 `y_tol = 1e-4 m`)
  - 기대: Root 단면이 `y=0` 평면에 정확히 정렬되어 있어야 함
- **외피 표면(face) 추출(`SURF_ALL`)**:
  - 볼륨 테트라 메쉬에서 경계 face를 추출하여 삼각형 표면 집합을 만든다.
- **Upper face 셋(`SURF_UPPER`)**:
  - 조건(기본): 경계 face의 단위 법선 \(\hat{n}\)에 대해 `n_z >= nz_min` (기본 `nz_min = +0.2`)
  - 추가 필터: Root 단면 부근 face(`y <= y_tol*5`)는 제외(경계조건 영역 혼입 방지)
  - 실패 감지: `SURF_UPPER`의 총 면적이 전체 외피 면적의 20% 미만이면 “Upper 분리 실패”로 처리(해당 케이스 fail)
  - 폴백(선택): `SURF_UPPER` 면적이 부족하면 `nz_min`을 `+0.1`로 1회 완화 후 재시도(그래도 실패 시 fail)

### 4) CalculiX 입력 파일(.inp) 템플릿 구조(등가 절점 하중)
```inp
*HEADING
Model: Parametric Wing FEM
*NODE
... (Gmsh에서 파싱한 노드 좌표)
*ELEMENT, TYPE=C3D4, ELSET=Eall
... (Gmsh에서 파싱한 엘리먼트 연결성)
*MATERIAL, NAME=AL6061
*ELASTIC
69e9, 0.33
*SOLID SECTION, ELSET=Eall, MATERIAL=AL6061
*BOUNDARY
Nroot, 1, 3, 0  (Root 노드 그룹의 x,y,z 변위를 0으로 고정)
*CLOAD
... (Upper 표면에 대한 등가 절점 하중: node_id, dof(1/2/3), value)
*STEP
*STATIC
*NODE FILE
U, S  (변위 U, 응력 S를 .frd 파일로 출력)
*END STEP
```

### 5) 결과 파싱 및 저장
- **대상 파일**: `{case_id}.frd` (Binary/Ascii)
- **파싱 원칙(안정성 우선)**:
  - 1차: `ccx2paraview`로 `.vtk/.vtu` 변환 후 PyVista로 로드
  - 2차(폴백): `.frd` 텍스트 파서(정규식)로 U(변위), S(응력) 추출
- **저장 데이터**:
  - 표면 노드 기준 `von Mises Stress`, `Displacement Vector (dx, dy, dz)`
  - `.npz` 저장을 표준으로 한다(가벼움 + 재현성)

### 6) 실패 감지 및 재시도 전략
1. **Meshing 단계**:
   - `gmsh` exit code 확인. 실패 시 `Element Size Factor`를 1.2배 키워서 재시도 (최대 3회).
   - 경계 셋 생성 실패(Upper 분리 실패/Root 노드 0개 등) 시: 케이스 fail로 기록
   - 품질 가드레일(권장):
     - 최대 노드 수/요소 수 상한을 두고(예: nodes <= 300k, tets <= 1.5M) 초과 시 메쉬 파라미터를 키워 재시도
     - 요소 품질이 기준 이하(예: min quality < threshold)면 fail 또는 재시도
2. **Solving 단계**:
   - `ccx` 로그에서 `*ERROR` 키워드 감지.
   - 수렴/수치 불안정 시: 압력(p)을 1/10로 줄여서 1회 재시도(그래도 실패 시 fail)
   - 결과 검증: 최대 변위/응력 값이 비현실적(예: `nan/inf` 또는 상식적 상한 초과)이면 fail 처리

### 7) 해석 결과 시각화 (Quality Check)
GNN 학습 전, 구축된 데이터셋의 건전성을 검증하기 위해 개별 FEM 결과를 시각화한다.
- **자동 후처리**: 해석 완료 직후 `wing_result.glb` 자동 생성.
- **Gradio FEM 뷰어**:
  - GNN 예측 없이 **FEM 결과만 단독으로 로드**하여 Stress/Displacement 분포 확인.
  - **Deformation(변형) 적용**: 변위 벡터(Displacement)에 스케일 팩터(예: x10)를 곱해 **날개가 실제로 휘어지는 형상**을 시각적으로 과장하여 표현 (물리적 타당성 검증 용도).

---

## GNN Surrogate 모델 스펙(이번 버전 포함, 표면 노드 기준)
### 1) 데이터셋 구조 (PyTorch Geometric)
  하나의 데이터 샘플(`Data` 객체)은 다음 필드를 가짐:
  - **`x` (Node Features)**: `[N, 10]` (6 + 4)
    - `pos_x, pos_y, pos_z` (노드 좌표, 정규화됨)
    - `norm_x, norm_y, norm_z` (표면 노드 법선 벡터, 형상 정보 핵심)
    - **Global Params**: `[span, chord, sweep, thickness]`를 모든 노드에 동일하게 Concatenate (전역 형상 정보 주입)
  - **`edge_index` (Connectivity)**: `[2, E]`
    - 표면 삼각형(mesh) 인접성 기반 edge (표면 그래프)
  - **`y` (Target)**: `[N, 1]`
    - Nodal von Mises Stress (Log-scale 변환 권장)
    - **전처리(Outlier 제거)**: Root 부근(`y < 0.05 * span`)의 노드는 **응력 특이점(Singularity)** 가능성이 높으므로 학습 Loss 계산 시 제외(Masking)하거나, 상한값(Clipping)을 적용한다.
    - **권장 구현(명확화)**:
      - `loss_mask: [N] bool`을 만들어 loss 계산에만 적용한다.
      - 지표(MAE/RMSE/Max)는 `all_nodes`와 `masked_nodes`를 모두 계산하여 보고한다(성능 “좋아 보이는 착시” 방지).
  - **`pos`**: `[N, 3]` (3D 좌표, Message Passing 거리 계산용)

### 2) 모델 아키텍처 (Baseline: PointNet++ or GraphSAGE)
초기 모델은 **GraphSAGE** 변형을 사용하여 Inductive Learning(새로운 형상 예측)을 목표로 함.
- **Encoder**: MLP(`[10, 64, 128]`) -> 각 노드 Feature 임베딩
- **GNN Layers**: 3~4 Layers of `SAGEConv` or `GATConv`
  - Activation: `ReLU`
  - Skip Connection: ResNet 스타일 (Vanishing Gradient 방지)
- **Decoder**: MLP(`[128, 64, 1]`) -> 각 노드의 Stress 값 예측

### 3) 학습 전략
- **Loss Function**:
  - Main: `MSELoss` (Log-scale stress 기준)
  - Auxiliary(선택): `L1Loss` (이상치에 덜 민감하도록)
- **Optimizer**: `AdamW`, LR=1e-3, Weight Decay=1e-4
- **Scheduler**: `ReduceLROnPlateau` (Patience=10)
- **Batch Size**: 1 (메쉬 크기가 크므로 1 graph per batch) 또는 Sub-sampling

### 4) Gradio 통합 평가 및 시각화 (최종 검증)
모델 성능을 직관적으로 검증하기 위해 웹 데모에 다음 기능을 통합한다.
- **Side-by-Side 비교 뷰**:
  - **좌측 (Ground Truth)**: FEM 해석으로 계산된 von Mises Stress
  - **우측 (Prediction)**: GNN 모델이 예측한 Stress
  - 두 모델의 컬러바(Colorbar) 범위를 동일하게 고정(Normalize)하여 색상만으로 차이를 비교 가능하게 함.
- **오차 히트맵 (Error Map)**:
  - `|FEM - AI|` (절대 오차)를 계산하여 3D 메쉬 위에 매핑.
  - 예측이 부정확한 부위(예: 급격한 곡률 부위, 접합부)를 붉은색으로 강조.
- **성능 지표 대시보드 (Metrics)**:
  - **정량 지표**: 해당 케이스에 대한 MAE, RMSE, Max Error를 텍스트로 표시.
  - **속도 비교**: FEM Solver 소요 시간(수 분) vs GNN Inference 시간(수 밀리초)을 바 차트나 텍스트로 강조.

---

## 검증(Validation) — “결과가 잘 나오는지” 확인용 고정 체크리스트
> 구현이 완료되면 아래 3개 기준 케이스를 반드시 돌려서, “물리적으로 그럴듯한지”를 빠르게 판정한다.

### 공통: 입력/출력 산출물 체크
- **입력 파라미터 저장**: `params.json`이 존재하고, 값이 슬라이더 입력과 일치한다.
- **형상 체크**: `wing.stl`이 watertight(또는 최소한 볼륨 메싱 가능)하며, Root가 `y=0`에 정렬되어 있다.
- **경계 셋 체크**: `boundary_sets.json`에서
  - `NROOT` 노드 수가 0이 아니다.
  - `SURF_UPPER` 총 면적이 전체 외피의 20% 이상이다(또는 폴백 1회 후에도 실패 시 케이스 fail).
- **FEM 결과 체크**: `surface_results.npz`에 `nan/inf`가 없고, `pos`/`disp`/`stress_vm`의 shape가 일관적이다.
- **시각화 체크**: `wing_result.glb`에서 stress 컬러가 “항상” 보인다(버텍스 컬러 기반).

### 검증 케이스 1 (쉬움 / Stable)
- **Params**
  - span=1.2, chord=0.35, sweep=0, thickness_ratio=0.10
- **기대 현상(정성)**
  - 변형(Deformation): Upper 압력이 아래 방향(-Z) 성분으로 작용하므로, 날개 외피는 전체적으로 **-Z 방향으로 처짐**이 나타난다.
  - 응력(Stress): Root 고정으로 인해 Root 근방에서 응력 집중이 크고, span 방향으로 갈수록 감소 경향을 보인다.
- **실패 판정(자동/정성)**
  - 처짐의 평균 Z 변위가 0에 가깝거나(+Z로 뒤집힘 포함) 방향이 이상하면 fail(하중 방향/법선/부호 오류 가능성).
  - Root가 아닌 임의의 중간 구간에서만 비정상적으로 응력이 폭발하면 fail(메시 품질/결과 파싱/단위 오류 가능성).

### 검증 케이스 2 (중간 / Sweep + 얇음)
- **Params**
  - span=1.6, chord=0.30, sweep=20, thickness_ratio=0.07
- **기대 현상(정성)**
  - 변형: sweep으로 인해 단순 처짐 외에 약간의 비대칭(비틀림처럼 보이는) 성분이 관찰될 수 있다.
  - 응력: Root 근방 집중 + sweep이 큰 쪽(LE/TE 방향)에서 분포가 약간 비대칭이 될 수 있다.
- **실패 판정**
  - `SURF_UPPER` 분리가 자주 실패하면 fail(Upper 분류 규칙이 해당 형상군에서 약함 → `nz_min`/필터 재검토).
  - 결과가 케이스 1과 “거의 동일한 패턴/스케일”로 나오면 fail(전역 파라미터/형상 변화가 실제로 반영되지 않았을 가능성).

### 검증 케이스 3 (어려움 / 가장 실패가 잦은 영역)
- **Params**
  - span=2.0, chord=0.22, sweep=30, thickness_ratio=0.05
- **기대 현상(정성)**
  - 메싱/해석 실패 가능성이 높다(얇고 sweep 큰 형상은 self-intersection, poor quality mesh 위험).
  - 성공 시: 처짐은 더 크게 나타나는 경향(얇아서) + Root 집중은 유지.
- **실패 판정**
  - 이 케이스는 실패해도 괜찮지만, 반드시 `failure_reason`이 구조적으로 기록되어야 한다(“왜 실패하는지”가 스펙 준수의 핵심).

### 추가: 학습/평가 지표 보고 규칙(착시 방지)
- Root 특이점 때문에 `loss_mask`를 쓰는 경우:
  - **Metrics는 2종을 모두 보고**한다: `all_nodes`, `masked_nodes`
  - 시각화(Error map)도 2종(전체/마스크 적용)을 옵션으로 제공하는 것을 권장한다.

---

## 2~3주 로드맵(실행 단위)
### Week 1
- 파라미터 샘플링/형상 생성 자동화(Blender)
- **PyVista 기반 시각화 파이프라인 구축 (STL → GLB 변환)**
- 볼륨 메싱(Gmsh) 스펙 확정 및 최소 파이프라인

### Week 2
- CalculiX 자동화로 정답 데이터 누적(최소 200 solved 목표)
- **PyVista로 FEM 결과(Stress) 매핑 및 시각화 검증**
- 그래프 데이터셋 빌드/정규화/스플릿

### Week 3(선택)
- GNN 학습/추론/시각화 데모 고도화
- “FEM vs AI” side-by-side, error map, 속도 비교(문구/표) 완성
- **Hugging Face Spaces 배포 (Docker)**:
  - `Dockerfile` 작성: `gmsh`, `ccx`, `xvfb` 설치 포함
  - Spaces 설정 및 CI/CD(Github Action 또는 자동 빌드) 연동
