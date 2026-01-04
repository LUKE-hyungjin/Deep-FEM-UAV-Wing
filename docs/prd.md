# PRD — AI 기반 UAV 날개 구조 해석 예측 모델 (Deep-FEM-UAV-Wing)

## 0) 용어/정의(Glossary)
- **Case**: 하나의 파라미터 조합으로 생성된 단일 실험 단위(`case_id`로 식별)
- **Surface Nodes**: 볼륨 테트라 메쉬의 “외피(외부 표면)”에 속하는 노드 집합(이번 버전의 학습/시각화/평가 기준)
- **Boundary Faces**: 볼륨 메쉬에서 외부에 노출된 삼각형 면(외피)
- **NROOT**: Root 고정에 쓰이는 노드 셋(`y <= y_tol`)
- **SURF_UPPER**: Upper 압력 하중 적용을 위한 boundary face 셋(`n_z >= nz_min`, Root 근방 제외)
- **Equivalent Nodal Load**: 면 압력(p)을 삼각형 면적/법선 기반으로 노드 힘으로 환산하여 `*CLOAD`로 적용하는 방식
- **Unified Colorbar**: FEM/AI를 같은 min/max로 정규화하여 동일 색상 스케일로 비교하는 규칙

## 1) 배경/목표
### 배경
- FEM 해석은 정확하지만 느리다. 반복 설계(파라미터 스윕/최적화)에서 병목이 된다.
- 목표는 “파라메트릭 날개”에 대해 **FEM 기반 GT 데이터 생성 → GNN surrogate 학습 → 웹에서 FEM vs AI를 한 화면에 비교**하는 End-to-End 파이프라인을 재현 가능하게 만드는 것이다.

### 목표(Outcome)
- 단일 케이스 기준:
  - 입력 파라미터(span/chord/sweep/thickness_ratio)로 형상 생성
  - Gmsh로 볼륨 메싱
  - CalculiX로 선형 정적 해석 실행
  - 표면 노드에 대해 von Mises stress + displacement를 산출
  - 동일 표면 메쉬에서 FEM/AI/오차를 GLB로 생성하여 웹에서 시각 비교
- 데이터셋 기준:
  - 최소 **200 solved 케이스**를 누적(실패는 실패 사유 기록)
  - Baseline GNN(GraphSAGE) 학습/추론이 “동일 파라미터 분포”에서 동작

### Non-Goals(이번 버전 제외)
- 비선형/좌굴/피로/복합재 등 고급 해석
- 고정익 공력 하중(실제 CFD/분포 하중) 연동
- 대규모 분산 학습/서빙 최적화

---

## 2) 사용자/사용 시나리오
### 사용자
- 1인 개발자(본인): 연구/포트폴리오 목적, 로컬에서 재현 가능한 파이프라인 구축

### 핵심 시나리오(User Stories)
- **US1**: 데모 페이지 접속 시 **미리 계산된 샘플(Pre-computed)**이 즉시 로드되어, 사용자가 실행 전에 결과를 먼저 볼 수 있다.
- **US2**: 슬라이더로 파라미터를 입력하고 `Generate`를 누르면 형상이 3D로 보인다.
- **US3**: `Run FEM` 실행 시, **단계별 진행 상황(형상 → 메싱 → 해석 → 후처리)**이 시각적으로 표시되어 기다림의 지루함을 줄인다.
- **US4**: 학습된 모델이 있으면 `Run AI`로 예측 결과가 3D로 보이고, FEM과 나란히 비교된다.
- **US5**: `Error Map`을 선택하면 |FEM-AI|가 3D로 시각화되고, MAE/RMSE/MaxError가 표시된다.
- **US6**: 생성된 산출물(STL/GLB/NPZ/로그)을 다운로드할 수 있다.

---

## 2-1) 제품 원칙(Product Principles)
- **재현성 우선**: 동일 파라미터는 동일 `case_id` 및 동일 산출물(캐시)로 귀결되어야 한다.
- **실패를 숨기지 않기**: 실패 케이스도 “왜 실패했는지”가 산출물/로그로 남아야 한다.
- **데모 UX는 기다림을 관리한다**: 진행률/단계/시간/캐시 힌트가 반드시 보여야 한다.
- **비교는 공정해야 한다**: FEM/AI는 Unified Colorbar로 정규화하고, 지표는 `all_nodes`/`masked_nodes`를 함께 보고한다.
- **SI 단위계 엄수**: 모든 물리량은 `meter(m)`, `Pascal(Pa)`, `Newton(N)`을 사용하며, 코드 내부에서 임의 변환(mm 등)을 금지한다.

---

## 3) 기능 요구사항(Functional Requirements)
### FR1. Geometry 생성
- **기술 스택**: Blender Python API (`bpy`)
  - 이유: 강력한 Mesh Boolean 연산 및 Python 스크립팅 지원
- 입력: span/chord/sweep/thickness_ratio
- 출력:
  - `wing.stl`
  - `params.json`
  - `build_report.json`(유효성 검사: watertight/단위/좌표계/노멀)
- 좌표계/단위 고정:
  - meter(m), Pascal(Pa), Newton(N)
  - Root는 `y=0`, span은 `+Y`, chord는 `+X`, thickness는 `+Z`

### FR2. Meshing(Gmsh)
- 입력: `wing.stl`
- 출력:
  - `wing.msh`(테트라 볼륨 메쉬)
  - `mesh_report.json`(요소 수/품질/시간/실패 사유)
  - `boundary_sets.json`
    - `NROOT`: y<=y_tol 노드
    - `SURF_ALL`: 외피 face
    - `SURF_UPPER`: n_z>=nz_min 외피 face(+root 근방 제외)
 - 법선 방향(outward) 보정:
   - `dot(n, C_f - C_vol) < 0`이면 `n=-n`
 - 폴백:
   - `SURF_UPPER` 면적이 부족하면 `nz_min`을 1회 완화 후 재시도(그래도 실패 시 fail)
 - 메쉬 가드레일(권장):
   - 요소/노드 수 상한 초과 시 더 거친 메쉬로 재시도
   - 최소 품질 기준 미달 시 fail 또는 재시도

### FR3. FEM(CalculiX) 자동화
- 입력: `wing.msh`, `boundary_sets.json`, 물성(E, nu), 압력(p)
- 출력:
  - `{case_id}.inp`, `{case_id}.frd`
  - `surface_results.npz`(표면 노드 기준 pos/stress_vm/disp)
  - `fem_report.json`(실행 시간/로그/검증/재시도 기록)
- 하중 구현:
  - 압력은 **등가 절점 하중으로 변환하여 `*CLOAD`로 적용**
- 검증:
  - 결과에 `nan/inf`가 없어야 함
  - `SURF_UPPER` 총 면적이 전체의 20% 미만이면 케이스 fail
 - 결과 파싱:
   - 1차: `ccx2paraview`로 변환 후 PyVista로 로드
   - 2차: `.frd` 텍스트 파서 폴백

### FR4. Visualization(GLB)
- 입력: 표면 메쉬 + 스칼라(stress_vm / error) + 변위(선택)
- 출력:
  - `wing_viz.glb`(형상)
  - `wing_result.glb`(FEM stress)
  - `wing_pred.glb`(AI stress, 모델 있을 때)
  - `wing_error.glb`(abs error)
- 색상:
  - **버텍스 컬러(RGB) 기반으로 고정**
  - **Unified Colorbar**: FEM vs AI 비교 시, 두 데이터의 Global Min/Max를 찾아 **동일한 스케일로 정규화(Sync)**하여 색상 왜곡을 방지한다.
 - 변형(Deformation) 표시:
   - disp 벡터에 `deform_scale`을 곱해 과장 표현을 허용한다(품질 검증용)

### FR5. Dataset/Training
- 데이터셋:
  - 샘플 단위: 케이스 1개(표면 노드 그래프)
  - input: pos+normal(6) + global params(4) = **10D**, edge_index(표면 인접), target: stress_vm(log-scale 권장)
  - Root 부근 특이점 대응: 학습 loss는 `loss_mask`로 제외/클리핑을 허용하되, 지표는 `all_nodes`/`masked_nodes`를 함께 보고
- 모델:
  - GraphSAGE(3~4 layer) baseline
- 산출물:
  - 체크포인트/학습 로그/스플릿 정보(재현 가능)
 - 지표 보고:
   - 케이스 단위: MAE/RMSE/Max (all_nodes, masked_nodes)
   - 시각화: error map도 전체/마스크 버전을 옵션으로 제공(권장)

### FR6. Gradio Web Demo
- UI:
  - 슬라이더: span/chord/sweep/thickness_ratio
  - 버튼: Generate, Run FEM, Run AI(모델 있을 때)
  - 출력:
    - Model3D: Geometry / FEM / AI / Error
    - File: STL/GLB/NPZ 다운로드
    - Textbox: 로그/실행 시간/실패 사유
- v0(Week 1) 구현 메모(뷰어 모드):
  - 이번 단계에서는 “생성(Generate)”은 **Blender 배치 스크립트**로 수행하고, Gradio는 **pre-computed 결과를 선택/확인하는 뷰어**로 동작한다.
  - 케이스 목록은 `data/raw/geometry/params.csv(status=success)`에서 로드하며, 선택한 `case_id`의 `wing_viz.glb`(3D 미리보기) + `wing.stl`(다운로드) + `build_report.json` 로그를 표시한다.
- 캐시:
  - 동일 파라미터는 동일 `case_id`로 캐시 재사용
 - 진행 상태:
   - 단계 상태를 최소 다음으로 표준화한다: `IDLE → GEOMETRY → MESHING → FEM → POSTPROCESS → DONE/FAILED`
   - 각 단계에서 `elapsed_ms`, `message`, `artifacts`를 UI에 표시한다.

---

## 4) 비기능 요구사항(Non-Functional Requirements)
### 재현성
- 모든 케이스는 `params.json` + tool 버전(gmsh/ccx/blender)을 `manifest.json`에 기록
 - `case_id` 규칙:
   - 파라미터 + 파이프라인 버전(결정사항/임계값 포함)의 해시를 권장
   - 파라미터 반올림 규칙을 고정(예: sweep_deg는 0.1 deg 단위, length는 1e-3 m 단위)

### 안정성/실패 처리
- 모든 단계는 `status, failure_reason, stdout/stderr 요약`을 남긴다.
- 실패는 조용히 넘어가지 않고, UI/로그에 명확히 표시한다.
 - 실패 사유는 가능한 한 “액션 가능”하게 분류한다(예: watertight 실패, upper 분리 실패, gmsh 실패, ccx 실패, 파싱 실패, 컬러 export 실패).

### 배포 및 환경(Deployment)
- **타겟 플랫폼**: Hugging Face Spaces (CPU Basic)
- **배포 방식**: Docker Container (Dockerfile 기반)
- **필수 패키지**: `gmsh`, `calculix-ccx`, `xvfb`(Headless 3D 렌더링용)
- **리소스 제약**: GPU 사용 없음(CPU Only), 메모리 16GB 내외 최적화
 - 배포 전략:
   - 기본 UX는 **pre-computed 샘플을 즉시 로드**(US1)하여 “항상 볼 것”을 보장한다.
   - 실시간 FEM은 로컬 기준이 되며, Spaces에서는 타임아웃/리소스 상황에 따라 제한될 수 있음을 UI에 명시한다.

### 성능(현실적인 목표)
- 로컬 기준:
  - Geometry+Viz: 수 초 이내
  - Meshing+FEM: 수십 초~수 분(케이스에 따라)
- 웹 데모는 장시간 작업을 고려해 “캐시 재사용”을 최우선으로 한다.

---

## 4-1) 데이터/산출물 스키마(구현 가이드)
> 실제 구현 시 “문서와 파일이 다르게 나오는” 문제를 막기 위해 PRD에서 최소 스키마를 고정한다.

### `params.json`
- 필수 키:
  - `case_id`, `span_m`, `chord_m`, `sweep_deg`, `thickness_ratio`
  - `created_at`, `pipeline_version`

### `boundary_sets.json`
- 필수 키:
  - `y_tol`, `nz_min`
  - `nroot_node_ids: int[]`
  - `surf_all_faces: int[][]` (각 face는 3개 node_id)
  - `surf_upper_faces: int[][]`
  - `stats`: `{ nroot_count, surf_all_area, surf_upper_area, surf_upper_ratio }`

### `surface_results.npz`
- 필수 배열:
  - `node_id: [N] int`
  - `pos: [N,3] float`
  - `normal: [N,3] float`
  - `stress_vm: [N] float`
  - `disp: [N,3] float`
  - `loss_mask: [N] bool` (Root 특이점 대응)

### `*_report.json`(build/mesh/fem)
- 공통 필수 키:
  - `status: success|failed`
  - `failure_reason` (failed일 때)
  - `elapsed_ms`
  - `stdout_tail`, `stderr_tail` (필요 시)
  - `artifacts` (생성된 파일 목록)

---

## 5) 성공 기준(Success Metrics)
- **E2E 1케이스 성공**: Generate→FEM→Result GLB가 사람 눈으로 정상(컬러/변형) 확인 가능
- **데이터셋 누적**: 200 solved 케이스 + 실패 사유가 구조적으로 기록됨
- **GNN baseline**: 학습이 수렴하고, 임의 케이스에서 예측 GLB 및 error map 생성 가능
- **데모 품질**: FEM vs AI side-by-side, metrics(최소 MAE/RMSE/Max) 표시

---

## 6) 리스크 & 대응
- **R1: STL 기반 Upper/Root 태깅 불안정**
  - 대응: 좌표계 고정 + 법선/면적 기반 규칙 + 실패 감지(면적 비율)로 조기 차단
- **R2: GLB export에서 컬러 누락**
  - 대응: 버텍스 컬러를 직접 생성하고, 필요 시 export 경로를 폴백(trimesh)
- **R3: FEM 시간이 너무 길어 데모 UX가 나쁨**
  - 대응: 캐시/재사용, 단계 분리(Generate vs Run FEM), 로그/진행상태 표시
- **R4: 볼륨 결과와 학습 feature 불일치**
  - 대응: 이번 버전은 표면 노드만 학습/시각화로 고정
 - **R5: Root 특이점으로 인한 학습/지표 왜곡**
   - 대응: `loss_mask` 도입 + 지표는 all_nodes/masked_nodes 동시 보고 + error map도 옵션 분리

---

## 7) 마일스톤(2~3주)
- Week 1: Geometry + STL→GLB(버텍스 컬러 경로 PoC) + 케이스 캐시/산출물 구조 확정
- Week 2: Gmsh 메싱 + 경계 셋 + CalculiX 자동화 + surface_results.npz 생성/검증
- Week 3: GNN 학습/추론 + Gradio side-by-side + error map + metrics 완성

---

## 7-1) 진행 현황 체크리스트(구현 기준)

### 1단계 — Geometry + Viz + Viewer (완료)
- [x] Blender(headless)로 파라메트릭 `wing.stl` 생성(배치 생성 지원)
- [x] `case_id` 기반 캐시/산출물 폴더 구조 고정(`data/raw/geometry/{case_id}/...`)
- [x] `params.json` / `build_report.json` 생성 및 로그 기록
- [x] STL → **바이너리 GLB**(`wing_viz.glb`) 생성
- [x] 인덱스 파일 생성/갱신: `data/raw/geometry/params.csv`, `data/raw/manifest.json`
- [x] Gradio는 **뷰어 모드**: 케이스 선택 → 3D 미리보기(GLB) + STL 다운로드 + 로그 확인
- [x] (환경 이슈 대응) 기존 `wing_viz.glb` 복구 스크립트: `scripts/repair_geometry_glb.py`

### 2단계 — Meshing(Gmsh) (미진행)
- [ ] `wing.stl` → `wing.msh`(테트라 볼륨 메쉬) 생성
- [ ] `boundary_sets.json` 생성: `NROOT`, `SURF_ALL`, `SURF_UPPER` + 면적 통계/검증
- [ ] `mesh_report.json` 생성(시간/품질/실패 사유)
- [ ] (시각 디버그) `surf_sets.glb` 생성(Upper/Root 분리 결과 색상 확인)

### 3단계 — FEM(CalculiX) + Postprocess (미진행)
- [ ] `.inp` 생성(등가 절점 하중 `*CLOAD`)
- [ ] `ccx` 실행 → `.frd` 산출
- [ ] 표면 결과 추출: `surface_results.npz`(pos/normal/stress_vm/disp/loss_mask)
- [ ] 결과 시각화: `wing_result.glb`(버텍스 컬러 기반)

### 4단계 — Dataset 누적/검증 (미진행)
- [ ] 최소 200 solved 케이스 누적(`status=success`), 실패 케이스도 `failure_reason` 구조적으로 기록
- [ ] 품질 검증 체크리스트 자동화: Root 노드 수, Upper 면적 비율, nan/inf 결과 검출, 스케일 sanity check
- [ ] `manifest.json`에 툴 버전(gmsh/ccx/blender) 및 파이프라인 버전/임계값 기록(재현성)

### 5단계 — GNN 학습/추론 (미진행)
- [ ] 그래프 데이터셋 빌드(표면 노드 기준): `x=pos+normal+global_params`, `edge_index`, `y=stress_vm(log-scale 권장)`
- [ ] `loss_mask` 생성/적용(Root 특이점 대응) + 지표는 `all_nodes`/`masked_nodes` 동시 보고
- [ ] GraphSAGE baseline 학습 스크립트 + 체크포인트/스플릿/로그 저장(재현 가능)
- [ ] 단일 케이스 추론 → `wing_pred.glb` 생성(버텍스 컬러) + `wing_error.glb`(abs error) 생성

### 6단계 — Gradio 비교 데모(FEM vs AI) (미진행)
- [ ] Side-by-Side 뷰: FEM(`wing_result.glb`) vs AI(`wing_pred.glb`)
- [ ] Error Map 토글: `|FEM-AI|` 시각화(`wing_error.glb`)
- [ ] Metrics 표시: MAE/RMSE/Max (all_nodes, masked_nodes)
- [ ] Unified Colorbar: FEM/AI 동일 min/max로 정규화(공정 비교)
- [ ] 단계 상태머신 노출: `IDLE → GEOMETRY → MESHING → FEM → POSTPROCESS → DONE/FAILED` + `elapsed_ms/message/artifacts`

### 7단계 — 배포(Hugging Face Spaces) (미진행)
- [ ] Dockerfile 기반 배포 준비(gmsh/ccx/xvfb 포함) + 런타임에서 동작 확인
- [ ] 기본 UX: pre-computed 샘플 즉시 로드(빈 화면 방지) + 실시간 FEM 제한/안내 문구

## 8) 수용 기준(Acceptance Criteria)
- AC1: 파라미터 입력 후 `wing_viz.glb`가 생성되고 Model3D에서 보인다.
- AC2: 동일 파라미터 재실행 시 같은 `case_id`로 캐시를 재사용한다.
- AC3: `Run FEM` 실행 후 `{case_id}.frd`, `surface_results.npz`, `wing_result.glb`가 생성된다.
- AC4: `wing_result.glb`에서 stress 컬러가 “항상” 보인다(버텍스 컬러 기반).
- AC5: 학습된 모델이 있으면 `wing_pred.glb`, `wing_error.glb`가 생성되고 side-by-side로 비교된다.
 - AC6: 데모 최초 진입 시 pre-computed 샘플이 즉시 로드되어 “빈 화면”이 없다.
 - AC7: 단계 진행 상태가 표준 상태머신으로 노출되고, 실패 시 `failure_reason`이 사용자에게 표시된다.
 - AC8: Root 특이점 대응을 사용하더라도 지표는 all_nodes/masked_nodes가 동시에 표기된다.


