# PRD — AI 기반 UAV 날개 구조 해석 예측 모델 (Deep-FEM-UAV-Wing)

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

## 3) 기능 요구사항(Functional Requirements)
### FR1. Geometry 생성
- 입력: span/chord/sweep/thickness_ratio
- 출력:
  - `wing.stl`
  - `params.json`
  - `build_report.json`(유효성 검사: watertight/단위/좌표계/노멀)

### FR2. Meshing(Gmsh)
- 입력: `wing.stl`
- 출력:
  - `wing.msh`(테트라 볼륨 메쉬)
  - `mesh_report.json`(요소 수/품질/시간/실패 사유)
  - `boundary_sets.json`
    - `NROOT`: y<=y_tol 노드
    - `SURF_ALL`: 외피 face
    - `SURF_UPPER`: n_z>=nz_min 외피 face(+root 근방 제외)

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

### FR5. Dataset/Training
- 데이터셋:
  - 샘플 단위: 케이스 1개(표면 노드 그래프)
  - input: pos+normal(6) + global params(4) = **10D**, edge_index(표면 인접), target: stress_vm(log-scale 권장)
  - Root 부근 특이점 대응: 학습 loss는 `loss_mask`로 제외/클리핑을 허용하되, 지표는 `all_nodes`/`masked_nodes`를 함께 보고
- 모델:
  - GraphSAGE(3~4 layer) baseline
- 산출물:
  - 체크포인트/학습 로그/스플릿 정보(재현 가능)

### FR6. Gradio Web Demo
- UI:
  - 슬라이더: span/chord/sweep/thickness_ratio
  - 버튼: Generate, Run FEM, Run AI(모델 있을 때)
  - 출력:
    - Model3D: Geometry / FEM / AI / Error
    - File: STL/GLB/NPZ 다운로드
    - Textbox: 로그/실행 시간/실패 사유
- 캐시:
  - 동일 파라미터는 동일 `case_id`로 캐시 재사용

---

## 4) 비기능 요구사항(Non-Functional Requirements)
### 재현성
- 모든 케이스는 `params.json` + tool 버전(gmsh/ccx/blender)을 `manifest.json`에 기록

### 안정성/실패 처리
- 모든 단계는 `status, failure_reason, stdout/stderr 요약`을 남긴다.
- 실패는 조용히 넘어가지 않고, UI/로그에 명확히 표시한다.

### 배포 및 환경(Deployment)
- **타겟 플랫폼**: Hugging Face Spaces (CPU Basic)
- **배포 방식**: Docker Container (Dockerfile 기반)
- **필수 패키지**: `gmsh`, `calculix-ccx`, `xvfb`(Headless 3D 렌더링용)
- **리소스 제약**: GPU 사용 없음(CPU Only), 메모리 16GB 내외 최적화

### 성능(현실적인 목표)
- 로컬 기준:
  - Geometry+Viz: 수 초 이내
  - Meshing+FEM: 수십 초~수 분(케이스에 따라)
- 웹 데모는 장시간 작업을 고려해 “캐시 재사용”을 최우선으로 한다.

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

---

## 7) 마일스톤(2~3주)
- Week 1: Geometry + STL→GLB(버텍스 컬러 경로 PoC) + 케이스 캐시/산출물 구조 확정
- Week 2: Gmsh 메싱 + 경계 셋 + CalculiX 자동화 + surface_results.npz 생성/검증
- Week 3: GNN 학습/추론 + Gradio side-by-side + error map + metrics 완성

---

## 8) 수용 기준(Acceptance Criteria)
- AC1: 파라미터 입력 후 `wing_viz.glb`가 생성되고 Model3D에서 보인다.
- AC2: 동일 파라미터 재실행 시 같은 `case_id`로 캐시를 재사용한다.
- AC3: `Run FEM` 실행 후 `{case_id}.frd`, `surface_results.npz`, `wing_result.glb`가 생성된다.
- AC4: `wing_result.glb`에서 stress 컬러가 “항상” 보인다(버텍스 컬러 기반).
- AC5: 학습된 모델이 있으면 `wing_pred.glb`, `wing_error.glb`가 생성되고 side-by-side로 비교된다.


