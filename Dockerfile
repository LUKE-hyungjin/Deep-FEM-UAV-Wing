FROM python:3.10-slim

# 1. 필수 시스템 패키지 설치 (Gmsh, CalculiX, 3D 렌더링용 라이브러리)
# libgl1-mesa-glx, xvfb: PyVista/VTK Headless 렌더링 지원
# calculix-ccx: FEM 솔버
# gmsh: 메싱 툴
RUN apt-get update && apt-get install -y \
    calculix-ccx \
    gmsh \
    libgl1-mesa-glx \
    libxrender1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

# 5. 권한 설정 (Hugging Face는 랜덤 유저 ID로 실행될 수 있음)
RUN chmod -R 777 /app

# 6. 환경 변수 설정
# PyVista가 화면 없는 곳에서 돌아가도록 설정
ENV PYVISTA_OFF_SCREEN=true
ENV PYVISTA_USE_IPYVTK=true

# 7. 실행 명령 (Gradio 앱 실행)
CMD ["python", "app.py"]

