# Blender로 에어포일 200개 데이터 생성
## Blender 데이터 생성
naca0000으로 200개 데이터 생성
- stl
- gltf
- .json
- .json
```
python scripts/generate_geometry_dataset.py --count 200 --seed 42
```
![[Pasted image 20260104170019.png]]

graido에서 보이지않는다. 
사유: json gltf가 gradio에서 보이려면 변경이 필요함
![[Pasted image 20260104170318.png]]

# Meshing
