import sys
from pathlib import Path
import json
import numpy as np

# Add src to sys.path
sys.path.append(str(Path.cwd() / "src"))

from deep_fem_uav_wing.config import Paths
from deep_fem_uav_wing.types import WingParams
from deep_fem_uav_wing.geometry import prepare_geometry_case
from deep_fem_uav_wing.meshing import run_meshing_case
from deep_fem_uav_wing.fem import run_fem_case, find_ccx_bin

def main():
    print(">>> Verifying Pipeline...")
    
    paths = Paths(project_root=Path.cwd())
    
    # 1. Validation Case 1
    params = WingParams(
        span_m=1.2,
        chord_m=0.35,
        sweep_deg=0.0,
        thickness_ratio=0.10
    )
    
    print(f">>> Case 1: {params}")
    
    # 2. Geometry
    print(">>> [1/3] Running Geometry...")
    case_id, geom_artifacts, geom_report, geom_log = prepare_geometry_case(
        paths=paths,
        params=params,
        input_stl_path=None,
        force_rebuild=True
    )
    print(f"    Case ID: {case_id}")
    print(f"    Status: {geom_report.status}")
    if geom_report.status != "success":
        print(f"    Reason: {geom_report.failure_reason}")
        print(geom_log)
        return

    # 3. Meshing
    print(">>> [2/3] Running Meshing...")
    ok, mesh_report, msg = run_meshing_case(
        case_id=case_id,
        geometry_dir=paths.geometry_dir,
        mesh_dir=paths.mesh_dir
    )
    print(f"    Status: {mesh_report['status']}")
    if not ok:
        print(f"    Reason: {mesh_report.get('failure_reason')}")
        print(msg)
        return

    # 4. FEM
    print(">>> [3/3] Running FEM...")
    
    ccx = find_ccx_bin()
    if not ccx:
        print("!!! WARNING: ccx binary not found. FEM step will likely fail.")
    else:
        print(f"    Found ccx: {ccx}")

    ok, fem_report, fem_artifacts = run_fem_case(
        case_id=case_id,
        geometry_dir=paths.geometry_dir,
        mesh_dir=paths.mesh_dir,
        fem_dir=paths.fem_dir,
        pressure_pa=100.0, # Small pressure for test
    )
    print(f"    Status: {fem_report['status']}")
    
    if not ok:
        print(f"    Reason: {fem_report.get('failure_reason')}")
        if 'stdout_tail' in fem_report:
            print("    Stdout:", fem_report['stdout_tail'])
        if 'stderr_tail' in fem_report:
            print("    Stderr:", fem_report['stderr_tail'])
        return

    # 5. Check Results
    print(">>> Verifying Results...")
    npz_path = Path(fem_artifacts.surface_results_npz)
    if not npz_path.exists():
        print("!!! surface_results.npz missing")
        return
        
    data = np.load(npz_path)
    stress = data['stress_vm']
    disp = data['disp']
    mask = data['loss_mask']
    
    print(f"    Nodes: {stress.shape[0]}")
    print(f"    Stress Range: {stress.min():.2f} ~ {stress.max():.2f} Pa")
    print(f"    Disp Range: {disp.min():.2e} ~ {disp.max():.2e} m")
    
    valid_stress = stress[mask]
    if valid_stress.size > 0:
         print(f"    Valid Stress Range (Masked): {valid_stress.min():.2f} ~ {valid_stress.max():.2f} Pa")
    
    if np.any(np.isnan(stress)) or np.any(np.isinf(stress)):
        print("!!! Found NaN/Inf in stress")
    else:
        print("    No NaN/Inf in stress.")

    print(">>> Success!")

if __name__ == "__main__":
    main()

