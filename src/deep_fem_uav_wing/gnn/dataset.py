"""Graph Dataset Builder for Wing Stress Prediction.

Converts surface_results.npz + boundary_sets.json + params.json
into PyTorch Geometric Data objects for GNN training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch_geometric.data import Data, InMemoryDataset

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    Data = None
    InMemoryDataset = object


def _faces_to_edge_index(
    faces: list[list[int]],
    node_id_to_idx: dict[int, int],
) -> np.ndarray:
    """Convert triangle faces to edge_index (undirected edges).

    Args:
        faces: List of [n0, n1, n2] node IDs
        node_id_to_idx: Mapping from original node ID to 0-based index

    Returns:
        edge_index: [2, E] array of edges
    """
    edges = set()
    for face in faces:
        # Get indices, skip if any node not in mapping
        idxs = []
        for nid in face:
            if nid not in node_id_to_idx:
                break
            idxs.append(node_id_to_idx[nid])
        else:
            # Add all 3 edges (undirected)
            n0, n1, n2 = idxs
            edges.add((min(n0, n1), max(n0, n1)))
            edges.add((min(n1, n2), max(n1, n2)))
            edges.add((min(n2, n0), max(n2, n0)))

    # Convert to edge_index format (both directions for undirected)
    edge_list = []
    for i, j in edges:
        edge_list.append([i, j])
        edge_list.append([j, i])

    if not edge_list:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(edge_list, dtype=np.int64).T


def build_graph_data(
    surface_npz_path: Path,
    boundary_sets_path: Path,
    params_path: Path,
    *,
    log_scale_stress: bool = True,
    normalize_pos: bool = True,
) -> dict[str, Any]:
    """Build graph data from FEM results.

    Args:
        surface_npz_path: Path to surface_results.npz
        boundary_sets_path: Path to boundary_sets.json
        params_path: Path to params.json
        log_scale_stress: Apply log1p transform to stress (recommended)
        normalize_pos: Normalize positions to [0, 1] range

    Returns:
        Dictionary with:
            - x: [N, 10] node features (pos + normal + global_params)
            - edge_index: [2, E] edge connectivity
            - y: [N, 1] target stress
            - loss_mask: [N] bool mask for loss computation
            - pos: [N, 3] original positions
            - case_id: str
            - global_params: [4] (span, chord, sweep, thickness)
    """
    # Load data
    npz = np.load(surface_npz_path)
    boundary_sets = json.loads(boundary_sets_path.read_text(encoding="utf-8"))
    params = json.loads(params_path.read_text(encoding="utf-8"))

    # Extract arrays
    node_ids = npz["node_id"]  # [N]
    pos = npz["pos"].astype(np.float32)  # [N, 3]
    normal = npz["normal"].astype(np.float32)  # [N, 3]
    stress_vm = npz["stress_vm"].astype(np.float32)  # [N]
    disp = npz["disp"].astype(np.float32)  # [N, 3]
    loss_mask = npz["loss_mask"].astype(bool)  # [N]

    N = len(node_ids)

    # Create node_id to index mapping
    node_id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}

    # Build edge_index from surface faces
    surf_all_faces = boundary_sets["surf_all_faces"]
    edge_index = _faces_to_edge_index(surf_all_faces, node_id_to_idx)

    # Global parameters (normalized to roughly [0, 1] range)
    # Based on PRD ranges: span 1-2m, chord 0.2-0.5m, sweep 0-30deg, t/c 0.05-0.15
    span_m = params["span_m"]
    chord_m = params["chord_m"]
    sweep_deg = params["sweep_deg"]
    thickness_ratio = params["thickness_ratio"]

    global_params = np.array([
        (span_m - 1.0) / 1.0,  # [0, 1] for span 1-2m
        (chord_m - 0.2) / 0.3,  # [0, 1] for chord 0.2-0.5m
        sweep_deg / 30.0,  # [0, 1] for sweep 0-30deg
        (thickness_ratio - 0.05) / 0.10,  # [0, 1] for t/c 0.05-0.15
    ], dtype=np.float32)

    # Normalize positions if requested
    pos_normalized = pos.copy()
    if normalize_pos:
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        pos_range = pos_max - pos_min
        pos_range[pos_range < 1e-8] = 1.0  # Avoid division by zero
        pos_normalized = (pos - pos_min) / pos_range

    # Normalize normals (should already be unit, but ensure)
    norm_lengths = np.linalg.norm(normal, axis=1, keepdims=True)
    norm_lengths[norm_lengths < 1e-8] = 1.0
    normal_normalized = normal / norm_lengths

    # Build node features: [pos_normalized(3) + normal(3) + global_params(4)] = 10D
    global_params_repeated = np.tile(global_params, (N, 1))  # [N, 4]
    x = np.concatenate([pos_normalized, normal_normalized, global_params_repeated], axis=1)  # [N, 10]

    # Target: stress (optionally log-scaled)
    y = stress_vm.copy()
    if log_scale_stress:
        y = np.log1p(y)  # log(1 + stress) for numerical stability
    y = y.reshape(-1, 1)  # [N, 1]

    return {
        "x": x.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "y": y.astype(np.float32),
        "loss_mask": loss_mask,
        "pos": pos.astype(np.float32),
        "disp": disp.astype(np.float32),
        "stress_vm_raw": stress_vm.astype(np.float32),  # Original scale for evaluation
        "case_id": params["case_id"],
        "global_params": global_params.astype(np.float32),
        "global_params_raw": np.array([span_m, chord_m, sweep_deg, thickness_ratio], dtype=np.float32),
    }


if HAS_TORCH_GEOMETRIC:
    class WingStressDataset(InMemoryDataset):
        """PyTorch Geometric Dataset for Wing Stress Prediction.

        Args:
            root: Root directory containing data/raw/fem, data/raw/mesh, data/raw/geometry
            split: 'train', 'val', or 'test'
            split_ratio: (train_ratio, val_ratio) - test is remainder
            seed: Random seed for reproducibility
            log_scale_stress: Apply log1p transform to stress
            normalize_pos: Normalize positions to [0, 1]
            transform: Optional transform to apply
            pre_transform: Optional pre-transform
        """

        def __init__(
            self,
            root: str | Path,
            split: str = "train",
            split_ratio: tuple[float, float] = (0.7, 0.15),
            seed: int = 42,
            log_scale_stress: bool = True,
            normalize_pos: bool = True,
            transform=None,
            pre_transform=None,
        ):
            self.split = split
            self.split_ratio = split_ratio
            self.seed = seed
            self.log_scale_stress = log_scale_stress
            self.normalize_pos = normalize_pos

            root = Path(root)
            super().__init__(str(root), transform, pre_transform)

            # Load appropriate split
            split_idx = {"train": 0, "val": 1, "test": 2}[split]
            self.data, self.slices = torch.load(self.processed_paths[split_idx])

        @property
        def raw_dir(self) -> str:
            return str(Path(self.root) / "data" / "raw")

        @property
        def processed_dir(self) -> str:
            return str(Path(self.root) / "data" / "processed" / "gnn")

        @property
        def raw_file_names(self) -> list[str]:
            return ["fem", "mesh", "geometry"]

        @property
        def processed_file_names(self) -> list[str]:
            return [
                f"train_s{self.seed}.pt",
                f"val_s{self.seed}.pt",
                f"test_s{self.seed}.pt",
            ]

        def download(self):
            # Data should already exist from FEM pipeline
            pass

        def process(self):
            """Process raw data into PyG Data objects."""
            raw_dir = Path(self.raw_dir)
            fem_dir = raw_dir / "fem"
            mesh_dir = raw_dir / "mesh"
            geometry_dir = raw_dir / "geometry"

            # Find all valid cases
            case_ids = []
            for case_dir in sorted(fem_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                case_id = case_dir.name

                # Check all required files exist
                surface_npz = fem_dir / case_id / "surface_results.npz"
                boundary_json = mesh_dir / case_id / "boundary_sets.json"
                params_json = geometry_dir / case_id / "params.json"

                if surface_npz.exists() and boundary_json.exists() and params_json.exists():
                    case_ids.append(case_id)

            print(f"[Dataset] Found {len(case_ids)} valid cases")

            # Build Data objects
            data_list = []
            for case_id in case_ids:
                try:
                    graph_data = build_graph_data(
                        surface_npz_path=fem_dir / case_id / "surface_results.npz",
                        boundary_sets_path=mesh_dir / case_id / "boundary_sets.json",
                        params_path=geometry_dir / case_id / "params.json",
                        log_scale_stress=self.log_scale_stress,
                        normalize_pos=self.normalize_pos,
                    )

                    data = Data(
                        x=torch.from_numpy(graph_data["x"]),
                        edge_index=torch.from_numpy(graph_data["edge_index"]),
                        y=torch.from_numpy(graph_data["y"]),
                        loss_mask=torch.from_numpy(graph_data["loss_mask"]),
                        pos=torch.from_numpy(graph_data["pos"]),
                        disp=torch.from_numpy(graph_data["disp"]),
                        stress_vm_raw=torch.from_numpy(graph_data["stress_vm_raw"]),
                        case_id=graph_data["case_id"],
                        global_params=torch.from_numpy(graph_data["global_params"]),
                        global_params_raw=torch.from_numpy(graph_data["global_params_raw"]),
                    )

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                except Exception as e:
                    print(f"[Dataset] Failed to process {case_id}: {e}")

            print(f"[Dataset] Successfully processed {len(data_list)} cases")

            # Split into train/val/test
            np.random.seed(self.seed)
            indices = np.random.permutation(len(data_list))

            n_train = int(len(data_list) * self.split_ratio[0])
            n_val = int(len(data_list) * self.split_ratio[1])

            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]

            splits = [
                [data_list[i] for i in train_indices],
                [data_list[i] for i in val_indices],
                [data_list[i] for i in test_indices],
            ]

            print(f"[Dataset] Split: train={len(splits[0])}, val={len(splits[1])}, test={len(splits[2])}")

            # Save splits
            Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
            for split_data, path in zip(splits, self.processed_paths):
                data, slices = self.collate(split_data)
                torch.save((data, slices), path)

            # Save split info for reproducibility
            split_info = {
                "seed": self.seed,
                "split_ratio": self.split_ratio,
                "n_total": len(data_list),
                "n_train": len(splits[0]),
                "n_val": len(splits[1]),
                "n_test": len(splits[2]),
                "train_case_ids": [data_list[i].case_id for i in train_indices],
                "val_case_ids": [data_list[i].case_id for i in val_indices],
                "test_case_ids": [data_list[i].case_id for i in test_indices],
            }
            split_info_path = Path(self.processed_dir) / f"split_info_s{self.seed}.json"
            split_info_path.write_text(json.dumps(split_info, indent=2), encoding="utf-8")
            print(f"[Dataset] Saved split info to {split_info_path}")


else:
    # Dummy class when PyTorch Geometric is not available
    class WingStressDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Geometric is required. Install with:\n"
                "  pip install torch torch-geometric"
            )
