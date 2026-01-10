"""GNN Surrogate Model for UAV Wing Stress Prediction."""

from deep_fem_uav_wing.gnn.dataset import WingStressDataset, build_graph_data
from deep_fem_uav_wing.gnn.model import GraphSAGEModel

__all__ = ["WingStressDataset", "build_graph_data", "GraphSAGEModel"]
