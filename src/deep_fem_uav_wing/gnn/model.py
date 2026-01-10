"""GraphSAGE Model for Wing Stress Prediction.

Architecture:
- Encoder: MLP [10 -> 64 -> 128]
- GNN: 3-4 layers of SAGEConv with skip connections
- Decoder: MLP [128 -> 64 -> 1]
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, global_mean_pool

    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


if HAS_TORCH_GEOMETRIC:

    class GraphSAGEModel(nn.Module):
        """GraphSAGE model for node-level stress prediction.

        Args:
            in_channels: Input feature dimension (default: 10)
            hidden_channels: Hidden layer dimension (default: 128)
            out_channels: Output dimension (default: 1)
            num_layers: Number of SAGE layers (default: 4)
            dropout: Dropout rate (default: 0.1)
        """

        def __init__(
            self,
            in_channels: int = 10,
            hidden_channels: int = 128,
            out_channels: int = 1,
            num_layers: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()

            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels
            self.num_layers = num_layers
            self.dropout = dropout

            # Encoder: input features -> hidden
            self.encoder = nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.ReLU(),
                nn.Linear(64, hidden_channels),
                nn.ReLU(),
            )

            # GNN layers with skip connections
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))

            # Decoder: hidden -> output
            self.decoder = nn.Sequential(
                nn.Linear(hidden_channels, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, out_channels),
            )

        def forward(self, x, edge_index, batch=None):
            """Forward pass.

            Args:
                x: Node features [N, in_channels]
                edge_index: Edge connectivity [2, E]
                batch: Batch assignment (optional, for batched graphs)

            Returns:
                Node-level predictions [N, out_channels]
            """
            # Encode
            h = self.encoder(x)

            # GNN layers with skip connections
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_new = conv(h, edge_index)
                h_new = norm(h_new)
                h_new = F.relu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                # Skip connection
                h = h + h_new

            # Decode
            out = self.decoder(h)
            return out

        def predict(self, data):
            """Convenience method for inference.

            Args:
                data: PyG Data object with x, edge_index

            Returns:
                Predictions tensor [N, 1]
            """
            self.eval()
            with torch.no_grad():
                return self.forward(data.x, data.edge_index, getattr(data, "batch", None))


    class MaskedMSELoss(nn.Module):
        """MSE Loss with masking support for root singularity.

        Args:
            reduction: 'mean' or 'sum'
        """

        def __init__(self, reduction: str = "mean"):
            super().__init__()
            self.reduction = reduction

        def forward(self, pred, target, mask=None):
            """Compute masked MSE loss.

            Args:
                pred: Predictions [N, 1]
                target: Targets [N, 1]
                mask: Boolean mask [N] - True = include in loss

            Returns:
                Loss scalar
            """
            if mask is None:
                # No mask, use all nodes
                loss = F.mse_loss(pred, target, reduction=self.reduction)
            else:
                # Apply mask
                if mask.dim() == 1:
                    mask = mask.unsqueeze(-1)
                masked_pred = pred[mask.expand_as(pred)].view(-1, pred.size(-1))
                masked_target = target[mask.expand_as(target)].view(-1, target.size(-1))

                if masked_pred.numel() == 0:
                    # All masked out, return 0
                    return torch.tensor(0.0, device=pred.device, requires_grad=True)

                loss = F.mse_loss(masked_pred, masked_target, reduction=self.reduction)

            return loss


    def compute_metrics(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        log_scale: bool = True,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            pred: Predictions [N, 1] (in log scale if log_scale=True)
            target: Targets [N, 1] (in log scale if log_scale=True)
            mask: Boolean mask [N] - True = include
            log_scale: Whether predictions are in log scale

        Returns:
            Dictionary with MAE, RMSE, Max error (all in original scale)
        """
        pred = pred.detach().cpu().numpy().flatten()
        target = target.detach().cpu().numpy().flatten()

        if mask is not None:
            mask = mask.detach().cpu().numpy().flatten()
        else:
            mask = None

        # Convert back to original scale if log-scaled
        if log_scale:
            import numpy as np
            pred_orig = np.expm1(pred)  # inverse of log1p
            target_orig = np.expm1(target)
        else:
            pred_orig = pred
            target_orig = target

        def compute_for_subset(p, t, m):
            if m is not None:
                p = p[m]
                t = t[m]

            if len(p) == 0:
                return {"mae": 0.0, "rmse": 0.0, "max_error": 0.0, "count": 0}

            import numpy as np
            errors = np.abs(p - t)
            return {
                "mae": float(np.mean(errors)),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "max_error": float(np.max(errors)),
                "count": int(len(p)),
            }

        # All nodes
        all_metrics = compute_for_subset(pred_orig, target_orig, None)

        # Masked nodes only
        masked_metrics = compute_for_subset(pred_orig, target_orig, mask)

        return {
            "all_nodes": all_metrics,
            "masked_nodes": masked_metrics,
        }


else:
    # Dummy classes when PyTorch Geometric is not available
    class GraphSAGEModel:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Geometric is required. Install with:\n"
                "  pip install torch torch-geometric"
            )

    class MaskedMSELoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required")

    def compute_metrics(*args, **kwargs):
        raise ImportError("PyTorch is required")
