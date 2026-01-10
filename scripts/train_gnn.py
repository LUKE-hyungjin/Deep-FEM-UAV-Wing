#!/usr/bin/env python3
"""Train GNN model for Wing Stress Prediction.

Usage:
    python scripts/train_gnn.py --epochs 100 --batch-size 4

The script will:
1. Build/load the dataset from data/raw/
2. Train GraphSAGE model with loss_mask support
3. Save checkpoints to checkpoints/
4. Log training progress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    import torch.optim as optim
    from torch_geometric.loader import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch and PyTorch Geometric are required.")
    print("Install with:")
    print("  pip install torch torch-geometric")
    sys.exit(1)

from deep_fem_uav_wing.gnn.dataset import WingStressDataset
from deep_fem_uav_wing.gnn.model import GraphSAGEModel, MaskedMSELoss, compute_metrics


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y, data.loss_mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        n_samples += data.num_graphs

    return total_loss / n_samples if n_samples > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, log_scale=True):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_metrics_list = {"mae": [], "rmse": [], "max_error": []}
    masked_metrics_list = {"mae": [], "rmse": [], "max_error": []}

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y, data.loss_mask)

        total_loss += loss.item() * data.num_graphs
        n_samples += data.num_graphs

        # Compute metrics per sample
        metrics = compute_metrics(out, data.y, data.loss_mask, log_scale=log_scale)
        all_metrics_list["mae"].append(metrics["all_nodes"]["mae"])
        all_metrics_list["rmse"].append(metrics["all_nodes"]["rmse"])
        all_metrics_list["max_error"].append(metrics["all_nodes"]["max_error"])
        masked_metrics_list["mae"].append(metrics["masked_nodes"]["mae"])
        masked_metrics_list["rmse"].append(metrics["masked_nodes"]["rmse"])
        masked_metrics_list["max_error"].append(metrics["masked_nodes"]["max_error"])

    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0

    # Average metrics
    avg_metrics = {
        "all_nodes": {
            "mae": sum(all_metrics_list["mae"]) / len(all_metrics_list["mae"]) if all_metrics_list["mae"] else 0.0,
            "rmse": sum(all_metrics_list["rmse"]) / len(all_metrics_list["rmse"]) if all_metrics_list["rmse"] else 0.0,
            "max_error": max(all_metrics_list["max_error"]) if all_metrics_list["max_error"] else 0.0,
        },
        "masked_nodes": {
            "mae": sum(masked_metrics_list["mae"]) / len(masked_metrics_list["mae"]) if masked_metrics_list["mae"] else 0.0,
            "rmse": sum(masked_metrics_list["rmse"]) / len(masked_metrics_list["rmse"]) if masked_metrics_list["rmse"] else 0.0,
            "max_error": max(masked_metrics_list["max_error"]) if masked_metrics_list["max_error"] else 0.0,
        },
    }

    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GNN for Wing Stress Prediction")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Hidden channels")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Train] Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("[Train] Loading datasets...")
    train_dataset = WingStressDataset(PROJECT_ROOT, split="train", seed=args.seed)
    val_dataset = WingStressDataset(PROJECT_ROOT, split="val", seed=args.seed)
    test_dataset = WingStressDataset(PROJECT_ROOT, split="test", seed=args.seed)

    print(f"[Train] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = GraphSAGEModel(
        in_channels=10,
        hidden_channels=args.hidden_channels,
        out_channels=1,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"[Train] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    train_log = []

    print(f"[Train] Starting training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Log
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mae_all": val_metrics["all_nodes"]["mae"],
            "val_mae_masked": val_metrics["masked_nodes"]["mae"],
            "val_rmse_all": val_metrics["all_nodes"]["rmse"],
            "val_rmse_masked": val_metrics["masked_nodes"]["rmse"],
            "lr": current_lr,
            "epoch_time_s": epoch_time,
        }
        train_log.append(log_entry)

        # Print progress
        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"MAE(all/masked): {val_metrics['all_nodes']['mae']:.2e}/{val_metrics['masked_nodes']['mae']:.2e} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, best_path)
            print(f"  -> Saved best model to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[Train] Early stopping at epoch {epoch}")
                break

    total_time = time.time() - start_time
    print(f"\n[Train] Training completed in {total_time:.1f}s")

    # Load best model for final evaluation
    print("[Train] Loading best model for final evaluation...")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test evaluation
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print("\n[Train] Final Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  All Nodes   - MAE: {test_metrics['all_nodes']['mae']:.2e} Pa, RMSE: {test_metrics['all_nodes']['rmse']:.2e} Pa, Max: {test_metrics['all_nodes']['max_error']:.2e} Pa")
    print(f"  Masked Nodes - MAE: {test_metrics['masked_nodes']['mae']:.2e} Pa, RMSE: {test_metrics['masked_nodes']['rmse']:.2e} Pa, Max: {test_metrics['masked_nodes']['max_error']:.2e} Pa")

    # Save training log
    log_path = checkpoint_dir / "training_log.json"
    log_data = {
        "args": vars(args),
        "device": str(device),
        "total_time_s": total_time,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "train_log": train_log,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    print(f"[Train] Saved training log to {log_path}")

    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "in_channels": 10,
            "hidden_channels": args.hidden_channels,
            "out_channels": 1,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "test_metrics": test_metrics,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }, final_path)
    print(f"[Train] Saved final model to {final_path}")


if __name__ == "__main__":
    main()
