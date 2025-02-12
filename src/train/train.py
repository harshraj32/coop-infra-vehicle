import argparse
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.losses.losses import rmsd_icp_with_rot_trans_loss, rmsd_loss, sixd_to_matrix
from src.models.pointnet import build_pointnet

# Device selection: use MPS (for Mac) if available, otherwise CUDA or CPU.
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

BEST_MODEL_DIR = "saved_models"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data.
    Returns average loss, rotation error, and translation error.
    """
    model.eval()
    total_loss = 0
    rotation_errors = []
    translation_errors = []

    with torch.no_grad():
        for point_clouds, gt_matrix in test_loader:
            point_clouds = point_clouds.to(device)
            gt_matrix = gt_matrix.to(device)

            pred_params = model(point_clouds)
            loss = rmsd_loss(pred_params, gt_matrix, point_clouds)
            total_loss += loss.item()

            pred_rot = sixd_to_matrix(pred_params[:, :6])
            pred_trans = pred_params[:, 6:]

            for i in range(pred_rot.shape[0]):
                R_diff = torch.matmul(pred_rot[i].T, gt_matrix[i, :3, :3])
                rotation_error = (
                    torch.acos(torch.clamp((torch.trace(R_diff) - 1) / 2, -1.0, 1.0))
                    * 180
                    / np.pi
                )
                rotation_errors.append(rotation_error.item())
                trans_error = torch.norm(pred_trans[i] - gt_matrix[i, :3, 3]).item()
                translation_errors.append(trans_error)

    avg_loss = total_loss / len(test_loader)
    avg_rotation_error = np.mean(rotation_errors)
    avg_translation_error = np.mean(translation_errors)

    return avg_loss, avg_rotation_error, avg_translation_error


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointNet registration model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="pointnet-vanilla",
        choices=["pointnet-vanilla", "pointnet-cnn"],
        help="Type of model to use.",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate.")
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Fraction of data for training."
    )
    parser.add_argument(
        "--hidden-size", type=int, default=1024, help="Hidden size of the model."
    )
    parser.add_argument(
        "--num-points", type=int, default=2048, help="Number of points per point cloud."
    )
    parser.add_argument(
        "--batch-norm", action="store_true", help="Use batch normalization."
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="data/cooperative-vehicle-infrastructure/cooperative/data_info_new.json",
        help="Path to dataset JSON.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/cooperative-vehicle-infrastructure",
        help="Base directory for dataset files.",
    )
    return parser.parse_args()


def train_and_test_pointnet(
    dataset_path,
    epochs=100,
    batch_size=16,
    lr=0.005,
    train_split=0.8,
    model=None,
    base_dir="data/cooperative-vehicle-infrastructure",
):
    # Load dataset with the given base directory.
    dataset = PointCloudDataset(dataset_path, base_dir=base_dir)

    if model is None:
        # Fallback: build a default vanilla model with default hyperparameters.
        model = build_pointnet(
            model_type="vanilla", hidden_size=1024, num_points=2048, batch_norm=True
        ).to(device)

    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Training set size: {train_size}")
    print(f"Test set size: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    warmup_epochs = 5
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )

    best_loss = float("inf")
    training_history = []
    test_history = []

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for point_clouds, gt_matrix in train_loader:
            point_clouds = point_clouds.to(device)
            gt_matrix = gt_matrix.to(device)

            optimizer.zero_grad()
            pred_params = model(point_clouds)
            loss = rmsd_icp_with_rot_trans_loss(pred_params, gt_matrix, point_clouds)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_train_loss += loss.item()

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        training_history.append(avg_train_loss)

        test_loss, rotation_error, translation_error = evaluate_model(
            model, test_loader, device
        )
        test_history.append(test_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Avg Rotation Error: {rotation_error:.2f}°")
        print(f"Avg Translation Error: {translation_error:.4f} units")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 50)

        if test_loss < best_loss:
            best_loss = test_loss
            model_save_path = os.path.join(BEST_MODEL_DIR, "pointnet_best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "test_loss": test_loss,
                    "rotation_error": rotation_error,
                    "translation_error": translation_error,
                },
                model_save_path,
            )

    print("\nFinal Evaluation on Test Set:")
    test_loss, rotation_error, translation_error = evaluate_model(
        model, test_loader, device
    )
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Rotation Error: {rotation_error:.2f}°")
    print(f"Final Translation Error: {translation_error:.4f} units")

    return model, training_history, test_history


if __name__ == "__main__":
    args = parse_args()

    # Print training configuration.
    print("Training configuration:")
    print("  Model type:       ", args.model_type)
    print("  Epochs:           ", args.epochs)
    print("  Batch size:       ", args.batch_size)
    print("  Learning rate:    ", args.lr)
    print("  Train split:      ", args.train_split)
    print("  Hidden size:      ", args.hidden_size)
    print("  Num points:       ", args.num_points)
    print("  Batch norm:       ", args.batch_norm)
    print("  JSON path:        ", args.json_path)
    print("  Base directory:   ", args.base_dir)

    # Build the desired model based on the model type argument.
    if args.model_type == "pointnet-vanilla":
        model = build_pointnet(
            model_type="vanilla",
            hidden_size=args.hidden_size,
            num_points=args.num_points,
            batch_norm=args.batch_norm,
        ).to(device)
    elif args.model_type == "pointnet-cnn":
        model = build_pointnet(
            model_type="cnn",
            hidden_size=args.hidden_size,
            num_points=args.num_points,
            batch_norm=args.batch_norm,
        ).to(device)
    else:
        raise ValueError("Invalid model type specified.")

    print("Using model: ", args.model_type)

    train_and_test_pointnet(
        args.json_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_split=args.train_split,
        model=model,
        base_dir=args.base_dir,
    )
