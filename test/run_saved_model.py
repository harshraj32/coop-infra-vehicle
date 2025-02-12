import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.pointcloud_dataset import PointCloudDataset
from src.losses.losses import rmsd_loss, sixd_to_matrix
from src.models.pointnet import PointNet


def evaluate_saved_model(model, test_loader, device):
    """
    Evaluate the loaded model on test data.
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


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the saved model
    saved_model_path = os.path.join("saved_models", "pointnet_best_model.pth")
    if not os.path.exists(saved_model_path):
        print("Saved model not found at", saved_model_path)
        exit(1)

    # Instantiate the model and load checkpoint
    model = PointNet().to(device)
    checkpoint = torch.load(saved_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded saved model from", saved_model_path)

    # Create the test dataset.
    # Update the JSON file path and base directory as needed.
    json_path = "data/cooperative-vehicle-infrastructure/cooperative/data_info_new.json"
    dataset = PointCloudDataset(
        json_path, base_dir="data/cooperative-vehicle-infrastructure"
    )

    # For this test, we use the entire dataset as test data.
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Evaluate the loaded model
    avg_loss, avg_rot_error, avg_trans_error = evaluate_saved_model(
        model, test_loader, device
    )

    print("\nEvaluation Results for the saved model:")
    print("Average Loss:", avg_loss)
    print("Average Rotation Error (deg):", avg_rot_error)
    print("Average Translation Error:", avg_trans_error)
