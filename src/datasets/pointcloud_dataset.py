import json
import os

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    def __init__(self, json_path, n_points=2048, base_dir=None):
        """
        Args:
            json_path: Path to JSON file containing dataset information.
            n_points: Number of points to sample from each point cloud.
            base_dir: Base directory to prepend for local file paths.
        """
        with open(json_path, "r") as f:
            self.data_info = json.load(f)
        self.n_points = n_points
        self.base_dir = base_dir

    def random_sample_points(self, points, n_points):
        """Randomly sample a fixed number of points from a point cloud."""
        num_points = points.shape[0]
        if num_points == 0:
            print("Warning: Empty point cloud detected, returning zeros.")
            return np.zeros((n_points, 3))
        if num_points >= n_points:
            idx = np.random.choice(num_points, n_points, replace=False)
            return points[idx]
        else:
            idx = np.random.choice(num_points, n_points, replace=True)
            return points[idx]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        pair = self.data_info[idx]
        vehicle_path = pair["vehicle_pointcloud_path"]
        infra_path = pair["infrastructure_pointcloud_path"]
        if self.base_dir:
            vehicle_path = os.path.join(self.base_dir, vehicle_path)
            infra_path = os.path.join(self.base_dir, infra_path)
        vehicle_pcd = o3d.io.read_point_cloud(vehicle_path)
        infra_pcd = o3d.io.read_point_cloud(infra_path)

        # Convert to numpy arrays and sample a fixed number of points.
        vehicle_points = np.asarray(vehicle_pcd.points)
        infra_points = np.asarray(infra_pcd.points)

        vehicle_points = self.random_sample_points(vehicle_points, self.n_points)
        infra_points = self.random_sample_points(infra_points, self.n_points)

        # Load ground truth transformation.
        gt_matrix = np.eye(4)
        if "calib_lidar_i2v_path" in pair:
            calib_path = pair["calib_lidar_i2v_path"]
            if self.base_dir:
                calib_path = os.path.join(self.base_dir, calib_path)
            with open(calib_path, "r") as f:
                calib = json.load(f)
                rotation = np.array(calib["rotation"])
                translation = np.array(calib["translation"])
                gt_matrix[:3, :3] = rotation
                gt_matrix[:3, 3] = translation.flatten()

        # Convert to torch tensors.
        vehicle_tensor = torch.tensor(vehicle_points, dtype=torch.float32)
        infra_tensor = torch.tensor(infra_points, dtype=torch.float32)
        gt_tensor = torch.tensor(gt_matrix, dtype=torch.float32)

        # Stack the point clouds into a tensor with shape [2, n_points, 3].
        point_clouds = torch.stack([vehicle_tensor, infra_tensor], dim=0)

        return point_clouds, gt_tensor
