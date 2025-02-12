import torch
import torch.nn.functional as F
import numpy as np
import src.penalties.penalties as P  # penalties.py should be located at the project root

def sixd_to_matrix(sixd):
    """Convert six-d rotation representation to rotation matrix."""
    a1, a2 = sixd[..., :3], sixd[..., 3:6]
    # Normalize the first vector.
    b1 = F.normalize(a1, dim=-1)
    # Get a second vector orthogonal to b1.
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    # Get a third vector orthogonal to both.
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_sixd(matrix):
    """Convert rotation matrix to six-d representation."""
    return torch.cat([matrix[..., :3, 0], matrix[..., :3, 1]], dim=-1)

def compute_rotation_error(pred_rot, gt_rot):
    """Compute the geodesic distance between predicted and ground truth rotations."""
    relative_rotation = torch.bmm(pred_rot, gt_rot.transpose(1, 2))
    trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    return angle.mean()

def rmsd_icp_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute RMSD loss combined with ICP loss for better point cloud alignment.
    """
    batch_size = pred_params.shape[0]
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4, device=pred_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)  # [B, 3, N]
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1], device=pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)

    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()

    dists = torch.cdist(transformed_points.transpose(1, 2), target_cloud.transpose(1, 2), p=2)
    min_dists, _ = torch.min(dists, dim=2)
    icp_loss = min_dists.mean()

    total_loss = rmsd + 0.1 * icp_loss
    return total_loss

def rmsd_icp_with_rot_trans_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute RMSD loss combined with ICP loss, rotation error, and translation error.
    """
    batch_size = pred_params.shape[0]
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4, device=pred_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    gt_rot = gt_matrix[:, :3, :3]
    gt_trans = gt_matrix[:, :3, 3]

    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1], device=pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]

    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()

    dists = torch.cdist(transformed_points.transpose(1, 2), target_cloud.transpose(1, 2), p=2)
    min_dists, _ = torch.min(dists, dim=2)
    icp_loss = min_dists.mean()

    rot_error = compute_rotation_error(pred_rot, gt_rot)
    trans_error = torch.norm(pred_trans - gt_trans, dim=1).mean()

    total_loss = rmsd + 0.1 * icp_loss + 0.1 * rot_error + 0.1 * trans_error
    return total_loss

def rmsd_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute RMSD loss between transformed source and target points.
    """
    batch_size = pred_params.shape[0]
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4, device=pred_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1], device=pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)

    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()
    return rmsd

def frobenius_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute the Frobenius norm loss between predicted and ground truth transformation matrices.
    """
    batch_size = pred_params.shape[0]
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4, device=pred_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1], device=pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)

    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]
    loss = torch.norm(transformed_points - target_cloud, p='fro', dim=[1, 2])
    squared_diff = (transformed_points - target_cloud).pow(2).sum(1)
    rmsd = torch.sqrt(squared_diff.mean(1)).mean()
    penalty_loss = P.penalty_sum(pred_params, P.pply_constraints)
    total_loss = rmsd + 0.1 * penalty_loss
    return total_loss

def chordal_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute Chordal distance loss between transformed source and target points.
    """
    batch_size = pred_params.shape[0]
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4, device=pred_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1], device=pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]

    diff_matrix = torch.matmul(pred_rot.transpose(1, 2), gt_matrix[:, :3, :3]) - torch.eye(3, device=pred_params.device)
    loss_matrix = torch.norm(diff_matrix, p='fro', dim=[1, 2])
    point_cloud_loss = torch.norm(transformed_points - target_cloud, p=2, dim=[1, 2])
    return (loss_matrix + point_cloud_loss).mean()

def svd_loss(pred_params, gt_matrix, point_clouds):
    """
    Compute loss based on Singular Value Decomposition (SVD), incorporating point cloud alignment.
    """
    batch_size = pred_params.shape[0]
    pred_rot = sixd_to_matrix(pred_params[:, :6])
    pred_trans = pred_params[:, 6:]
    
    pred_transform = torch.eye(4, device=pred_params.device).unsqueeze(0).repeat(batch_size, 1, 1)
    pred_transform[:, :3, :3] = pred_rot
    pred_transform[:, :3, 3] = pred_trans

    source_cloud = point_clouds[:, 0, :, :].transpose(1, 2)
    target_cloud = point_clouds[:, 1, :, :].transpose(1, 2)
    ones = torch.ones(batch_size, 1, source_cloud.shape[-1], device=pred_params.device)
    source_homogeneous = torch.cat([source_cloud, ones], dim=1)
    transformed_points = torch.bmm(pred_transform, source_homogeneous)[:, :3, :]

    u, _, v = torch.svd(torch.matmul(pred_rot, gt_matrix[:, :3, :3].transpose(1, 2)))
    optimal_rot = torch.matmul(u, v.transpose(1, 2))
    rot_loss = torch.norm(optimal_rot - gt_matrix[:, :3, :3], p='fro', dim=[1, 2])
    point_cloud_loss = torch.norm(transformed_points - target_cloud, p=2, dim=[1, 2])
    return (rot_loss + point_cloud_loss).mean() 