import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, hidden_size=1024, num_points=2048, batch_norm=True):
        """
        Vanilla PointNet architecture.
        Output is 9 parameters: 6 for rotation (six-d representation) + 3 for translation.
        """
        super(PointNet, self).__init__()
        self.out_dim = 9

        self.feat_net = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.BatchNorm1d(64) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, hidden_size, 1),
            nn.BatchNorm1d(hidden_size) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.hidden_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.out_dim),
        )

    def forward(self, x):
        # x shape: [batch_size, 2, n_points, 3]
        batch_size = x.size(0)
        # Reshape and concatenate the point clouds.
        x = x.view(batch_size, 6, -1)  # [batch_size, 6, n_points]
        # Extract features.
        x = self.feat_net(x)  # [batch_size, hidden_size, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_size]
        # Compute transformation parameters.
        x = self.hidden_mlp(x)  # [batch_size, 9]
        return x


class PointNetAlt(nn.Module):
    def __init__(self, hidden_size=1024, num_points=1000, batch_norm=True):
        """
        Alternative PointNet architecture using a dedicated CNN feature extractor.
        Output is 9 parameters: 6 for rotation (six-d representation) + 3 for translation.
        """
        super(PointNetAlt, self).__init__()
        self.out_dim = 9

        self.feat_net = PointFeatCNN(hidden_size, batch_norm)
        self.hidden_mlp = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.out_dim),
        )

    def forward(self, x):
        # x shape: [batch_size, 2, n_points, 3]
        batch_size = x.size(0)
        # Reshape and concatenate the point clouds.
        x = x.view(batch_size, 6, -1)  # [batch_size, 6, n_points]
        # Extract features using the CNN-based feature extractor.
        x = self.feat_net(x)  # [batch_size, hidden_size, 1]
        x = x.squeeze(-1)  # [batch_size, hidden_size]
        # MLP for transformation parameters.
        x = self.hidden_mlp(x)  # [batch_size, 9]
        return x


class PointFeatCNN(nn.Module):
    def __init__(self, feature_dim, batch_norm=False):
        super(PointFeatCNN, self).__init__()
        if batch_norm:
            self.net = nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=1),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Conv1d(128, feature_dim, kernel_size=1),
                nn.AdaptiveMaxPool1d(output_size=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(6, 64, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv1d(128, feature_dim, kernel_size=1),
                nn.AdaptiveMaxPool1d(output_size=1),
            )

    def forward(self, x):
        return self.net(x)


def build_pointnet(model_type="vanilla", **kwargs):
    """
    Factory function to build a PointNet model.

    Args:
        model_type: one of "vanilla" (default) or "cnn".
        **kwargs: additional parameters to pass to the model.

    Returns:
        An instance of PointNet (vanilla or alternative) based on model_type.
    """
    if model_type == "vanilla":
        return PointNet(**kwargs)
    elif model_type == "cnn":
        return PointNetAlt(**kwargs)
    else:
        raise ValueError("Unknown model type. Available choices: 'vanilla', 'cnn'.")
