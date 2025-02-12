# Cooperative Infrastructure-Vehicle Point Cloud Registration

## Overview
This project implements point cloud registration between infrastructure and vehicle sensors using PointNet-based architectures. The implementation is based on the [PointCloud_Regression](https://github.com/flatironinstitute/PointCloud_Regression) repository from Flatiron Institute, with several modifications and improvements to work with cooperative vehicle-infrastructure datasets.

## Key Modifications

### Architecture Changes
- Added a new model variant `PointNetCNN` which incorporates a modified version of the `PointFeatCNN` module
- Implemented two model options:
  - `pointnet-vanilla`: Original PointNet architecture
  - `pointnet-cnn`: Enhanced version with dedicated CNN feature extractor

### Loss Functions
- Implemented various loss functions from the original repository in `losses.py`
- Experimented with different loss functions including:
  - RMSD (Root Mean Square Deviation)
  - ICP (Iterative Closest Point)
  - RMSD + ICP
  - Frobenius norm
  - Chordal distance
  - SVD-based loss
- Found RMSD+ ICP loss to be most effective for our cooperative vehicle-infrastructure dataset

### Dataset Handling
- Modified the training pipeline to work with cooperative sensing data
- Implemented custom data loading and preprocessing for infrastructure-vehicle point cloud pairs

## Project Structure

```
project_root/
├── README.md
├── requirements.txt
├── penalties.py
├── saved_models/             # Directory where model checkpoints are saved
├── src/
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── pointcloud_dataset.py
│   ├── losses/
│   │   ├── __init__.py
│   │   └── losses.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── pointnet.py
│   └── train/
│       ├── __init__.py
│       └── train.py
└── test/
    ├── __init__.py
    └── run_saved_model.py
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/harshraj32/coop-infra-vehicle.git
cd coop-infra-vehicle
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### 1. Download the Dataset
The cooperative vehicle-infrastructure dataset can be downloaded from:
- https://drive.google.com/file/d/1y8bGwI63TEBkDEh2JU_gdV7uidthSnoe/view

### 2. Dataset Structure
After downloading, change the name of the downloaded zip file example-cooperative-vehicle-infrastructure to cooperative-vehicle-infrastructure  and add it into the data folder structure as follows:
```

data/
└── cooperative-vehicle-infrastructure/
    ├── cooperative/
    │   └── data_info_new.json     # Contains mapping between vehicle and infrastructure frames
    ├── vehicle-side/
    │   ├── velodyne/              # Vehicle LiDAR point clouds
    │   │   ├── 000000.pcd
    │   │   ├── 000001.pcd
    │   │   └── ...
    │   └── calib/                 # Calibration files
    └── infrastructure-side/
        ├── velodyne/              # Infrastructure LiDAR point clouds
        │   ├── 000000.pcd
        │   ├── 000001.pcd
        │   └── ...
        └── calib/                 # Calibration files
```


### 4. JSON File Format
The `data_info_new.json` file should contain entries in the following format:
```json
[
    {
        "vehicle_pointcloud_path": "vehicle-side/velodyne/000123.pcd",
        "infrastructure_pointcloud_path": "infrastructure-side/velodyne/000456.pcd",
        "calib_lidar_i2v_path": "cooperative/calib/000123.json"
    },
    ...
]
```

### 5. Data Loading
The dataset loader (`src/datasets/pointcloud_dataset.py`) will:
- Read point clouds from both vehicle and infrastructure sides
- Load calibration information
- Sample a fixed number of points (configurable via `--num-points`)
- Return synchronized pairs for training

### Notes
- Point clouds are in PCD format
- Calibration files contain the ground truth transformations
- All paths in the JSON file should be relative to the `cooperative-vehicle-infrastructure` directory
- The dataset loader automatically handles point cloud sampling and normalization

## Running the Training Script

The training script supports various command-line arguments for model configuration and training parameters.

### Basic Usage

Train the vanilla PointNet model:
```bash
python src/train/train.py --model-type pointnet-vanilla
```

Train the CNN-based model:
```bash
python src/train/train.py --model-type pointnet-cnn
```

### Available Arguments

- `--model-type`: Model architecture to use [`pointnet-vanilla`, `pointnet-cnn`]
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.003)
- `--train-split`: Fraction of data for training (default: 0.8)
- `--hidden-size`: Hidden layer size (default: 1024)
- `--num-points`: Number of points per point cloud (default: 2048)
- `--batch-norm`: Enable batch normalization
- `--json-path`: Path to dataset JSON file
- `--base-dir`: Base directory for dataset files

### Example with Full Configuration

```bash
python src/train/train.py \
    --model-type pointnet-cnn \
    --epochs 20 \
    --batch-size 8 \
    --lr 0.001 \
    --hidden-size 1024 \
    --num-points 2048 \
    --batch-norm \
    --json-path data/cooperative-vehicle-infrastructure/cooperative/data_info_new.json \
    --base-dir data/cooperative-vehicle-infrastructure
```

## Model Details

### PointNet Vanilla
- Architecture: Series of 1D convolutions with optional batch normalization
- Feature extraction: Direct convolution on point cloud data
- Output: 9 parameters (6 for rotation, 3 for translation)
- Best for: Simpler point cloud registration tasks

### PointNet CNN
- Architecture: Dedicated CNN feature extractor followed by MLP
- Feature extraction: Enhanced through `PointFeatCNN` module
- Output: 9 parameters (6 for rotation, 3 for translation)
- Best for: Complex point cloud registration scenarios

## Testing the Model

After training, evaluate your model using:

```bash
python test/run_saved_model.py
```

This will:
1. Load the best model from `saved_models/pointnet_best_model.pth`
2. Run inference on the test dataset
3. Print evaluation metrics:
   - Average Loss
   - Rotation Error (degrees)
   - Translation Error

## Training Process Details

The training script:
1. Automatically selects the appropriate device (MPS, CUDA, or CPU)
2. Uses a combination of RMSD and ICP losses
3. Implements learning rate scheduling:
   - Warm-up period: 5 epochs
   - Cosine annealing schedule
4. Saves the best model based on test loss
5. Prints detailed metrics each epoch:
   - Training Loss
   - Test Loss
   - Average Rotation Error
   - Average Translation Error
   - Current Learning Rate

## Optimization Strategy

### Optimizer Configuration
- Implemented AdamW optimizer with the following parameters:
  ```python
  optimizer = optim.AdamW(model.parameters(),
                         lr=lr,
                         weight_decay=0.05,
                         betas=(0.9, 0.999),
                         eps=1e-8)
  ```
- Weight decay (0.05) helps prevent overfitting
- Custom betas for better convergence on point cloud data

### Learning Rate Scheduling
Implemented a two-phase learning rate schedule:

1. **Warm-up Phase**
   ```python
   warmup_epochs = 5
   warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
       optimizer,
       start_factor=0.1,
       total_iters=warmup_epochs
   )
   ```
   - Gradually increases learning rate for first 5 epochs
   - Helps stabilize early training

2. **Cosine Annealing**
   ```python
   scheduler = CosineAnnealingLR(
       optimizer,
       T_max=epochs,
       eta_min=lr*0.1
   )
   ```
   - Smoothly decreases learning rate after warm-up
   - T_max set to total number of epochs
   - Minimum learning rate set to 10% of initial lr

### Gradient Clipping
- Implemented to prevent exploding gradients:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
  ```
- Helps maintain stable training especially with point cloud data

### Training Loop Integration
```python
# During training
if epoch < warmup_epochs:
    warmup_scheduler.step()
else:
    scheduler.step()
```

This optimization strategy was found to be particularly effective for:
- Handling varying point cloud densities
- Managing the complex geometry of infrastructure-vehicle registration
- Stabilizing training with different loss functions
- Improving convergence speed and final accuracy

## Performance Notes
- Warm-up phase helps prevent early training instability
- Cosine annealing provides better final convergence compared to step scheduling
- AdamW's weight decay particularly helpful for regularization
- Gradient clipping essential for handling outliers in point cloud data

