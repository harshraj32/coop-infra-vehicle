# PointNet Registration Project

This project implements a point cloud registration framework using PointNet‐based models. Two variants of the PointNet architecture are provided:

- **PointNet Vanilla:** The original PointNet architecture that uses a series of 1D convolutions followed by fully connected layers to predict 9 registration parameters (6 for rotation and 3 for translation).  
- **PointNet CNN:** An alternative architecture that employs a dedicated CNN feature extractor (`PointFeatCNN`) before the fully connected layers. This variant can be selected via a command‑line argument.

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
git clone https://github.com/yourusername/pointnet-registration.git
cd pointnet-registration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Prepare your dataset in the following structure:
```
data/
└── cooperative-vehicle-infrastructure/
    ├── cooperative/
    │   └── data_info_new.json
    ├── vehicle-side/
    └── infrastructure-side/
```

2. Ensure your JSON file contains the correct paths to point cloud files and calibration information.

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

## Additional Features

- **Automatic Device Selection:** Supports MPS (Mac), CUDA (NVIDIA), and CPU
- **Model Checkpointing:** Automatically saves best model during training
- **Flexible Loss Functions:** Implements RMSD, ICP, and combined losses
- **Data Augmentation:** Random point sampling for point clouds
- **Progress Monitoring:** Detailed training metrics printed each epoch

## Troubleshooting

Common issues and solutions:
1. "No module named 'src'": Ensure you're running from the project root directory
2. CUDA out of memory: Reduce batch size or number of points
3. NaN losses: Try reducing the learning rate or enabling batch normalization

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 