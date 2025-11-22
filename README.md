# Brain Tumor Segmentation using BraTS2020 Dataset

A complete deep learning pipeline for automatic brain tumor segmentation from MRI scans. This project uses the BraTS2020 challenge dataset and implements a U-Net architecture with MONAI for accurate tumor detection and 3D reconstruction.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [3D Visualization](#3d-visualization)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Brain tumor segmentation is a critical task in medical imaging that helps radiologists and clinicians identify tumor regions in MRI scans. Manual segmentation is time-consuming and subject to inter-observer variability. This project automates the segmentation process using deep learning.

The pipeline includes:

- Data loading and preprocessing from HDF5 slice files
- Training a 2D U-Net model on individual MRI slices
- Validation with Dice score evaluation
- 3D volume reconstruction from predicted slices
- Interactive 3D tumor visualization using marching cubes

## Dataset

This project uses the BraTS2020 (Brain Tumor Segmentation Challenge 2020) dataset available on Kaggle. The dataset contains multi-modal MRI scans stored as HDF5 slice files.

**Dataset source:** [BraTS2020 Training Data on Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

Each HDF5 file contains:
- `image`: MRI slice (may contain multiple modalities)
- `mask`: Segmentation mask with tumor labels

The original BraTS labels include:
- 0: Background
- 1: Necrotic and non-enhancing tumor
- 2: Peritumoral edema
- 4: GD-enhancing tumor

For this project, we convert to binary segmentation (tumor vs. background).

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU

### Dependencies

Install the required packages:

```bash
pip install torch torchvision
pip install monai
pip install h5py
pip install numpy
pip install pandas
pip install matplotlib
pip install scikit-image
pip install scikit-learn
pip install opencv-python
pip install tqdm
pip install pyvista
```

Or install all at once:

```bash
pip install torch torchvision monai h5py numpy pandas matplotlib scikit-image scikit-learn opencv-python tqdm pyvista
```

### Kaggle Environment

If running on Kaggle, the notebook handles dependency installation automatically. You may need to fix protobuf compatibility issues:

```bash
pip uninstall -y tensorboard tensorboard-data-server protobuf
pip install protobuf==3.20.3
```

## Project Structure

```
brain-tumor-segmentation/
│
├── brats_segmentation.ipynb    # Main Jupyter notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── outputs/                    # Generated outputs
│   ├── brats_unet_model.pth   # Trained model weights
│   ├── training_curves.png    # Loss and Dice plots
│   ├── prediction_*.png       # Sample predictions
│   ├── mri_volume.npy         # Reconstructed MRI volume
│   ├── gt_mask_volume.npy     # Ground truth 3D mask
│   ├── pred_mask_volume.npy   # Predicted 3D mask
│   └── *.stl                  # 3D mesh files
│
└── data/                       # Dataset location (not included)
    └── *.h5                    # BraTS HDF5 slice files
```

## Usage



### Running Locally

1. Download the BraTS2020 dataset from Kaggle
2. Update the `DATA_PATH` variable to point to your data location
3. Run the Jupyter notebook:

```bash
jupyter notebook brats_segmentation.ipynb
```

### Training Configuration

You can adjust these parameters based on your hardware:

```python
BATCH_SIZE = 8          # Reduce if running out of GPU memory
NUM_EPOCHS = 10         # Increase for better results
NUM_WORKERS = 2         # Data loading workers
TARGET_SIZE = (128, 128) # Image resize dimensions
```

## Model Architecture

The segmentation model is a 2D U-Net implemented using MONAI. U-Net is a convolutional neural network designed for biomedical image segmentation that uses skip connections to combine low-level and high-level features.

![U-Net Architecture](./Images/u-net-architecture.png)



### Architecture Details

| Parameter | Value |
|-----------|-------|
| Spatial dimensions | 2D |
| Input channels | 1 |
| Output channels | 2 (background, tumor) |
| Feature channels | 16, 32, 64, 128 |
| Downsampling strides | 2, 2, 2 |
| Residual units | 2 per level |
| Dropout | 0.2 |
| Total parameters | ~470,000 |

### Loss Function

The model uses a combined loss function:

```
Total Loss = CrossEntropy Loss + Dice Loss
```

- CrossEntropy handles pixel-wise classification
- Dice Loss optimizes region overlap between prediction and ground truth

### Optimizer

- Adam optimizer with learning rate 1e-3
- Weight decay 1e-5 for regularization
- ReduceLROnPlateau scheduler (factor 0.5, patience 2)

## Results

Training on the full BraTS2020 dataset for 10 epochs typically achieves:

| Metric | Value |
|--------|-------|
| Training Loss | ~0.3 - 0.5 |
| Validation Loss | ~0.4 - 0.6 |
| Validation Dice Score | 0.75 - 0.85 |

Results may vary based on the specific data split and training configuration.

### Sample Predictions

The notebook generates visualization comparing:
- Input MRI slice
- Ground truth segmentation
- Model prediction
- Overlay comparison

## 3D Visualization

After training, the pipeline reconstructs 3D volumes by stacking 2D slice predictions. The marching cubes algorithm extracts a surface mesh from the binary segmentation volume.

### Visualization Options

1. **Orthogonal Slices**: Axial, coronal, and sagittal views through the volume center
2. **3D Surface Mesh**: Interactive visualization using matplotlib or PyVista
3. **Export**: Save meshes as STL or OBJ files for use in other software

### Generating 3D Mesh

```python
from skimage import measure

# Extract surface mesh
verts, faces, _, _ = measure.marching_cubes(binary_volume, level=0.5)

# Save as STL (requires numpy-stl)
# Or save as OBJ for universal compatibility
```

## Limitations

- The model processes 2D slices independently, which may miss 3D context
- Binary segmentation does not distinguish between tumor sub-regions
- Performance depends on the quality and consistency of the training data
- Training on full dataset requires significant GPU memory and time

## Future Improvements

- Implement 3D U-Net for volumetric segmentation
- Multi-class segmentation for tumor sub-regions
- Data augmentation for improved generalization
- Attention mechanisms for better feature selection
- Post-processing with conditional random fields



## References

1. Menze, B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE Transactions on Medical Imaging, 2015.

2. Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.

3. MONAI Consortium. "MONAI: Medical Open Network for AI." https://monai.io/

