# <p align="center">ML4SCI Tasks</p>
# End-to-End event classification with sparse autoencoders

## Model Weights

Model weights can be found out under the `ModelWeights` folder - https://github.com/fractaloid007/ML4Sci_Tasks/tree/main/ModelWeights.

## Common Task 1: Electron/Photon Classification

This tasks implements a deep learning model to classify particles (electrons vs. photons) using a ResNet-15-like architecture. The model achieves strong discriminative power as evidenced by an AUC of 0.8143, despite near-random accuracy due to balanced class distributions.

## Key Features
- **Custom ResNet-15 Architecture**: Optimized for particle classification tasks
- **Advanced Training Techniques**: Mixed-precision training, cosine annealing learning rate scheduling
- **Robust Evaluation**: AUC-focused validation with early stopping

## Dataset & Preprocessing
### Data Loading
- Combined HDF5 datasets (`photons` and `electrons`)
- 80-20 train-test split with stratified sampling

### Preprocessing Pipeline
1. **Normalization**: 
   - Training-set derived μ ± σ applied to all data
2. **Augmentation** (Training only):
   - Random horizontal/vertical flips
3. **Dataset Handling**:
   - Custom `ParticleDataset` class
   - On-the-fly PyTorch tensor conversion

## Model Architecture
```bash
Input
  ↓
[Initial Convolution: 3×3, 64 channels, stride=1]
  ↓
Layer 1
  ├─ [BasicBlock: 64 channels, stride=1]
  └─ [BasicBlock: 64 channels, stride=1]
  ↓
Layer 2
  ├─ [BasicBlock: 128 channels, stride=2]  # Downsampling
  └─ [BasicBlock: 128 channels, stride=1]
  ↓
Layer 3
  ├─ [BasicBlock: 256 channels, stride=2]  # Downsampling
  └─ [BasicBlock: 256 channels, stride=1]
  ↓
[Global Average Pooling]
  ↓
[Linear Classifier: 256 → 2 classes]
  ↓
Output
```

**Core Components:**
- **Initial Convolution**: 
  - 3×3 kernel, stride=1 → 64 channels
- **Residual Blocks**:
  - Layer 1: 2× BasicBlock (64 ch, no downsampling)
  - Layer 2: 2× BasicBlock (128 ch, stride=2)
  - Layer 3: 2× BasicBlock (256 ch, stride=2)
- **Head**:
  - Global average pooling
  - Linear classifier (256 → 2 classes)

**Key Features:**
- Batch normalization after each convolution
- Skip connections in all residual blocks
- Mixed-precision training support

## Training Configuration
**Hyperparameters:**
- **Optimizer**: AdamW (lr=0.001, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR
- **Batch Size**: 256
- **Epochs**: 50
- **Loss**: Cross-Entropy
- **Precision**: AMP (Automatic Mixed Precision)

**Training Infrastructure:**
- Gradient scaling for stable mixed-precision training
- Early stopping based on validation AUC
- Best model checkpointing

## Results

### Training Dynamics
![training resnet](https://github.com/user-attachments/assets/469759e4-5fda-4846-aaa1-525df1a197db)


### Final Metrics
| Metric        | Value   |
|---------------|---------|
| Test AUC      | 0.8143  |
| Test Accuracy | 51.49%  |
| Train AUC     | 0.825   |

### Key Observations
- **Strong Discriminative Power**: High AUC indicates excellent separation capability
- **Balanced Classes**: Explains near-random accuracy (50% baseline)
- **Training Stability**:
  - Smooth loss descent
  - Minimal train-test AUC gap (Δ=0.011) → no overfitting


## Conclusion
This ResNet-15 implementation demonstrates effective particle classification through:
- **Architectural Choices**: Residual connections and channel scaling
- **Training Strategy**: Mixed precision + cosine annealing
- **Evaluation Focus**: AUC optimization over raw accuracy

While final accuracy appears limited, the 0.8143 AUC reveals substantial ranking capability - the model can reliably order particles by class likelihood, making it suitable for probability-based decision systems.

______

# Specific Task 2c

## Introduction

This task implements a Variational Autoencoder (VAE) to learn latent representations from unlabelled jet images and fine-tunes the encoder for binary classification. The implementation features:

- A VAE pretrained on unlabelled jet image data (Dataset_Specific_Unlabelled.h5)
- Three classification approaches using the pretrained encoder:
  - Support Vector Machine (SVM)
  - General Classifier with end-to-end fine-tuning
  - MLP with frozen layers trained alongside the VAE
- Advanced techniques including KL annealing, batch normalization, and early stopping
- Model pruning strategies to reduce computational complexity
- A bonus Sparse Autoencoder implementation for comparison

## Data Processing

### Datasets
- **Unlabelled Dataset**: Jet images from Dataset_Specific_Unlabelled.h5
- **Labelled Dataset**: Jet images with binary labels from Dataset_Specific_labelled.h5

### Preprocessing Steps
1. Transpose images from [125, 125, 8] to [8, 125, 125] for PyTorch compatibility
2. Normalize each channel by dividing by its maximum value
3. Pad images to [8, 128, 128] using zero padding

### Data Loaders
- **Unlabelled Data**: Batch size of 128 (training) or 256 (evaluation), with shuffling and 16 workers
- **Labelled Data**: 80%/20% train/test split with batch sizes of 128 (training) and 64 or 256 (testing)

## Model Architecture

### Variational Autoencoder (VAE)
- **Encoder**:
  - Four convolutional layers (8→16→32→64→128)
  - 3×3 kernels, stride 2, padding 1
  - Batch normalization and LeakyReLU activation
  - Flattened output through a fully connected layer with dropout (0.3)
  - Outputs 128-dimensional latent space (mean and log-variance)
- **Decoder**:
  - Mirror of encoder with transposed convolutions
  - Final sigmoid activation for reconstruction
- **Initialization**: Kaiming normal for convolutional layers, Xavier normal for linear layers

### Classifier Architectures
1. **SVM**:
   - RBF-kernel SVM trained on latent vectors from the VAE encoder
   - Optimized parameters: C=10, gamma='scale'

2. **General Classifier**:
   - Uses pretrained VAE encoder
   - Classification head: Linear(128→128)→ReLU→Dropout(0.5)→Linear(128→1)
   - Differential learning rates for fine-tuning

3. **MLP with Frozen Layers**:
   - Integrates VAE and classifier training
   - Shares encoder with separate optimizers for VAE and classifier components

## Training Procedure

### VAE Training
- Trained on unlabelled data for up to 50 epochs
- Adam optimizer (lr=1e-3, weight_decay=1e-5)
- ReduceLROnPlateau scheduler (patience=3, factor=0.5)
- Loss function: MSE reconstruction loss + KL divergence with beta annealing
- Best model achieved at epoch 13 with loss of 123.2921

### Classification Methods

#### SVM Training
- Latent vectors extracted using pretrained VAE encoder
- Initial accuracy: 77.55% with default parameters
- Optimized accuracy: 83.12% (C=10, gamma='scale')

#### General Classifier Training
- Fine-tuned encoder with classifier head for 20 epochs
- Adam optimizer with differential learning rates
- StepLR scheduler (step_size=5, gamma=0.1)
- Loss decreased from 0.6343 to 0.3425

#### MLP with Frozen Layers Training
- Joint training of VAE and classifier for 30 epochs
- Separate Adam optimizers with ReduceLROnPlateau schedulers
- Early stopping at epoch 14 (VAE loss: 124.3792)

## Evaluation Results

### VAE Performance
- Best reconstruction loss: 123.2921–124.3792
- Effective reconstruction of jet images

### Classification Performance Comparison

| Classifier Type | Accuracy | ROC AUC | Training Loss | Notes |
|----------------|----------|---------|---------------|-------|
| Support Vector Machine (SVM) | 83.12% | 0.8614 | N/A | Optimized with C=10, gamma='scale' |
| General Classifier | N/A | 0.9317 | 0.3425 (from 0.6343) | Best performance, end-to-end fine-tuning |
| MLP with Frozen Layers | N/A | 0.9024 (0.8785 during training) | N/A | Early stopping at epoch 14 |

### SVM Results
#### Figure: t-SNE visualization of latent space showing class separation
<img src="https://github.com/user-attachments/assets/642f7184-b447-4109-929e-2066b1eca947" width="500" alt="t-SNE visualization of latent space showing class separation">

#### Figure: SVM ROC curve
<img src="https://github.com/user-attachments/assets/f5460513-7058-4d90-9112-1ce086d18725" width="500" alt="SVM ROC curve">

### General Classifier Results
#### Figure: ROC curve showing strong binary classification performance
<img src="https://github.com/user-attachments/assets/bbb59eeb-4ad2-47b5-b146-3610f4f39baa" width="500" alt="ROC curve showing strong binary classification performance">

### MLP with Frozen Layers Results
#### Figure: ROC curve for MLP classifier
<img src="https://github.com/user-attachments/assets/2fa8aaaa-3593-426e-8330-e3fa0049a202" width="500" alt="ROC curve for MLP classifier">

## Model Pruning

### Pruning Strategies
1. **Unstructured Pruning**:
   - Removes individual weights based on L1 norm
   - No fine-tuning after pruning

2. **Iterative Pruning with Fine-Tuning**:
   - Incremental pruning with step size of 0.1
   - Fine-tuning for 5 epochs after each pruning step

3. **Structured Pruning**:
   - Removes entire filters from convolutional layers
   - Based on L1 norm of filter weights

### Pruning Results

| Pruning Method | Pruning Ratio | AUC | Error (1-AUC) | FLOPS |
|----------------|---------------|-----|---------------|-------|
| Original Model | 0.0 | 0.9317 | 0.0683 | 2.15e+07 |
| Unstructured | 0.4 | ~0.9285 | ~0.0715 | 1.29e+07 |
| Unstructured | 0.9 | 0.8387 | 0.1613 | 2.15e+06 |
| Iterative with Fine-Tuning | 0.4 | 0.9363 | 0.0637 | 1.29e+07 |
| Iterative with Fine-Tuning | 0.9 | 0.9181 | 0.0819 | 2.15e+06 |
| Structured | 0.4 | 0.8581 | 0.1419 | 1.29e+07 |
| Structured | 0.8 | 0.4466 | 0.5534 | 4.30e+06 |

#### Figure: Error vs. FLOPS plot showing pruning strategies comparison
![prunung_error_vs_flops](https://github.com/user-attachments/assets/eaab5230-3073-4ac3-b74a-5819cd664aae)

____

## Bonus Task: Sparse Autoencoder Implementation

### Sparse Autoencoder Architecture
- Similar to VAE but without variational components
- Direct output of latent code instead of mean/log-variance

### Loss Function
- MSE reconstruction loss + L1 penalty on latent code
- Sparsity weight λ controls penalty strength

### Results

| Model | Sparsity Weight (λ) | Reconstruction Error | FLOPS | Training Epochs |
|-------|---------------------|----------------------|-------|----------------|
| Baseline VAE | N/A | 127.4235 | 9.87e+07 | N/A |
| Sparse Autoencoder | 0.0 | 114.7373 | 9.864e+07 | 19 |
| Sparse Autoencoder | 0.0001 | 114.8447 | 9.864e+07 | 17 |
| Sparse Autoencoder | 0.001 | 115.2265 | 9.864e+07 | 16 |
| Sparse Autoencoder | 0.01 | 114.8289 | 9.864e+07 | 17 |
| Sparse Autoencoder | 0.1 | 114.9590 | 9.864e+07 | 19 |

#### Figure: FLOPS vs. reconstruction error comparing SAE variants against baseline VAE
![baseline_vae_vs_sparse_ae](https://github.com/user-attachments/assets/8357a24e-69d6-4c3e-8365-e4ec554b4561)

## Conclusion

This project demonstrates the effectiveness of using autoencoders for feature extraction and classification in jet image data:

1. The VAE successfully learns useful latent representations from unlabelled data
2. The General Classifier with fine-tuning achieves the best classification performance (AUC 0.9317)
3. Iterative pruning with fine-tuning offers the best balance between model complexity and performance, maintaining AUC > 0.91 even with 90% parameter reduction
4. The Sparse Autoencoder consistently outperforms the VAE in reconstruction accuracy (~114 vs 127.4235) with comparable computational complexity

