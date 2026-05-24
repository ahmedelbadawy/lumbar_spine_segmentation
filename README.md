# Automated Segmentation of Lumbar Spine Structures

A deep-learning model for the automated segmentation of key lumbar spine structures from axial MRI, built to support the diagnosis and assessment of **lumbar spinal stenosis (LSS)**. The model is a **U-Net with Squeeze-and-Excitation (SE) attention**, tuned through **Weights & Biases (W&B) hyperparameter sweeps**, and segments five classes: background, intervertebral disc (IVD), posterior element (PE), thecal sac (TS), and the area between the anterior and posterior vertebrae (AAP).

The best model achieves a mean Dice score of **92.78% on the held-out test set**, outperforming the published benchmark on this dataset.

## Motivation

LSS is a common condition caused by narrowing of the spinal canal, and assessing it requires measuring anatomical structures on MRI. Manual measurement is slow and varies between observers. This project automates the segmentation step so those measurements can be made faster and more consistently.

## Dataset

- **Source:** [Lumbar Spine MRI Dataset (Mendeley Data)](https://data.mendeley.com/datasets/k57fr854j2/2) — 515 patients, 48,345 images (T1- and T2-weighted).
- **Subset used:** T1-weighted axial images in PNG format (the ground-truth masks were annotated on T1).
- **Classes (5):** background, IVD, PE, TS, AAP.
- **Split:** 60% train / 20% validation / 20% test, split **per patient** to prevent data leakage.

## Method

### Preprocessing & Augmentation
- All images resized to **256×256**.
- Geometric augmentation (via Albumentations) applied with probability 0.5: rotations (±10°), scaling (95–105%), translations (±5% on each axis), followed by normalization.

### Model
A U-Net implemented in PyTorch with:
- **Double-convolution blocks** with LeakyReLU activation and BatchNorm.
- **Squeeze-and-Excitation attention**, implemented in three selectable variants: channel-wise, spatial, and combined channel+spatial.
- Configurable channel widths and dropout.

### Training
- **Loss:** Dice Loss (also compared against Dice + Focal Loss).
- **Optimizer:** AdamW (compared against Adam).
- **Regularization:** dropout, BatchNorm, decoupled weight decay, and max-norm weight clipping.
- **Scheduler:** ReduceLROnPlateau on the validation Dice score.
- **Other:** batch size 16, 50 epochs, automatic mixed precision (AMP), Dice score (via MONAI) as the primary metric.

### Hyperparameter Search
Tuned via **W&B random-search sweeps (25 runs)** over learning rate, optimizer, scheduler factor/patience, weighted loss, loss function, channel sizes, SE variant, max-norm clipping, and dropout. The best configuration:

| Hyperparameter   | Value                    |
|------------------|--------------------------|
| Channel sizes    | 64 – 1024                |
| Dropout          | 0.3                      |
| Learning rate    | 3.18 × 10⁻⁴              |
| Loss             | Dice Loss                |
| Max-norm clip    | 5                        |
| Optimizer        | AdamW                    |
| Scheduler factor | 0.1066 (patience 8)      |
| SE block         | Channel-wise             |
| Class weighting  | None                     |

## Results

Mean Dice score: **93.0% (train) · 92.9% (val) · 92.78% (test)**.

Per-class Dice on the test split:

| Class | Dice (test) |
|-------|:-----------:|
| IVD   | 98.2%       |
| PE    | 94.6%       |
| TS    | 94.6%       |
| AAP   | 83.7%       |

Performance is consistent across train/val/test, indicating good generalization without overfitting. The lower AAP score reflects the difficulty of segmenting smaller structures.

## Notebook Structure

The full pipeline is in `Mohamed_Ahmed_final_code.ipynb`:
0. Boilerplate / setup
1. Data loading (custom PyTorch `Dataset`, patient-wise splitting)
2. Preprocessing and augmentation
3. Model definition (SE blocks + U-Net)
4. Hyperparameters
5. Training and validation engine + W&B sweep
6. Inference on the test split
7. Output visualization

## Getting Started

### Prerequisites
- Python 3.x and a CUDA-capable GPU (recommended)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedelbadawy/lumbar_spine_segmentation.git
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the [Lumbar Spine MRI Dataset](https://data.mendeley.com/datasets/k57fr854j2/2) and set the data directory path in the notebook.
4. (Optional) Log in to Weights & Biases to run hyperparameter sweeps:
   ```bash
   wandb login
   ```
5. Open and run the notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## License
[Add your license here]
