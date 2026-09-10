# PSO-ANN: Automatic License Plate Recognition using Particle Swarm Optimization

An Automatic License Plate Recognition (ALPR) pipeline that combines classical image-processing techniques for plate localization with an Artificial Neural Network (ANN) trained via Particle Swarm Optimization (PSO) for character recognition — achieving **98.65% accuracy** on the test set.


## Overview

Traditional OCR-based ALPR systems often struggle with noisy plates, inconsistent lighting, and regional plate format variation, and gradient-descent-trained networks can get stuck in local minima or converge slowly with limited plate data. This project addresses those issues with a hybrid approach:

- **Plate detection** using morphological filtering and contour-based localization
- **Character recognition** via an ANN whose weights are optimized with PSO instead of backpropagation, helping avoid local minima
- **Post-processing correction** using a custom character-mapping dictionary to resolve common OCR confusions (e.g., `K`/`4`, `V`/`Y`, `1`/`I`, `7`/`Z`)

## Key Features

- 🔍 Robust plate localization via grayscale conversion, bilateral filtering, edge detection, and morphological operations
- 🧠 PSO-optimized ANN classifier for character recognition (avoids gradient-descent local minima)
- 🧩 Character segmentation with bounding-box extraction
- 🔤 Dictionary-based post-processing to resolve alphanumeric look-alike confusions
- 🔒 Privacy-conscious design — only the plate region is processed, no driver/vehicle imagery retained

## System Architecture

## Dataset

The character recognizer is trained on the **MNIST** handwritten digit dataset, extended with custom alphabetic characters (A–Z) extracted from regional license plates.

| Metric | Value (Training Script) |
|---|---|
| Training Samples | 5,000 |
| Testing Samples | 1,000 |
| Swarm Size | 20 particles |
| Iterations | 30 |
| Accuracy Achieved | 98.6% |

## Results

| Metric | Score |
|---|---|
| Accuracy | 98.65% |
| Precision | 99.10% |
| Recall | 98.20% |
| F1 Score | 98.64% |
| AUC Score | 0.9997 |

### Model Comparison

| Model | Accuracy | F1 Score |
|---|---|---|
| Standard Backprop ANN | 89.2% | 88.5% |
| SSD + Tesseract | 94.1% | 93.8% |
| **Proposed PSO-ANN** | **98.6%** | **98.6%** |

### Error Analysis

Overall character error rate: **1.35%**. Remaining errors are attributed mainly to lighting conditions and plate damage rather than model architecture, with historical alphanumeric confusions (1/I, 7/Z, K/4, V/Y) resolved via the post-processing dictionary lookup.

## Applications

- **Smart city tolling** — automatic charging without requiring vehicles to stop
- **Law enforcement** — scanning for stolen or uninsured vehicles
- **Automated parking systems** — gate access based on real-time plate recognition

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Train the PSO-ANN model
python train_pso.py

# Run inference on a sample image
python detect.py --image path/to/vehicle.jpg
```

> Adjust script names/paths above to match your actual repository layout (e.g. `train_pso.py`).

## Privacy & Safety Considerations

- **Privacy-by-design**: only the license plate region is extracted and processed — no driver faces or vehicle interiors are retained, supporting compliance with regulations such as GDPR.
- **Reliability**: PSO-based training reduces "black box" false positives compared to standard backprop networks.
- **Robustness**: multi-step morphological processing improves tolerance to damaged plates and inconsistent lighting.

## Future Work

- Incorporate a memetic (local search) phase into PSO to speed up convergence
- Improve bounding-box detection for high-speed highway scenarios
- Extend support to multiple regional/international plate formats
- Optimize the model (pruning/quantization) for edge-device deployment

## References

Key references include Kennedy & Eberhart's original PSO paper, Du et al.'s ALPR state-of-the-art review, OpenCV, and several PSO-BP neural network studies for license plate recognition. See the full reference list in the paper.

