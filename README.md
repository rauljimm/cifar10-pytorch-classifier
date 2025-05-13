# 🖼️ CIFAR-10 Image Classifier 🤖

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📝 Description

A deep learning image classifier for the CIFAR-10 dataset built with PyTorch and GPU acceleration. This project implements a Convolutional Neural Network (CNN) that classifies images into 10 different categories with approximately 78.9% accuracy.

## 🌟 Features

- 🔥 **GPU Acceleration** - Utilizes CUDA for fast training and inference
- 🧠 **Custom CNN Architecture** - Optimized for CIFAR-10 classification
- 📊 **Comprehensive Evaluation** - Includes confusion matrix and per-class metrics
- 📈 **Data Augmentation** - Implements transformations to improve model generalization
- 🔍 **Visualization Tools** - View training progress and misclassified examples

## 🛠️ Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/cifar10-image-classifier.git
cd cifar10-image-classifier

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Training the model

```bash
python train.py
```

This will:
- Download the CIFAR-10 dataset
- Train the model for 15 epochs
- Save the trained model to `models/cifar10_model.pth`
- Save the best performing model to `models/best_model.pth`
- Generate training metrics visualization in the `results/` folder

### Evaluating the model

```bash
python evaluate.py
```

This will:
- Load the best model from `models/best_model.pth`
- Test it on the CIFAR-10 test set
- Generate a confusion matrix and classification report
- Display examples of misclassified images

### Making predictions

For a single image:
```bash
python predict.py image --image path/to/your/image.jpg
```

For multiple images in a directory:
```bash
python predict.py directory --directory path/to/images/ --limit 20
```

## 📊 Performance

The model achieves approximately 78.9% accuracy on the CIFAR-10 test set:

| Class       | Accuracy |
|-------------|----------|
| airplane    | 81.0%    |
| automobile  | 93.1%    |
| bird        | 62.6%    |
| cat         | 50.1%    |
| deer        | 74.4%    |
| dog         | 80.7%    |
| frog        | 82.6%    |
| horse       | 88.1%    |
| ship        | 86.5%    |
| truck       | 89.8%    |

## 📁 Project Structure

```
image_classifier_en/
├── model.py           # CNN architecture definition
├── train.py           # Training script with data augmentation
├── evaluate.py        # Evaluation script with metrics
├── predict.py         # Prediction script for new images
├── models/            # Saved model weights
└── results/           # Visualizations and metrics
```

## 📷 CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes:
- ✈️ airplane
- 🚗 automobile
- 🐦 bird
- 🐱 cat
- 🦌 deer
- 🐶 dog
- 🐸 frog
- 🐎 horse
- 🚢 ship
- 🚚 truck

## 🔧 Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- scikit-learn
- seaborn
- Pillow
- CUDA-compatible GPU (recommended)

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- Thanks to the PyTorch team for the amazing deep learning framework 