# MNIST Classification with PyTorch

This repository contains a Jupyter Notebook designed to implement and train a neural network for MNIST digit classification using PyTorch. The notebook demonstrates the use of deep learning techniques for image classification.

## Features
- Implementation of MNIST classification using PyTorch.
- Utilization of `DataLoader` for efficient data processing.
- Training a deep learning model with optimization techniques.
- Activation functions and loss calculations for improving model performance.

## Methods Used
- **PyTorch**: Framework for defining and training deep learning models.
- **DataLoader**: Handles batch processing for efficient data feeding.
- **Optimization**: Uses gradient-based optimization (`torch.optim`).
- **Activation Functions**: Implements activation layers for better feature learning.
- **Loss Function**: Computes classification loss to guide model training.
- **Convolutional Neural Network (CNN)**: Used to extract spatial features from images.
- **Batch Normalization**: Applied to stabilize and accelerate training.
- **Dropout**: Used to prevent overfitting by randomly deactivating neurons.

## Model Accuracy
The final trained model achieves an accuracy of **98%** on the test dataset.

## Prerequisites
Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook or Google Colab
- Required Python libraries:
  ```sh
  pip install torch torchvision matplotlib numpy
  ```

## How to Use
1. Open the Jupyter Notebook (`3-MNIST_Classification.ipynb`) in Google Colab or a local Jupyter environment.
2. Ensure PyTorch and necessary dependencies are installed.
3. Run the notebook cells sequentially to train and evaluate the MNIST classifier.

