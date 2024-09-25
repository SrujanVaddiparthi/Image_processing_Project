Here's the README formatted in Markdown (`README.md`) for your **COVID-19 Radiography Classification with ResNet50** project:

---

# COVID-19 Radiography Classification with ResNet50

## Overview
This project utilizes a ResNet50-based neural network to classify COVID-19 infection from chest radiography images. The dataset includes images of COVID-19 positive cases, normal cases, and viral pneumonia cases. Advanced image preprocessing techniques, such as histogram equalization, were used to enhance the image quality and improve model performance.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing Techniques](#preprocessing-techniques)
- [Training & Validation](#training--validation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)

## Project Description
This project aims to classify COVID-19 infection from lung radiography images using a ResNet50 neural network. The model was trained on a dataset comprising 12,000 images and achieved a training accuracy of 97% and a validation accuracy of 96%. Image enhancement techniques like histogram equalization were employed to improve the contrast of the images, significantly impacting the model's performance.

## Dataset
The dataset consists of chest x-ray images categorized as COVID-19, normal, and pneumonia cases. It was collected from various sources, including Kaggle, GitHub, and research papers.

- **COVID-19 Cases:** 3,616 images
- **Normal Cases:** 10,192 images
- **Pneumonia Cases:** 1,345 images

## Model Architecture
The model utilizes the ResNet50 architecture, a deep convolutional neural network that effectively handles complex image classification tasks. Transfer learning was used to leverage pre-trained weights due to limited computational resources.

- **Base Model:** ResNet50 with pre-trained weights
- **Top Layers:** Global Average Pooling, Dense Layers
- **Activation Function:** Softmax for classification

## Preprocessing Techniques
Preprocessing steps were crucial to enhance the quality of the dataset and improve model accuracy:

- **Grayscale Conversion:** Simplified the image data by reducing it to a single channel.
- **Histogram Equalization:** Enhanced contrast in radiography images, making it easier for the model to detect features.
- **Normalization:** Scaled pixel values to a range of 0 to 1 for consistent model input.

## Training & Validation
The model was trained for 10 epochs with an 80-20 train-validation split.

- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 32
- **Learning Rate:** 0.001

## Results
The model performed well on the validation set:

- **Training Accuracy:** 97%
- **Validation Accuracy:** 96%
- **Training Loss:** [Add value here]
- **Validation Loss:** [Add value here]

## How to Run
1. **Clone the repository:**
    ```bash
    git clone https://github.com/SrujanVaddiparthi/Image_processing_Project.git
    cd Image_processing_Project
    ```
2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the training script:**
    ```bash
    python train.py
    ```
4. **Evaluate the model:**
    ```bash
    python evaluate.py
    ```

## Requirements
- Python 3.x
- NumPy
- TensorFlow
- OpenCV
- Keras
- Matplotlib (for visualizations)
