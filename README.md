# Landmark-Retrieval: Campus Image Classification using VGG-19

## Overview
This project implements a deep learning-based classification pipeline that identifies landmarks across the KLE Tech campus. The dataset includes 15,268 images divided into 30 classes covering circles, buildings, and other campus sites. A pre-trained **VGG-19** model was fine-tuned to achieve robust feature extraction and classification performance.

## Dataset
- **Source:** KLE Tech Campus Landmark Dataset
- **Images:** ~15,268
- **Classes:** 30 (e.g., Circles, Blocks, Buildings)
- **Data Split:** 80% training, 10% validation, 10% test
- **Preprocessing:** Normalization, Resizing (224x224), Augmentation (flipping, rotation, color jitter)

## Model Architecture
The model uses **VGG-19**, a 19-layer convolutional neural network pre-trained on ImageNet.  
VGG-19 is selected for its:
- Consistent 3x3 convolution filters and deep hierarchical representation.
- Strong generalization for small datasets using **transfer learning**.
- Proven performance in similar classification tasks with accuracy above 90%.

### Modifications
- Replaced top classifier layers with custom dense layers optimized for 30-class output.
- Used **ReLU** activation and **Softmax** output layer.
- Implemented **Dropout** and **Batch Normalization** to prevent overfitting.
- 
## Implementation Highlights
- Built with **TensorFlow** and **Keras** for model training and evaluation.
- Android application deployment via **TensorFlow Lite (TFLite)** for on-device inference.
- Flask backend service used for API-based testing and remote predictions.

## Android Deployment
The application classifies captured campus images and displays the identified landmark name.
- Android Studio + TensorFlow Lite integration.
- Optimized for low latency and mobile GPU environments.

**Visualization:**  
- Confusion Matrix for classification accuracy per class  
- Grad-CAM heatmaps for interpretability

## Tech Stack
**Core:** TensorFlow, Keras, Python  
**Deployment:** Android Studio, TensorFlow Lite  
**Tools:** OpenCV, NumPy, Matplotlib, Flask

## Future Scope
- Extend dataset for better generalization across lighting and seasonal variations.
- Experiment with lightweight architectures (e.g., MobileNet, EfficientNet).
- Implement geolocation tagging for image verification.



