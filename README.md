# ai-tuberculosis-detection

🫁 AI-Powered Tuberculosis Detection System  An end-to-end deep learning application that detects Tuberculosis from chest X-ray images using Convolutional Neural Networks (CNNs). The system validates medical images before prediction and provides accurate classification through a user-friendly web interface.

📂 Dataset

- Source: Public Chest X-Ray Tuberculosis Dataset
- Classes:
  - Tuberculosis Positive
  - Normal
- Image Format: PNG / JPEG
- Preprocessing:
  - Resizing to 224×224
  - Normalization
  - Data Augmentation

🔍 Image Validation Module

Before prediction, the system verifies whether the uploaded image is a valid chest X-ray using:
- Image dimension checks
- Histogram-based grayscale analysis
- Edge density validation

This prevents incorrect inputs such as CT scans or non-medical images.

🧠 Model Architecture

- Base Model: Custom CNN
- Layers:
  - Convolution + ReLU
  - MaxPooling
  - Dropout
  - Fully Connected Layers
- Loss Function: Binary Crossentropy
- Optimizer: Adam

📊 Results

- Training Accuracy: ~95%
- Validation Accuracy: ~92%
- Model generalizes well with minimal overfitting.

🌐 Web Application

- Framework: Flask
- Features:
  - Image upload
  - Validation check
  - Real-time TB prediction

Required libraries
tensorflow
keras
opencv-python
numpy
pandas
flask
matplotlib
