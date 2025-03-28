# Deep_learning_Assignment3_Object-Detection

# 🚀 YOLOv11 Object Detection on Custom Dataset
This repository contains the code and documentation for a Google Colab lab assignment implementing YOLOv11 for real-time object detection using a custom dataset. The project covers environment setup, dataset acquisition and preprocessing via Roboflow, model training, inference, and performance evaluation.

## 📌 Overview
This assignment focuses on implementing and evaluating a YOLOv11 model for object detection. The project is executed on Google Colab, where the required libraries (Roboflow, Ultralytics, PyTorch, OpenCV, etc.) are installed. The custom dataset is acquired via Roboflow, preprocessed, and then used to train a YOLOv11 model. Finally, the trained model is tested on unseen data, and its performance is evaluated using metrics such as Mean Average Precision (mAP), Precision, Recall, and F1 Score.

## 📂 Dataset
### Dataset Acquisition
- **Source:** A custom dataset available on Roboflow.
- **Acquisition Method:**
  - The dataset is downloaded using the Roboflow API.
  - Example:
    ```python
    !pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key="jBRsMtfCa6QP67xjDuwU")
    project = rf.workspace("roboflow-58fyf").project("rock-paper-scissors-sxsw")
    version = project.version(14)
    dataset = version.download("yolov11")
    ```

### Dataset Structure and Characteristics
#### Structure:
After downloading, the dataset is organized in a YOLO-friendly format with separate directories for training, validation, and testing:
- `train/`: Contains training images and YOLO-format labels.
- `valid/`: Contains validation images and labels.
- `test/`: Contains test images for inference.

#### Characteristics:
- **Classes:** Custom object categories as per the dataset.
- **Annotations:** Bounding boxes with class labels in YOLO format (normalized coordinates).
- **Preprocessing:** Minimal preprocessing is required; however, verifying image sizes and annotations is crucial for the training pipeline.

## ⚙️ Methodology
### Environment Setup and Installation
- Set up your Google Colab environment by installing required libraries such as Roboflow, Ultralytics, PyTorch, and OpenCV.
- Verify that the environment is correctly configured for YOLOv11.

### Dataset Preparation & Preprocessing
- Download the custom dataset in YOLOv11 format using the Roboflow API.
- Verify the dataset structure (`train`, `valid`, `test`) and ensure the annotations are correctly formatted.
- Conduct a data quality check by reviewing sample images and labels.

### Model Training
- Initialize the YOLOv11 model using the pre-trained weights (e.g., `yolo11n.pt`).
- Configure training parameters, including the number of epochs (e.g., 100), batch size, learning rate, and input image size.
- Monitor training progress via loss, mAP, precision, and recall.
- Save the best-performing model based on validation metrics.

### Model Inference and Evaluation
- Load the best saved model weights.
- Run inference on unseen test images and save the results.
- Visualize the inference outputs (bounding boxes and confidence scores) using tools like matplotlib or PIL.
- Evaluate performance using metrics such as mAP (at IoU 0.5 and 0.5–0.95), precision, recall, and F1 score.
