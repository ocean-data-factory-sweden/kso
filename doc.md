# Guide for Processing Annotated Image Data Using Jupyter Notebooks

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Available Notebooks](#available-notebooks)
5. [Data Ingestion Process](#data-ingestion-process)
6. [Training Models](#training-models)
7. [Running Inference](#running-inference)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

This guide provides step-by-step instructions for researchers working with annotated image data. It covers how to prepare your data, use our Jupyter notebooks for ingestion, train models, and run inference. The platform is designed to make the machine learning workflow accessible while maintaining reproducibility and security.

## Prerequisites

Before beginning, ensure you have:

- Access to the platform (request credentials if needed)
- Python 3.8+ installed on your local machine
- Basic familiarity with Jupyter notebooks
- Your annotated image dataset prepared

Required Python packages:
```
jupyter
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
pillow>=8.0.0
torch>=1.8.0
torchvision>=0.9.0
```

## Data Preparation

### Supported Annotation Formats
- COCO JSON
- Pascal VOC XML
- YOLO txt
- Segmentation masks
- CSV annotations

### Directory Structure
Organize your data in the following structure:
```
dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── annotations/
│   ├── ann001.json (or .xml, .txt)
│   ├── ann002.json
│   └── ...
└── dataset_metadata.json
```

### Metadata File
Create a `dataset_metadata.json` file with:
```json
{
  "name": "your_dataset_name",
  "version": "1.0.0",
  "annotation_type": "coco", // or "pascal_voc", "yolo", "segmentation", "csv"
  "classes": ["class1", "class2", "..."],
  "split": {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15
  }
}
```

## Available Notebooks

The platform provides the following Jupyter notebooks:

1. `01_data_exploration.ipynb`: Visualize and analyze your dataset
2. `02_data_preprocessing.ipynb`: Clean, transform, and prepare your data
3. `03_data_ingestion.ipynb`: Upload data to the platform
4. `04_model_training.ipynb`: Configure and train models
5. `05_model_evaluation.ipynb`: Evaluate model performance
6. `06_inference.ipynb`: Run inference on new images
7. `07_export_model.ipynb`: Export models for deployment

You can find these notebooks in the platform's JupyterLab interface.

## Data Ingestion Process

### Step 1: Access the Data Ingestion Notebook
1. Log in to the platform
2. Navigate to JupyterLab interface
3. Open `03_data_ingestion.ipynb`

### Step 2: Configure Data Path
```python
# In the notebook
DATA_PATH = "/path/to/your/dataset"
METADATA_FILE = "/path/to/your/dataset/dataset_metadata.json"

# Validate the paths
import os
assert os.path.exists(DATA_PATH), "Dataset path not found"
assert os.path.exists(METADATA_FILE), "Metadata file not found"
```

### Step 3: Initialize the Ingestion Client
```python
# Execute this cell to authenticate
from platform_sdk import DataIngestionClient

# The notebook will handle authentication automatically
# Or you can specify your API key
client = DataIngestionClient(api_key="YOUR_API_KEY")
```

### Step 4: Validate Your Dataset
```python
# Run dataset validation
validation_result = client.validate_dataset(
    data_path=DATA_PATH,
    metadata_file=METADATA_FILE
)

# Check validation results
if validation_result.is_valid:
    print("Dataset is valid and ready for ingestion")
else:
    print("Validation issues found:")
    for issue in validation_result.issues:
        print(f"- {issue}")
```

### Step 5: Upload Your Dataset
```python
# Execute this cell to start uploading
dataset = client.upload_dataset(
    data_path=DATA_PATH,
    metadata_file=METADATA_FILE,
    verbose=True  # Show progress bar
)

# Save dataset ID for future reference
dataset_id = dataset.id
print(f"Dataset uploaded successfully. ID: {dataset_id}")
```

## Training Models

### Step 1: Open Training Notebook
Open `04_model_training.ipynb` from the JupyterLab interface.

### Step 2: Configure Training Parameters
```python
# Provide your dataset ID from the ingestion step
DATASET_ID = "your_dataset_id"

# Configure training parameters
training_config = {
    "architecture": "faster_rcnn",  # Options: faster_rcnn, yolo, mask_rcnn, etc.
    "backbone": "resnet50",        # Options: resnet50, resnet101, efficientnet, etc.
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 50,
        "optimizer": "adam"
    },
    "augmentation": {
        "horizontal_flip": True,
        "vertical_flip": False,
        "rotation_range": 15,
        "brightness_range": [0.8, 1.2]
    },
    # Optional: continue training from existing checkpoint
    "checkpoint": None  # or "model_id_to_continue_from"
}
```

### Step 3: Initialize Training Client
```python
from platform_sdk import ModelTrainer

trainer = ModelTrainer()  # Authentication handled by notebook environment
```

### Step 4: Start Training
```python
# Execute this cell to begin training
job = trainer.create_job(
    dataset_id=DATASET_ID,
    config=training_config
)

print(f"Training job started. Job ID: {job.id}")
```

### Step 5: Monitor Training Progress
```python
# The notebook includes interactive visualizations
# Execute this cell to display training progress
trainer.display_training_progress(job.id)
```

The notebook will display live metrics including:
- Training and validation loss
- Mean Average Precision (mAP)
- Learning rate schedule
- GPU utilization
- Estimated time remaining

## Running Inference

### Step 1: Open Inference Notebook
Open `06_inference.ipynb` from the JupyterLab interface.

### Step 2: Load Your Trained Model
```python
from platform_sdk import ModelInference

# Initialize inference client
inference = ModelInference()

# Load model by ID (from training step)
MODEL_ID = "your_trained_model_id"
model = inference.load_model(MODEL_ID)

print(f"Model loaded: {model.name}, Version: {model.version}")
```

### Step 3: Run Inference on Single Image
```python
# Path to your test image
IMAGE_PATH = "/path/to/test/image.jpg"

# Run inference
results = inference.predict(
    model=model,
    image_path=IMAGE_PATH,
    confidence_threshold=0.5
)

# Display results
inference.visualize_results(IMAGE_PATH, results)
```

### Step 4: Batch Inference
```python
# Directory containing test images
IMAGE_DIR = "/path/to/test/images/"

# Run batch inference
batch_results = inference.predict_batch(
    model=model,
    image_dir=IMAGE_DIR,
    confidence_threshold=0.5,
    batch_size=16  # Adjust based on your hardware
)

# Save results to CSV
batch_results.to_csv("inference_results.csv")

# Optional: Visualize a sample of results
inference.visualize_batch_results(
    image_dir=IMAGE_DIR,
    results=batch_results,
    max_images=5  # Number of images to visualize
)
```

### Step 5: Export Inference Results
```python
# Export options available in the notebook:
# - CSV export
# - COCO JSON format
# - Visual HTML report
# - Integration with external tools

# Example: Generate comprehensive HTML report
report_path = inference.generate_report(
    results=batch_results,
    output_dir="./inference_report"
)

print(f"Report generated at: {report_path}")
```

## Best Practices

### Data Management
- Version your datasets and keep detailed metadata
- Use consistent naming conventions across all files
- Maintain a separate test set that's never used in training

### Notebook Usage
- Execute cells in sequential order
- Save intermediate outputs for reproducibility
- Use checkpoints to save progress during long computations
- Document your parameter choices within the notebook

### Security
- Never hardcode API keys in notebooks
- Use environment variables or the platform's secure credential storage
- Encrypt sensitive data before uploading
- Follow the principle of least privilege for shared notebooks

### Resource Management
- Monitor GPU usage with the built-in tools
- Schedule resource-intensive jobs during off-peak hours
- Clean up unused data and models regularly

## Troubleshooting

### Common Issues and Solutions

#### Data Ingestion Failures
- **Issue**: Upload is rejected due to format issues
- **Solution**: Check the validation errors and correct the annotation format

#### Out of Memory Errors
- **Issue**: Training crashes with OOM errors
- **Solution**: Reduce batch size in the training configuration

#### Training Not Converging
- **Issue**: Loss not decreasing or very unstable
- **Solution**: Check for class imbalance and adjust learning rate

#### Inference Performance Issues
- **Issue**: Predictions are inaccurate or missing
- **Solution**: Try adjusting the confidence threshold and check for domain shift

### Getting Help
- Check the platform documentation at `https://platform-docs.url`
- Use the "Help" button in any notebook for context-specific assistance
- Contact support via the platform interface
- Join the community forum at `https://community.platform.url`

---

*This documentation was last updated on June 12, 2025. For the latest version, please visit our documentation portal.*
