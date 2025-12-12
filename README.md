# Project Overview

This final project focuses on facial emotion classification. The purpose is to build a model that can predict the emotion category from a single face image, using visual cues like facial expressions.  

The main goals are to train an accurate and reliable classifier, evaluate it with metrics such as accuracy and F1-score, and understand which emotions are easier or harder for the model to distinguish.

# Setup Instructions

Follow these step-by-step instructions to set up the environment for this facial emotion classification project.

## Prerequisites

- Python 3.8 or higher
- pip
- CUDA-compatible GPU for faster training and inference

## Step 1: Clone or Download the Project

If you haven't already, download or clone this project to your local machine.

## Step 2: Install Dependencies

Install all required packages from requirements.txt:

```bash
pip install -r requirements.txt
```

## Step 3: Download and Prepare the Dataset

Since the dataset is not included in this repository due to size constraints, you need to download it separately:

1. Download the dataset from Kaggle: [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions/data)

2. Create a `Data` directory in the project root:
   ```bash
   mkdir Data
   ```

3. Extract the downloaded dataset and organize it in the following structure:
   ```
   Data/
   ├── train/
   │   ├── angry/
   │   ├── fear/
   │   ├── happy/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── fear/
       ├── happy/
       ├── sad/
       └── surprise/
   ```

4. Ensure each emotion folder contains the corresponding facial images

Your environment is now ready!

# How to Run

This project provides two ways to interact with the facial emotion classification model:

## Option 1: Quick Demo

To run a simple demonstration of the trained model's emotion classification capabilities:

```bash
cd demo
python demo.py
```

This demo script will:
- Load the pre-trained model from `best_seresnet18.pth`
- Randomly select sample images from the `Data` directory
- Perform emotion classification on each image
- Generate visualization results in the `results` directory, including:
  - Individual prediction visualizations with confidence scores
  - A summary grid showing all predictions

The demo showcases the core functionality of the project without requiring any training.

## Option 2: Full Training, Evaluation, and Testing

To train the model from scratch, evaluate performance, and test on your own data:

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook final_project.ipynb
   ```

2. **Important**: Before running the notebook, you need to modify the `DATA_DIR` variable in the **Main** section to match your local data path:
   ```python
   DATA_DIR = "your/path/to/Data"  # Change this to your actual Data directory path
   ```

3. Run all cells in the notebook sequentially

The notebook includes:
- **Data loading and preprocessing** - Prepare the emotion dataset
- **Model training** - Train the SE-ResNet18 model with data augmentation
- **Evaluation** - Calculate accuracy, F1-score, and generate confusion matrix
- **Testing** - Test the model on validation/test data
- **Visualization** - Plot training curves and performance metrics

# Expected Output

After running the demo script, the model will:
- Randomly select 9 sample images from the `Data` directory
- Predict the emotion for each image (Angry, Fear, Happy, Sad, or Surprise)
- Display prediction results with confidence scores in the console
- Save visualization images to the `results` directory, including individual predictions and a summary grid

# Pre-trained Model

This project includes a pre-trained SE-ResNet18 model (`best_seresnet18.pth`) for emotion classification.

The model is automatically loaded when running the demo script. If you want to use it in your own code, refer to `demo/demo.py` for examples on how to load the model.

# Acknowledgments

This project utilizes the following resources:

- **Dataset**: [Human Face Emotions](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions/data) from Kaggle

- **Squeeze-and-Excitation Networks**: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)

- **Deep Residual Learning for Image Recognition**: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)