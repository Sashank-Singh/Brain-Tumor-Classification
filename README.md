# Brain-Tumor-Classification

## Overview
This project implements a deep learning model to classify brain tumor MRI images into different categories. The classification is performed using a neural network model built with TensorFlow/Keras.

## Dataset
The dataset consists of MRI brain scans organized into the following categories:
- Glioma
- Meningioma
- No tumor (healthy)
- Pituitary tumor

The images are split into Training and Testing directories, with each containing subfolders for the different tumor types.

## Model Architecture
The Jupyter notebook `Brain_tumor_Classification.ipynb` implements the classification model using:
- Transfer learning with Xception pre-trained weights
- Data augmentation techniques to improve model generalization
- Fine-tuning layers for better accuracy on brain tumor classification

## Requirements
- Python 3.6+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn
- Jupyter Notebook

## Usage
1. Clone this repository
2. Install the required dependencies
3. Open the Jupyter notebook: `jupyter notebook Brain_tumor_Classification.ipynb`
4. Run the cells in sequence to train and evaluate the model

## Results
The model achieves high accuracy in classifying brain tumor types from MRI scans. Detailed performance metrics, including accuracy, precision, recall, and F1-score, are available in the notebook.

## Future Improvements
- Hyperparameter optimization
- Ensemble methods
- More advanced data augmentation
- Integration with a web or mobile application for easier access


