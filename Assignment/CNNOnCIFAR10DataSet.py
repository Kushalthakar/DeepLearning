import os
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from keras import layers, models, regularizers

# Force CPU
DEVICE = torch.device('cpu')
print("Using device:", DEVICE)
print("TensorFlow Version:", tf.__version__)
print("Keras Version:", keras.__version__)
print("PyTorch Version:", torch.__version__)

# Reproducable and output folder
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok = True)

# Class Names
CLASS_NAMES = ["airplane", 
               "automobile",
               "birds",
               "cat",
               "deer",
               "dog",
               "frog",
               "horse",
               "ship",
               "truck"]

# Load CIFAR-10
(xTrainFull, yTrainFull), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
yTrainFull = yTrainFull.squeeze()
yTest = yTest.squeeze()

#Inspect the data
print("\nData Inspection")
print("xTrainFull Shape:", xTrainFull.shape)
print("yTrainFull Shape:", yTrainFull.shape)
print("xTest Shape:", xTest.shape)
print("yTest Shape:", yTest.shape)
print("Training pixel min/max:", xTrainFull.min(), xTrainFull.max())
print("Test Pixel min/max:", xTest.min(), xTest.max())
print("Training label min/max", yTrainFull.min(), yTrainFull.max())
print("Training label min/Max:", yTest.min(), yTest.max())
print("Training count:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name: 10s} -> {np.sum(yTrainFull == i)}")