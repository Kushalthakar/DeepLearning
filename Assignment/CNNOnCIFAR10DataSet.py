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

DIR = "output"
os.makedirs(DIR, exist_ok = True)

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

# Inspect the data
print("\nData Inspection")

print("xTrainFull Shape:", xTrainFull.shape)
print("yTrainFull Shape:", yTrainFull.shape)

print("xTest Shape:", xTest.shape)
print("yTest Shape:", yTest.shape)

print("Training pixel min/max:", xTrainFull.min(), xTrainFull.max())
print("Test Pixel min/max:", xTest.min(), xTest.max())

print("Training label min/max", yTrainFull.min(), yTrainFull.max())
print("Test Pixel min/Max:", yTest.min(), yTest.max())

print("Training count:")

for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name: 10s} -> {np.sum(yTrainFull == i)}")

# Image grid
def saveGrid(images, labels, classNames, fileName):
    plt.figure(figsize=(10, 6))
    for i in range(20):
        plt.imshow(images[i])
        plt.title(classNames[labels[i]], fontsize = 10)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(fileName, dip = 200)
    plt.close()

saveGrid(xTrainFull,
         yTrainFull,
         CLASS_NAMES,
         os.path.join(DIR, "image.png"))

print(f"\nSave image grid to {DIR}/image.png ")

# Normalize image
xTrainFull = xTrainFull.astype("float32") / 255.0
xTest = xTest.astype("float32") / 255.0

print("After Normalization")

print("Training pixel min/max:", xTrainFull.min(), xTrainFull.max())
print("Test pixel min/max:", xTest.min(), xTest.max())

print("Training label min/max:", yTrainFull.min(), yTrainFull.max())
print("Test label min/max:", yTest.min(), yTest.max())

# Validation Split
xTrain, xVal, yTrain, yVal = train_test_split(xTrainFull,
                                          yTrainFull,
                                          test_size = 0.10,
                                          random_state = SEED,
                                          stratify = yTrainFull)

print("x train:", xTrain.shape, "y train:", yTrain.shape)
print("x val:", xVal.shape, "y val:", yVal.shape)
print("x test:", xTest.shape, "y test:", yTest.shape)

# Data augmentation
dataAugmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"),
                                        layers.RandomRotation(0.08),
                                        layers.RandomZoom(0.10),],
                                        name = "dataAugmentation")

# CNN Model
def cnnModel(inputShape = (32, 32, 3), numClasses = 10):
    l2 = regularizers.l2(1e-4)

    inputs = layers.Input(shape = inputShape)

    x = dataAugmentation(inputs)

    # Block 1 (32 filters)
    x = layers.Conv2D(32, (3,3), padding = "same", kernel_regularizer = 12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relue")(x)

    x = layers.Conv2D(32, (3,3), padding = "same", kernel_regularizer = 12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.20)(x)

    # Block 2 (64 Filters)
    x = layers.Conv2D(64, (3,3), padding = "same", kernel_regularizer = 12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relue")(x)

    x = layers.Conv2D(64, (3,3), padding = "same", kernel_regularizer = 12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.30)(x)

    # Block 3 (128 Filters)
    x = layers.Conv2D(128, (3,3), kernel_regularizer=12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(128, (3,3), kernel_regularizer = 12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation()(x)

    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.40)(x)

    # Classifier
    x = layers.Flatten()(x)
    x = layers.Dense(256, kernel_regularizer = 12)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.50)(x)

    outputs = layers.Dense(numClasses, activation = "softmax")(x)

    model = models.Model(inputs = inputs, outputs = outputs, name = "CIFAR10_CNN")

    return model

model = cnnModel()
model.summary()

# Configure
model.compile(optimizer = tf.keras.optimizers.Adam(learningRate = 0.001),
              loss = "sparseCategoricalCrossentropy",
              metrics = ["accuracy"])

# Train Model
BATCH_SIZE = 64
EPOCHS = 15

goodModelPath = os.path.join(DIR, "goodCifar10Cnn.keras")

callback = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = goodModelPath,
        monitor = "val accuracy",
        mode = "max",
        saveOnly = True,
        verbose = 1),
    tf.keras.callbacks.EarlyStopping(
        monitor = "val accuracy",
        mode = "max",
        patience = 7,
        restore_best_weights = True,
        verbose = 1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor = "val loss",
        factor = "0.5",
        patience = 3,
        min_lr = 1e-6,
        verbose = 1)
]

history = model.fit(xTrain,
                    yTrain,
                    validation_data = (xVal, yVal),
                    epochs = EPOCHS,
                    batch_size = BATCH_SIZE,
                    callbacks = callback,
                    verbose = 1)

# Training Curve
def saveTrainingCurves(history, fileName):
    hist = history.history
    epochsRange = range(1, len(hist["accuracy"]) + 1)

    plt.figure(figsize = (10, 4))
    plt. subplot(1, 2, 1)
    plt.plot(epochsRange, hist["accuracy"], label = "Training Accuracy")
    plt.plot(epochsRange, hist["val accuracy"], label = "Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochsRange, hist["loss"], label = "Training Loss")
    plt.plot(epochsRange, hist["val_loss"], label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(fileName, dpi = 200)
    plt.close()

trainingCurvesPath = os.path.join(DIR, "trainingCurves.png")
saveTrainingCurves(history, trainingCurvesPath)
print(f"\nSaves curves to {trainingCurvesPath}")

# Final Evaluation
print("\n Test Evaluation")
testLoss, testAccuracy = model.evaluate(xTest,yTest, batchSize = BATCH_SIZE, verbose = 0)
print(f"Test Loss: {testLoss: .4f}")
print(f"Test Accuracy: {testAccuracy: .4f}")