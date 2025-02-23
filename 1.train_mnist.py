#!/usr/bin/env python3
"""
# Author : Saima Sanjida Shila
# Course: CSC7700 - LSU
# Import Necessary Libraries :

"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# %matplotlib inline
import struct
from array import array
from os.path import join
import random
from sklearn.model_selection import train_test_split
import numpy as np
from mlp import MultilayerPerceptron, Relu, Softmax, CrossEntropy, Dense
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


# MNIST Data Loader Class
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())

        images = np.array(image_data, dtype=np.uint8).reshape(size, rows, cols)

        return images, np.array(labels)

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


# Set file paths based on added MNIST Datasets
training_images_filepath = 'train-images.idx3-ubyte'
training_labels_filepath = 'train-labels.idx1-ubyte'
test_images_filepath =  't10k-images.idx3-ubyte'
test_labels_filepath =  't10k-labels.idx1-ubyte'

# Load MNIST dataset
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Normalize values and flatten 28/28
x_train = x_train.reshape(len(x_train), 784) / 255.0
x_test = x_test.reshape(len(x_test), 784) / 255.0

# Split training data into 80% train, 20% validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle=True)

# One-hot encode labels
number_classes = 10
y_train = to_categorical(y_train, num_classes=number_classes)
y_val = to_categorical(y_val, num_classes=number_classes)
y_test = to_categorical(y_test, num_classes=number_classes)

# Build the MLP Model
mlp_model = MultilayerPerceptron([
    Dense(784, 1024, Relu()),  # More neurons
    Dense(1024, 512, Relu()),
    Dense(512, 256, Relu()),
    Dense(256, 128, Relu()),
    Dense(128, 10, Softmax())
])

# Train the Model
training_loss, validation_loss = mlp_model.train(
    x_train, y_train, x_val, y_val,
    learning_rate=0.001, batch_size=64, epochs=20
)

# Evaluate Model on Test Data
print("\n Evaluating Model on Test Data...")
y_pred = mlp_model.predict(x_test)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true) * 100

# Print Accuracy
print(f"\n Final Test Accuracy: {accuracy:.2f}%")

# Visualize Training Loss & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 21), training_loss, label="Training Loss",color ='r',marker='o')
plt.plot(range(1, 21), validation_loss, label="Validation Loss", color='b',marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over 20 Epochs")
plt.legend()
plt.grid()
plt.show()

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')

    # 🛠️ Fix the label display
    true_label = np.argmax(y_test[i])  # Convert one-hot to digit
    ax.set_title(f"Predicted: {y_pred[i]}\nTrue: {true_label}")

    ax.axis("off")

plt.show()

# Build the MLP Model 2
mlp_model2 = MultilayerPerceptron([
    Dense(784, 2048, Relu()),
    Dense(2048, 1024, Relu()),
    Dense(1024, 512, Relu()),
    Dense(512, 256, Relu()),
    Dense(256, 128, Relu()),
    Dense(128, 10, Softmax())
])

# Train the Model
training_loss2, validation_loss2 = mlp_model2.train(
    x_train, y_train, x_val, y_val,
    learning_rate=0.001, batch_size=64, epochs=50
)

# Evaluate Model on Test Data
print("\n Evaluating Model on Test Data...")
y_pred2 = mlp_model2.predict(x_test)
y_true2 = np.argmax(y_test, axis=1)
accuracy2 = np.mean(y_pred2 == y_true2) * 100

print(f"\n Final Test Accuracy: {accuracy2:.2f}%")

# Visualize Training Loss & Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, 51), training_loss2, label="Training Loss", marker='o')
plt.plot(range(1, 51), validation_loss2, label="Validation Loss", marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Over 50 Epochs")
plt.legend()
plt.grid()
plt.show()

# Visualization
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')

    # 🛠️ Fix the label display
    true_label = np.argmax(y_test[i])  # Convert one-hot to digit
    ax.set_title(f"Predicted: {y_pred2[i]}\nTrue: {true_label}")

    ax.axis("off")

plt.show()

