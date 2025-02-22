#!/usr/bin/env python3
"""
# Author : Saima Sanjida Shila
# Course: CSC7700 - LSU
# Import Necessary Libraries :

"""
import numpy as np
from typing import Tuple

#  Activation Functions
class ActivationFunction:
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        softmax_x = self.forward(x)
        return softmax_x * (1 - softmax_x)  

# Loss Functions
class CrossEntropy:
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[0]  


# MLP Layers
class Dense:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.W) + self.b
        return self.activation.forward(self.z)

    def backward(self, grad_output, learning_rate):
        activation_grad = grad_output * self.activation.derivative(self.z)
        grad_W = np.dot(self.input.T, activation_grad)
        grad_b = np.sum(activation_grad, axis=0, keepdims=True)
        grad_input = np.dot(activation_grad, self.W.T)

        # Update weights immediately using Vanilla SGD
        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b

        return grad_input

#  MLP Model
class MultilayerPerceptron:
    def __init__(self, layers):
        self.layers = layers
        self.loss_func = CrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, learning_rate):
        """Applies backpropagation through all layers"""
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, x_train, y_train, x_val, y_val, learning_rate=0.001, batch_size=64, epochs=20):
       training_losses = []
       validation_losses = []

       for epoch in range(epochs):
         indices = np.arange(x_train.shape[0])
         np.random.shuffle(indices)
         x_train, y_train = x_train[indices], y_train[indices]

         for start_idx in range(0, x_train.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, x_train.shape[0])
            batch_x, batch_y = x_train[start_idx:end_idx], y_train[start_idx:end_idx]

            # Forward pass
            y_pred = self.forward(batch_x)
            
            # Compute loss gradient
            loss_grad = self.loss_func.derivative(batch_y, y_pred)
            
            # 🛠️ FIX: Pass learning_rate as an argument to backward()
            self.backward(loss_grad, learning_rate)

         # Compute losses after the epoch
         train_loss = self.loss_func.loss(y_train, self.forward(x_train))
         val_loss = self.loss_func.loss(y_val, self.forward(x_val))

         training_losses.append(train_loss)
         validation_losses.append(val_loss)

         print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

       return training_losses, validation_losses  

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
