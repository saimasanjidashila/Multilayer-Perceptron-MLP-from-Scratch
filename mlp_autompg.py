#!/usr/bin/env python3
# Import necessary libraries
import math
import numpy as np  
import matplotlib.pyplot as plt  
from abc import ABC, abstractmethod  
from typing import Tuple 
import numpy as np   
from abc import ABC, abstractmethod  
from typing import Tuple

# Mini-Batch Data Generator
def batch_generator(train_x, train_y, batch_size):
    """
    Instead of processing all training data at once, we split it into smaller batches.
    This function generates batches of training data (train_x and train_y).
    """
    number_samples = train_x.shape[0]  
    input_count = np.arange(number_samples) 
    np.random.shuffle(input_count)  
    for start_index in range(0, number_samples, batch_size):
        end_index = min(start_index + batch_size, number_samples)  
        batch_index = input_count[start_index:end_index] 
        yield train_x[batch_index], train_y[batch_index]  

# Activation Functions (Non-linearity)
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes the function output"""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculates its gradient, needed for backpropagation"""
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

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
        sm = self.forward(x)
        return sm * (1 - sm)

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

class Mish(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x)))
    def derivative(self, x: np.ndarray) -> np.ndarray:
        omega = np.exp(x)
        delta = 1 + omega + omega ** 2
        return omega * delta / ((1 + omega) ** 2 + omega ** 2)

# Loss Functions
class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_prediction) ** 2)
    def derivative(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return 2 * (y_prediction - y_true) / y_true.size

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return -np.mean(y_true * np.log(y_prediction + 1e-9))
    def derivative(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return y_prediction - y_true  

# Layer
class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate=0.0):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        self.b = np.zeros((1, fan_out)) 
    
    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        self.h = h  
        self.z = np.dot(h, self.W) + self.b  
        if self.z.ndim == 1:
            self.z = self.z[:, np.newaxis]
        self.activations = self.activation_function.forward(self.z)
        if self.activations.ndim == 1:
            self.activations = self.activations[:, np.newaxis]
        if training and self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.activations.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            self.activations *= self.dropout_mask
        return self.activations
    
    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        activation_derivative = self.activation_function.derivative(self.z)
        if activation_derivative.ndim == 1:
            activation_derivative = activation_derivative[:, np.newaxis]
        if delta.ndim == 1:
            delta = delta[:, np.newaxis]
        self.delta = delta * activation_derivative
        delta_prev = np.dot(self.delta, self.W.T)
        dL_dW = np.dot(self.h.T, self.delta)
        dL_db = np.sum(self.delta, axis=0, keepdims=True)
        return dL_dW, dL_db, delta_prev

# Multilayer Perceptron
class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)

    def backward(self, loss_grad: np.ndarray) -> Tuple[list, list]:
        deltaW_list = []
        deltaB_list = []
        for layer in reversed(self.layers):
            dW, dB, loss_grad = layer.backward(layer.h, loss_grad)
            deltaW_list.insert(0, dW)
            deltaB_list.insert(0, dB)
        return deltaW_list, deltaB_list

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
              loss_func: LossFunction, learning_rate: float = 1E-4, batch_size: int = 16, epochs: int = 32,
              rmsprop: bool = False, momentum: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        training_loss = []
        validation_loss = []
        num_samples = train_x.shape[0]

        # Set initial learning rate and decay rate
        initial_lr = learning_rate
        # Decay learning rate by 5% each epoch
        decay_rate = 0.95  
        # RMSProp parameters
        beta = 0.9
        epsilon = 1e-8
        squared_gradients = [np.zeros_like(layer.W) for layer in self.layers]

        # Velocities for momentum
        velocities_W = [np.zeros_like(layer.W) for layer in self.layers]
        velocities_b = [np.zeros_like(layer.b) for layer in self.layers]

        for epoch in range(epochs):
            current_lr = initial_lr * (decay_rate ** epoch)
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                # Forward pass (with dropout enabled)
                y_pred = self.forward(batch_x, training=True)
                # Compute loss gradient
                loss_grad = loss_func.derivative(batch_y, y_pred)
                # Backpropagation to compute gradients for each layer
                deltaW_list, deltaB_list = self.backward(loss_grad)
                # Update weights and biases for each layer
                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        squared_gradients[i] = beta * squared_gradients[i] + (1 - beta) * deltaW_list[i] ** 2
                        grad_update = current_lr * deltaW_list[i] / (np.sqrt(squared_gradients[i]) + epsilon)
                    else:
                        grad_update = current_lr * deltaW_list[i]
                    velocities_W[i] = momentum * velocities_W[i] - grad_update
                    layer.W += velocities_W[i]
                    bias_update = current_lr * deltaB_list[i]
                    velocities_b[i] = momentum * velocities_b[i] - bias_update
                    layer.b += velocities_b[i]

            # Compute training and validation losses without dropout
            train_loss = loss_func.loss(train_y, self.forward(train_x, training=False))
            val_loss = loss_func.loss(val_y, self.forward(val_x, training=False))
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        return np.array(training_loss), np.array(validation_loss)
