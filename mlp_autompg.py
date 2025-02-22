#!/usr/bin/env python3
# Import necessary libraries
import math
import numpy as np  
import matplotlib.pyplot as plt  
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

    # Yield batches of size `batch_size`
    for start_index in range(0, number_samples, batch_size):
        end_index = min(start_index + batch_size, number_samples)  
        batch_index = input_count[start_index:end_index] 
        yield train_x[batch_index], train_y[batch_index]  


# Activation Functions (Non-linearity)
class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the function output

        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates its gradient, needed for backpropagation

        """
        pass

class Sigmoid(ActivationFunction):
    """Smooth curve"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1. / (1. + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)

class Tanh(ActivationFunction):
    """Zero-centered"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class Relu(ActivationFunction):
    """used for hidden layers"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

class Softmax(ActivationFunction):
    """Used for classification"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        softmax_x = self.forward(x)
        return softmax_x * (1 - softmax_x)  

class Linear(ActivationFunction):
    """used in specific scenarios"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class Softplus(ActivationFunction):
    """used in specific scenarios"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.log(1 + np.exp(x))  

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))  

class Mish(ActivationFunction):
    """used in specific scenarios"""
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x * np.tanh(np.log(1 + np.exp(x))) 

    def derivative(self, x: np.ndarray) -> np.ndarray:
        omega = np.exp(x)
        delta = 1 + omega + omega ** 2
        return omega * delta / ((1 + omega) ** 2 + omega ** 2)  


# Loss Functions (How much the model is wrong)
class LossFunction(ABC):
    """
    Abstract Base Class for Loss Functions.
    Each loss function must define `loss()` and `derivative()`.
    """
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        """Computes the loss value"""
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        """Computes the derivative of the loss function"""
        pass

# Mean Squared Error (MSE) for regression
class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return np.mean((y_true - y_prediction) ** 2)

    def derivative(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return 2 * (y_prediction - y_true) / y_true.size

# Cross-Entropy Loss (Used for classification)
class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return -np.mean(y_true * np.log(y_prediction + 1e-9)) 

    def derivative(self, y_true: np.ndarray, y_prediction: np.ndarray) -> np.ndarray:
        return y_prediction - y_true  


# Layer
class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate=0.0):
        """
        Initializes a layer of neurons
        Initialize weights and biases using Glorot (Xavier) initialization
        Each layer applies a linear transformation (W * X + b),
        then passes it through an activation function.
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        limit = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))  # Xavier Initialization

        self.b = np.zeros((1, fan_out)) 
    
    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        """
        self.h = h  
        self.z = np.dot(h, self.W) + self.b  
        self.activations = self.activation_function.forward(self.z)  

        if self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.activations.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            self.activations *= self.dropout_mask

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        """
        activation_derivative = self.activation_function.derivative(self.z)

        if delta.ndim == 1:
          delta = delta[:, np.newaxis]  

        if activation_derivative.shape != delta.shape:
          activation_derivative = activation_derivative.reshape(delta.shape)

        self.delta = delta * activation_derivative

        if self.W.shape[1] != self.delta.shape[1]:
          self.delta = self.delta.reshape(-1, self.W.shape[1])

        delta_prev = np.dot(self.delta, self.W.T)
        # Weight gradient 
        dL_dW = np.dot(self.h.T, self.delta)  
        # Bias gradient 
        dL_db = np.sum(self.delta, axis=0, keepdims=True)  
        # Pass gradients for weight update
        return dL_dW, dL_db, delta_prev  
    
# Build Multilayer Perceptron from scratch
class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        
        """

        for layer in self.layers:
            # Pass input through each layer
            x = layer.forward(x)  
        return x 
    
    def predict(self, x: np.ndarray) -> np.ndarray:
      """
      Predict class probabilities or outputs using the trained MLP model.
    
      """
      return self.forward(x)

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        
        """
        deltaL_deltaW_all = []
        deltaL_deltaB_all = []

        for layer in reversed(self.layers):
            deltaL_deltaW, deltaL_deltaB, loss_grad = layer.backward(layer.h, loss_grad)  
            deltaL_deltaW_all.insert(0, deltaL_deltaW)  
            deltaL_deltaB_all.insert(0, deltaL_deltaB)

        return deltaL_deltaW_all, deltaL_deltaB_all  # Return all weight and bias gradients

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-4, batch_size: int=16, epochs: int=32, rmsprop: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        """
        training_loss = []
        validation_loss = []

        number_samples = train_x.shape[0]
        # Initialize RMSProp parameters
        beta = 0.9
        epsilon = 1e-8
        squared_gradients = [np.zeros_like(layer.W) for layer in self.layers]  

        for epoch in range(epochs):
            # Shuffle the training data
            input_count = np.arange(number_samples)
            np.random.shuffle(input_count)
            train_x, train_y = train_x[input_count], train_y[input_count]

            # Mini-batch training
            for start_index in range(0, number_samples, batch_size):
                end_index = min(start_index + batch_size, number_samples)
                batch_x, batch_y = train_x[start_index:end_index], train_y[start_index:end_index]

                # Forward pass
                y_prediction = self.forward(batch_x)

                # Compute loss gradient
                if batch_y.shape != y_prediction.shape:
                  y_prediction = y_prediction.reshape(batch_y.shape)

                loss_grad = loss_func.derivative(batch_y, y_prediction)


                # Backpropagation
                deltaL_deltaW_all, deltaL_deltaB_all = self.backward(loss_grad, batch_x)


                # Update weights and biases
                for i, layer in enumerate(self.layers):
                    if rmsprop:
                        if deltaL_deltaW_all[i].shape != squared_gradients[i].shape:
                           print(f"Warning: Shape mismatch for layer {i}. deltaL_deltaW_all[i]: {deltaL_deltaW_all[i].shape}, squared_gradients[i]: {squared_gradients[i].shape}")
                           continue  


                        squared_gradients[i] = beta * squared_gradients[i] + (1 - beta) * deltaL_deltaW_all[i] ** 2
                        layer.W -= learning_rate * deltaL_deltaW_all[i] / (np.sqrt(squared_gradients[i]) + epsilon)
                    else:
                        layer.W -= learning_rate * deltaL_deltaW_all[i]
    
                    layer.b -= learning_rate * deltaL_deltaB_all[i]  
 
            train_loss = loss_func.loss(train_y, self.forward(train_x))
            val_loss = loss_func.loss(val_y, self.forward(val_x))
            training_loss.append(train_loss)
            validation_loss.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")

        return np.array(training_loss), np.array(validation_loss)

