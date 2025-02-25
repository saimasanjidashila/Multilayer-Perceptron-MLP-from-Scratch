#!/usr/bin/env python3
"""
# Author : Saima Sanjida Shila
# Course: CSC7700 - LSU
# Import Necessary Libraries :

"""
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import pandas as pd
from mlp_autompg import LossFunction, CrossEntropy, ActivationFunction, batch_generator
from mlp_autompg import MultilayerPerceptron, Layer, Relu, Linear, SquaredError, Tanh, Sigmoid, Softmax, Mish, Softplus
from sklearn.model_selection import train_test_split

# Fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# Data (as pandas DataFrames)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering
data = pd.concat([X, y], axis=1)

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
X = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

print("Head of the dataset:")
print(data.head())
print("\nTail of the dataset:")
print(data.tail())

# Split the entire dataset into 70% training and 30% leftover.
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

# Split the leftover 30% equally into validation and test sets 
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

print(f"Samples in Training:   {len(X_train)}")
print(f"Samples in Validation: {len(X_val)}")
print(f"Samples in Testing:    {len(X_test)}")

# Compute statistics for features based on training data
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

# Normalize features for train, validation, and test sets using training stats
X_train = (X_train - X_mean) / X_std
X_val   = (X_val   - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

# Compute statistics for targets based on training data
y_mean = y_train.mean()
y_std  = y_train.std()

# Normalize targets for train, validation, and test sets using training stats
y_train = (y_train - y_mean) / y_std
y_val   = (y_val   - y_mean) / y_std
y_test  = (y_test  - y_mean) / y_std

print(f'x_train shape here: {X_train.shape}')
print(f'x_val shape here: {X_val.shape}')
print(f'x_test shape here: {X_test.shape}')
print(f'y_train shape here: {y_train.shape}')
print(f'y_val shape here: {y_val.shape}')
print(f'y_test shape here: {y_test.shape}')

"""
Convert Pandas DataFrames to NumPy arrays
"""
X_train = X_train.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
X_val = X_val.to_numpy()
y_val = y_val.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()
y_test = y_test.to_numpy().reshape(-1, 1)

print("Data Processing done....")

# Define MLP Architecture
mlp = MultilayerPerceptron([
    Layer(fan_in=X_train.shape[1], fan_out=256, activation_function=Relu(), dropout_rate=0.3),
    Layer(fan_in=256, fan_out=128, activation_function=Relu(), dropout_rate=0.3),
    Layer(fan_in=128, fan_out=64, activation_function=Relu(), dropout_rate=0.02),
    Layer(fan_in=64, fan_out=32, activation_function=Relu(), dropout_rate=0.02),
    Layer(fan_in=32, fan_out=16, activation_function=Relu(), dropout_rate=0.01),
    Layer(fan_in=16, fan_out=1, activation_function=Linear())
])

# Train the Model....
loss_function = SquaredError()
train_loss, val_loss = mlp.train(
    X_train, y_train, X_val, y_val,
    loss_function, learning_rate=1e-3,
    batch_size=16, epochs=50, rmsprop=True, momentum=0.9
)

# Evaluate Model on Test Data
y_test_pred = mlp.forward(X_test, training=False)
test_loss = loss_function.loss(y_test, y_test_pred)
print(f"\nTotal Test Loss: {test_loss:.4f}")

# Print 10 sample comparisons: Actual MPG vs. Predicted MPG from the Test Set
comparison = np.hstack((y_test, y_test_pred))
print("\nActual MPG vs. Predicted MPG (Test Set):")
print("Actual MPG | Predicted MPG")
for i in range(min(10, comparison.shape[0])):
    print(f"{comparison[i, 0]:.2f} | {comparison[i, 1]:.2f}")

# Plot Training vs. Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(train_loss, label="Training Loss", color="blue")
plt.plot(val_loss, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss (Vehicle MPG)")
plt.legend()
plt.show()
