#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import sklearn
import pandas as pd
from  mlp_autompg import MultilayerPerceptron, Layer, Relu, Linear, SquaredError, Tanh, Sigmoid, Softmax, Softplus, Mish
from sklearn.model_selection import train_test_split

# fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# data (as pandas dataframes)
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


# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,    # for reproducibility
    shuffle=True,       # whether to shuffle the data before splitting
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X (features)
X_mean = X_train.mean(axis=0)  # Mean of each feature
X_std = X_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

X_train.shape

print(f"Samples in Training:   {len(X_train)}")
print(f"Samples in Validation: {len(X_val)}")
print(f"Samples in Testing:    {len(X_test)}")

# Convert Pandas DataFrame to NumPy array before training
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_val = X_val.to_numpy()
y_val = y_val.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()

# Define MLP Architecture 
mlp = MultilayerPerceptron([
    Layer(fan_in=X_train.shape[1], fan_out=256, activation_function=Relu(), dropout_rate=0.1),
    Layer(fan_in=256, fan_out=128, activation_function=Mish(), dropout_rate=0.1),
    Layer(fan_in=128, fan_out=64, activation_function=Mish(), dropout_rate=0.05),
    Layer(fan_in=64, fan_out=32, activation_function=Mish(), dropout_rate=0.05),
    Layer(fan_in=32, fan_out=16, activation_function=Relu(), dropout_rate=0.02),
    Layer(fan_in=16, fan_out=1, activation_function=Linear())
])
# Train the Model
loss_function = SquaredError()
train_losses, val_losses = mlp.train(X_train, y_train, X_val, y_val,
                                         loss_function, learning_rate=0.0005,
                                         batch_size=16, epochs=50, rmsprop=True)

# Evaluate Model on Test Data
y_pred = mlp.forward(X_test)
test_loss = loss_function.loss(y_test, y_pred)
print(f"\n Total Test Loss: {test_loss:.4f}\n")

# Print True vs Predicted MPG for First 10 Samples
comparison = np.hstack((y_test.reshape(-1,1), y_pred.reshape(-1,1)))
print("\n True MPG vs Predicted MPG:\n")
print("True MPG | Predicted MPG")
for i in range(10):
    print(f"{comparison[i, 0]:.2f} | {comparison[i, 1]:.2f}")

# Plot Training vs Validation Loss
plt.plot(train_losses, label="Training Loss", color='b')
plt.plot(val_losses, label="Validation Loss", color='r')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss (Vehicle MPG)")
plt.show()
