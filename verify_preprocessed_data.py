import numpy as np

# Load the preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Inspect the shape and data type
print("Reloaded X_train shape:", X_train.shape)
print("Reloaded y_train shape:", y_train.shape)
print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)

# Optional: check a single label and corresponding pixel data summary
print("Example label:", y_train[0])
print("Example image pixel min/max:", X_train[0].min(), X_train[0].max())