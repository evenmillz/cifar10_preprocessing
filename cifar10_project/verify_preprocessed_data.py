import numpy as np
import matplotlib.pyplot as plt

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

# Label name mapping
label_names = {
    0: "Aircraft",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

# Display 15 example images with labels
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for i, ax in enumerate(axes.flat):
    img = X_train[i].reshape(32, 32, 3)
    ax.imshow(img)
    ax.set_title(f"Label: {y_train[i]}\n{label_names[y_train[i]]}")
    ax.axis('off')

plt.tight_layout()
plt.show()