import tarfile

# Step 1: Extract the CIFAR-10 dataset
tar = tarfile.open("cifar-10-python.tar.gz")
tar.extractall()
tar.close()

print("Dataset extracted successfully.") 

import pickle
import numpy as np
import os

# Step 2: Load and combine CIFAR-10 batches
def load_batch(batch_name):
    with open(batch_name, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels

data_dir = './cifar-10-batches-py'
combined_data = []
combined_labels = []

for i in range(1, 6):
    batch_file = os.path.join(data_dir, f'data_batch_{i}')
    data, labels = load_batch(batch_file)
    combined_data.append(data)
    combined_labels.extend(labels)

# Stack all data into one NumPy array
X_train = np.vstack(combined_data)
y_train = np.array(combined_labels)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)

# Step 3: Normalize the data
X_train = X_train.astype('float32') / 255.0

# Step 4: Save the preprocessed dataset
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

print("Preprocessed data saved as 'X_train.npy' and 'y_train.npy'")