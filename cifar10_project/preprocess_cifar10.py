import tensorflow as tf
import numpy as np
import os

def load_and_preprocess_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("Training data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)

    return (x_train, y_train), (x_test, y_test)

def save_to_npz(x_train, y_train, x_test, y_test, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "cifar10_preprocessed.npz"),
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"Saved preprocessed data to: {output_dir}/cifar10_preprocessed.npz")

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_cifar10()
    save_to_npz(x_train, y_train, x_test, y_test)