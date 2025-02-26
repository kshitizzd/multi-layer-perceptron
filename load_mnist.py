import numpy as np
import os
import struct
import kaggle
from sklearn.model_selection import train_test_split

def read_mnist_dataset(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]
#
# # Authenticate and download the dataset
# kaggle.api.authenticate()
# dataset_slug = "hojjatk/mnist-dataset"
# kaggle.api.dataset_download_files(dataset_slug, path='./mnist', unzip=True)

# Load and preprocess the dataset
base_path = './mnist'
train_images_path = os.path.join(base_path, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(base_path, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(base_path, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(base_path, 't10k-labels.idx1-ubyte')

train_images = read_mnist_dataset(train_images_path)
train_labels = read_mnist_dataset(train_labels_path)
test_images = read_mnist_dataset(test_images_path)
test_labels = read_mnist_dataset(test_labels_path)

train_images = train_images.reshape(-1, 784)/ 255.0
test_images = test_images.reshape(-1, 784)/255.0

# Standardization
X_mean = np.mean(train_images, axis=0)
X_std = np.std(train_images, axis=0)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_images,
    train_labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# One-hot encode labels
y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)

# Save preprocessed data to disk
np.savez('preprocessed_data/mnist_data.npz',
         X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
         test_images=test_images, test_labels=test_labels)