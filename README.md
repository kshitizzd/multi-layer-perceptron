# MLP Project

This repository contains an implementation of a Multi-Layer Perceptron (MLP) model for training on the MNIST and MPG datasets. 
The project is organized into multiple directories and files, each serving a specific purpose.

## Folder Structure
```
mlp/
│── mnist/                # Contains the MNIST dataset
│── data/                 # Stores training losses and validation losses
│── mpg_table_result/     # Contains predicted and true MPG values
│── preprocessed_data/    # Stores preprocessed x_train and y_train datasets
│── mlp.py                # Main MLP model, activation functions, loss functions, and neural layers
│── load_mnist.py         # Loads the MNIST dataset
│── load_mpg.py           # Loads the MPG dataset
│── loss_curves_mnist.png # Loss curve for MNIST
│── loss_curves_mpg.png   # Loss curve for MPG
│── train_mnist.py        # Trains the MLP model on MNIST dataset
│── train_mpg.py          # Trains the MLP model on MPG dataset
```

## Description of Files and Directories

### 1. **mnist/**
   - This directory contains the MNIST dataset used for training and testing the MLP model.

### 2. **data/**
   - Stores the training and validation losses collected during training.
   - These losses are used for plotting graphs to visualize the model’s performance.

### 3. **mpg_table_result/**
   - Contains predicted and true MPG values.
   - The stored data includes:
     - `true_mpg`: The actual MPG values from the dataset.
     - `predicted_mpg`: The computed MPG values from the trained model.

### 4. **preprocessed_data/**
   - After obtaining the dataset, the preprocessed training data eg:(`x_train`, `y_train`) is stored in this folder.
   - This ensures that all regularization and preprocessing steps are saved and can be reused.

### 5. **mlp.py**
   - The core implementation of the Multi-Layer Perceptron (MLP) model.
   - Includes the following:
     - Model architecture
     - Activation functions
     - Loss functions
     - Neural network layers

### 6. **load_mnist.py**
   - Loads the MNIST dataset for training and evaluation.

### 7. **load_mpg.py**
   - Loads the MPG dataset for training and evaluation.

### 8. **loss_curves_mnist.png**
   - Stored loss curve visualization for MNIST training.

### 9. **loss_curves_mpg.png**
   - Stored loss curve visualization for MPG training.

### 10. **train_mnist.py**
   - Script to train the MLP model using the MNIST dataset.

### 11. **train_mpg.py**
   - Script to train the MLP model using the MPG dataset.

## Usage

### Training the MLP Model
Run the following scripts to train the model:

- **For MNIST dataset:**
  ```sh
  python train_mnist.py
  ```
- **For MPG dataset:**
  ```sh
  python train_mpg.py
  ```

### Visualizing Loss Curves
- Open `loss_curves_mnist.png` and `loss_curves_mpg.png` to view the training loss curves.

## Dependencies
Ensure you have the necessary dependencies installed

## Author
- **Kshitiz Dhungana**

