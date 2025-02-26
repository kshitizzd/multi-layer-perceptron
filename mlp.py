import numpy as np
from typing import Tuple

def batch_generator(train_x, train_y, batch_size):
    number_of_samples = train_x.shape[0]
    indices = np.arange(number_of_samples)
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]
    for start in range(0, number_of_samples, batch_size):
        end = min(start + batch_size, number_of_samples) 
        batch_x = train_x[start : end]
        batch_y = train_y[start : end]
        yield batch_x, batch_y
    pass


class ActivationFunction():
    def forward(self, x):
        pass

    def derivative(self, x):
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)
    pass


class Tanh(ActivationFunction):
    def forward(self, x):
        return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x):
        return 1 - np.power(self.forward(x), 2)
    pass


class Relu(ActivationFunction):
    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    pass


class Softmax(ActivationFunction):
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)
    pass


class Linear(ActivationFunction):
    def forward(self, x):
        return x 

    def derivative(self, x):
        return np.ones_like(x)
    pass

class SoftPlus(ActivationFunction):
    def forward(self, x):
        return np.log1p(np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))
    pass

class Mish(ActivationFunction):
    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def derivative(self, x):
        return self.forward(x) + np.tanh(x)
    pass

class LossFunction():
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        return 1/2 * np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        return y_pred - y_true
    pass


class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15) #To avoid log(0) for the calculation
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        return y_pred - y_true
    pass

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0.0):

        self.activations = None
        self.dropout = None
        self.delta = None

        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        self.W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in))         # Initialize weights using Glorot
        self.B = np.zeros((1, fan_out)) #

    def forward(self, h: np.ndarray):
        Z = np.dot(h, self.W)+self.B
        self.activations = self.activation_function.forward(Z)
        if self.dropout_rate > 0:
            self.dropout = (np.random.rand(*self.activations.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            self.activations *= self.dropout
        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.dropout_rate > 0:
            delta *= self.dropout
        dL_dz = delta * self.activation_function.derivative(self.activations)
        dL_dW = np.dot(h.T, dL_dz)
        dL_db = np.sum(dL_dz, axis = 0, keepdims = True)
        self.delta = np.dot(dL_dz, self.W.T)
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        self.layers = layers

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        dL_dw_all = []
        dL_db_all = []
        delta = loss_grad
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == 0:
                layer_input = input_data
            else:
                # Input for hidden layers is activation of L-1
                layer_input = self.layers[i-1].activations
            if delta.ndim == 1:
                delta = delta.reshape(-1, 1)
            dL_dw, dL_db = layer.backward(layer_input, delta)
            delta = layer.delta
            dL_db_all.append(dL_db)
            dL_dw_all.append(dL_dw)
        dL_db_all.reverse()
        dL_dw_all.reverse()
        return dL_db_all, dL_dw_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,loss_func: LossFunction,learning_rate: float = 1E-3, batch_size: int = 16, epochs: int = 32, rmsProp: bool = False) -> Tuple[
        np.ndarray, np.ndarray]:
        training_losses = []
        validation_losses = []
        cache_w = [np.zeros_like(layer.W) for layer in self.layers]
        cache_b = [np.zeros_like(layer.B) for layer in self.layers]
        beta2 = 0.999  # RMSprop hyperparameter for decay rate
        epsilon = 1e-8  # Stabilizer to prevent division by zero

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0.0
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                prediction = self.forward(batch_x)

                batch_loss = loss_func.loss(batch_y, prediction)
                gradient_loss = loss_func.derivative(batch_y, prediction)
                dl_db_all, dl_dw_all = self.backward(gradient_loss, batch_x)

                for i, layer in enumerate(self.layers):
                    if rmsProp:
                        # Apply RMSprop
                        cache_w[i] = beta2 * cache_w[i] + (1 - beta2) * (dl_dw_all[i] ** 2)
                        cache_b[i] = beta2 * cache_b[i] + (1 - beta2) * (dl_db_all[i] ** 2)
                        layer.W -= learning_rate * dl_dw_all[i] / (np.sqrt(cache_w[i]) + epsilon)
                        layer.B -= learning_rate * dl_db_all[i] / (np.sqrt(cache_b[i]) + epsilon)
                    else:
                        # Standard gradient descent
                        layer.W -= learning_rate * dl_dw_all[i]
                        layer.B -= learning_rate * dl_db_all[i]

                epoch_loss += batch_loss
                num_batches += 1.0

            train_loss = epoch_loss / num_batches

            valid_prediction = self.forward(val_x)
            valid_loss = loss_func.loss(val_y, valid_prediction)

            training_losses.append(train_loss)
            validation_losses.append(valid_loss)

            print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {valid_loss}')

        return np.array(training_losses), np.array(validation_losses)

    def evaluate_mlp_classification(self, X_test, y_test):
        y_pred = self.forward(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        accuracy = np.mean(y_pred_labels == y_true_labels)
        return accuracy, y_pred_labels

    def evaluate_mlp_regression(self, X_test, y_test):
        y_pred = self.forward(X_test)
        y_pred = y_pred.reshape(-1)
        y_true = y_test.reshape(-1)

        # Calculate MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # Calculate R-squared (coefficient of determination)
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        # Calculate Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))

        return r2, mse, mae, y_pred


