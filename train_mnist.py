from load_mnist import one_hot_encode
from mlp import *
import matplotlib.pyplot as plt

# Load preprocessed data
data = np.load('preprocessed_data/mnist_data.npz')
X_train, X_val, y_train, y_val = data['X_train'], data['X_val'], data['y_train'], data['y_val']
test_images, test_labels = data['test_images'], data['test_labels']

# Define the network layers
layers = [
    Layer(fan_in=784, fan_out=128, activation_function=Relu()),
    Layer(fan_in=128, fan_out=64, activation_function=Relu()),
    Layer(fan_in=64, fan_out=32, activation_function=Relu()),
    Layer(fan_in=32, fan_out=10, activation_function=Softmax())
]

# Initialize and train the MLP
mlp = MultilayerPerceptron(layers=layers)
training_losses, validation_losses = mlp.train(X_train, y_train, X_val, y_val, loss_func=SquaredError(), learning_rate=0.001, epochs=20, batch_size=32, rmsProp=True)

# Save losses to disk
np.save("./data/losses_mnist.npy", {"training_losses": training_losses, "validation_losses": validation_losses})

# Evaluate the model
test_accuracy, pred_labels = mlp.evaluate_mlp_classification(test_images, one_hot_encode(test_labels))
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
validation_accuracy, pred = mlp.evaluate_mlp_classification(X_val, y_val)
print(f'Validation Accuracy: {validation_accuracy*100:.2f}%')

# Plot the losses
losses = np.load("data/losses_mnist.npy", allow_pickle=True).item()
training_losses = losses["training_losses"]
validation_losses = losses["validation_losses"]

# Loss curves
plt.figure(1)
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Curves")
plt.savefig("loss_curves_mnist.png")
plt.close()

# Digit samples
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()
# For each digit (0-9)
for digit in range(10):
    # Find indices where true label is the current digit
    digit_indices = np.where(test_labels == digit)[0]
    if len(digit_indices) > 0:
        idx = digit_indices[0]
        # Reshape image to 28x28
        img = test_images[idx].reshape(28, 28)
        # Predicted class
        test_sample = test_images[idx].reshape(1, -1)
        pred = np.argmax(mlp.forward(test_sample))
        # Plot
        axes[digit].imshow(img, cmap='gray')
        axes[digit].axis('off')
        axes[digit].set_title(f'True: {digit}\nPred: {pred}')
plt.tight_layout()
plt.savefig('mnist_samples.png')
plt.close()
