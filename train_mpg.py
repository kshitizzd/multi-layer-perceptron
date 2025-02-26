import numpy as np
from matplotlib import pyplot as plt
from mlp import *
import os
from datetime import datetime

# Load preprocessed data
data = np.load('preprocessed_data/mpg_data.npz')
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
y_mean, y_std = data['y_mean'], data['y_std']  # Load the scaling parameters

# Define the network layers
layers = [
    Layer(fan_in=7, fan_out=32, activation_function=Relu(), dropout_rate=0.02),
    Layer(fan_in=32, fan_out=16, activation_function=Relu(), dropout_rate=0.01),
    Layer(fan_in=16, fan_out=1, activation_function=Linear(), dropout_rate=0)
]

# Initialize the MLP with the defined layers
mlp = MultilayerPerceptron(layers=layers)

# Train the model
training_losses, validation_losses = mlp.train(
    X_train, y_train, X_val, y_val,
    loss_func=SquaredError(),
    learning_rate=0.001,
    epochs=150,
    batch_size=16,
    rmsProp=True,
)

# Save losses to disk
np.save("./data/losses_mpg.npy", {"training_losses": training_losses, "validation_losses": validation_losses})

# Evaluate the model
r2, mse, mae, predictions = mlp.evaluate_mlp_regression(X_test, y_test)
print(f"R² Score (Accuracy): {r2:.2%}")

# Plot the losses
losses = np.load("data/losses_mpg.npy", allow_pickle=True).item()
training_losses = losses["training_losses"]
validation_losses = losses["validation_losses"]

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss Curves")
plt.savefig("loss_curves_mpg.png")
plt.close()

# Create a directory for results if it doesn't exist
if not os.path.exists('mpg_table_results'):
    os.makedirs('mpg_table_results')

# Get timestamp for unique filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Select 10 random samples and compare predictions with actual values
np.random.seed(42)  # for reproducibility
sample_indices = np.random.choice(len(X_test), 10, replace=False)

# Get predictions for these samples
sample_predictions = predictions[sample_indices]
sample_true_values = y_test[sample_indices]

# Prepare the data for saving
true_mpg_values = [(true.item() * y_std) + y_mean for true in sample_true_values]
pred_mpg_values = [(pred.item() * y_std) + y_mean for pred in sample_predictions]

# Save as formatted text file
txt_filename = f'mpg_table_results/mpg_predictions_{timestamp}.txt'
with open(txt_filename, 'w') as f:
    # Write the metrics first
    f.write(f"R² Score (Accuracy): {r2:.2%}\n")
    f.write(f"Total Testing Loss (MSE): {mse:.4f}\n\n")
    # Write the comparison table
    f.write("Predicted vs True MPG Comparison:\n")
    f.write("-" * 45 + "\n")
    f.write(f"{'Sample #':^10}{'True MPG':^15}{'Predicted MPG':^20}\n")
    f.write("-" * 45 + "\n")
    for i, (true, pred) in enumerate(zip(true_mpg_values, pred_mpg_values)):
        f.write(f"{i + 1:^10}{true:^15.2f}{pred:^20.2f}\n")
    f.write("-" * 45 + "\n")

# Display the table in console
print(f"\nTotal Testing Loss (MSE): {mse:.4f}")
print("\nPredicted vs True MPG Comparison:")
print("-" * 45)
print(f"{'Sample #':^10}{'True MPG':^15}{'Predicted MPG':^20}")
print("-" * 45)
for i, (true, pred) in enumerate(zip(true_mpg_values, pred_mpg_values)):
    print(f"{i + 1:^10}{true:^15.2f}{pred:^20.2f}")
print("-" * 45)

print(f"\nResults saved to: {txt_filename}")
plt.show()