import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the Auto MPG dataset directly
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin',
                'car_name']
data = pd.read_csv(url, delim_whitespace=True, names=column_names, na_values='?')

# Drop the car_name column
data = data.drop('car_name', axis=1)

# Remove rows with missing values
cleaned_data = data.dropna()

# Split features and target
X = cleaned_data.drop('mpg', axis=1)
y = cleaned_data['mpg']

rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

# Convert to numpy arrays
X = X.values
y = y.values

# Split into training, validation, and test sets
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    shuffle=True
)

X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

# Standardize the data
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_mean = y_train.mean()
y_std = y_train.std()
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


# Save preprocessed data to disk
np.savez('preprocessed_data/mpg_data.npz',
         X_train=X_train, X_val=X_val, X_test=X_test,
         y_train=y_train, y_val=y_val, y_test=y_test,
         X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)