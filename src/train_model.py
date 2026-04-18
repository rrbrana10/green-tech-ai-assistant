import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Load the data
print("Loading data...")
try:
    df = pd.read_csv('ENB2012_data.csv')
except FileNotFoundError:
    print("Error: ENB2012_data.csv not found in the root directory. Please add it first.")
    exit(1)

# X1-X8 are features, Y1 and Y2 are targets
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values
y = df[['Y1', 'Y2']].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the Keras Model via tf.keras
print("Building and training the Surrogate Model...")

# We use a Normalization layer to standardize inputs directly inside the model.
# This avoids the need to save external scalers (like joblib/pickle files).
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8,)),
    normalizer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2) # Output layer (Linear activation for regression of Y1 and Y2)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), 
              loss='mse',
              metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    verbose=1
)

# 3. Evaluate the model
print("\nEvaluating model on test data...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error: {mse:.4f}")
print(f"Test R^2 Score: {r2:.4f}")

# 4. Save the model
model_path = os.path.join('models', 'surrogate_model.keras')
model.save(model_path)
print(f"\nModel saved successfully to {model_path}.")
