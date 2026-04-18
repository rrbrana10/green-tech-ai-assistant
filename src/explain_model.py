import os
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Loading Surrogate Model...")
model_path = os.path.join('models', 'surrogate_model.keras')
if not os.path.exists(model_path):
    print("Error: Train model first by running src/train_model.py")
    exit(1)

model = tf.keras.models.load_model(model_path)

print("Loading subset of data for SHAP Explainer...")
try:
    df = pd.read_csv('ENB2012_data.csv')
except FileNotFoundError:
    print("Error: ENB2012_data.csv not found.")
    exit(1)

# Select features and generate a representative background set to use for KernelExplainer
feature_names = ['Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 
                 'Height', 'Orientation', 'Glazing Area', 'Glazing Dist']
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values

# We utilize shap.sample to grab 50 representative data points as background
# to speed up Kernel explainer (which can be very slow)
background = shap.sample(X, 50)
test_instances = shap.sample(X, 100) # what we are explaining

# Wrapping Keras prediction logic natively
predict_func = lambda x: model.predict(x, verbose=0)

# Initialize KernelExplainer (agnostic model explainer)
print("Initializing SHAP Kernel Explainer... This may take a few seconds.")
explainer = shap.KernelExplainer(predict_func, background)

print("Calculating SHAP values for summary visualization...")
shap_values = explainer.shap_values(test_instances)

# shap_values is a list for Multi-output.
# index 0 -> Heating Load (Y1)
# index 1 -> Cooling Load (Y2)

os.makedirs('outputs', exist_ok=True)

print("Generating SHAP summary plot for Heating Load (Y1)")
plt.figure()
shap_vals_Y1 = shap_values[:, :, 0] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 else shap_values[0]
shap.summary_plot(shap_vals_Y1, test_instances, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join("outputs", 'shap_heating_load.png'))

print("Generating SHAP summary plot for Cooling Load (Y2)")
plt.figure()
shap_vals_Y2 = shap_values[:, :, 1] if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 else shap_values[1]
shap.summary_plot(shap_vals_Y2, test_instances, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join("outputs", 'shap_cooling_load.png'))

print("\n--- SHAP EXPLANATION COMPLETE ---")
print("SHAP Summary plots have been generated and saved to the 'outputs' directory.")
print("Look for 'shap_heating_load.png' and 'shap_cooling_load.png'.")
print("These plots illustrate how changes in features organically drive energy loads up or down.")
