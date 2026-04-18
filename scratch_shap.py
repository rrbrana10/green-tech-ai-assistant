import os
import pandas as pd
import numpy as np
import tensorflow as tf
import shap

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = os.path.join('models', 'surrogate_model.keras')
model = tf.keras.models.load_model(model_path)

df = pd.read_csv('ENB2012_data.csv')
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']].values

background = shap.sample(X, 10)
test_instances = shap.sample(X, 5)

predict_func = lambda x: model.predict(x, verbose=0)
explainer = shap.KernelExplainer(predict_func, background)
shap_values = explainer.shap_values(test_instances)

print(type(shap_values))
if isinstance(shap_values, list):
    print("List length:", len(shap_values))
    print("Shape of element 0:", np.array(shap_values[0]).shape)
else:
    print("Base shape:", np.array(shap_values).shape)
