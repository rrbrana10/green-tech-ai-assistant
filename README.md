# Green-Tech AI Design Assistant

A two-part AI system acting as an energy efficiency expert and generative engine to optimize building designs based on the UCI Energy Efficiency Dataset.

## Features

1. **Surrogate Model**: A TensorFlow/Keras Dense Neural Network trained to predict Heating Load (Y1) and Cooling Load (Y2) based on 8 building parameters (X1-X8).
2. **Generative Engine**: A PyGAD-based genetic algorithm that uses the surrogate model as a fitness function to evolve and optimize building parameters to minimize overall energy impact.
3. **Explainable AI (SHAP)**: Integration with SHAP to explain why particular designs are considered "green" by revealing feature importance and relationships.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure you have the dataset `ENB2012_data.csv` in the root folder.
3. Train the Keras model:
   ```bash
   python src/train_model.py
   ```
4. Run the generative optimization to find the most efficient building parameters:
   ```bash
   python src/optimize_design.py
   ```
5. Explain the model's predictions using SHAP:
   ```bash
   python src/explain_model.py
   ```
