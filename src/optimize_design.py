import os
import numpy as np
import tensorflow as tf
import pygad

# Suppress lengthy TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model_path = os.path.join('models', 'surrogate_model.keras')

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}. Please run train_model.py first.")
    exit(1)

print("Loading Surrogate Model...")
model = tf.keras.models.load_model(model_path)

# Genetic Algorithm Gene space configuration based on ENB2012 Dataset parameters:
# X1: Compactness (0.62 - 0.98),  X2: Surface Area (514.5 - 808.5)
# X3: Wall Area (245.0 - 416.5), X4: Roof Area (110.25 - 220.5)
# X5: Overall Height (3.5 - 7.0), X6: Orientation (2, 3, 4, 5)
# X7: Glazing Area (0.0 - 0.4),   X8: Glazing Dist (0, 1, 2, 3, 4, 5)
gene_space = [
    {'low': 0.62, 'high': 0.98},
    {'low': 514.5, 'high': 808.5},
    {'low': 245.0, 'high': 416.5},
    {'low': 110.25, 'high': 220.5},
    {'low': 3.5, 'high': 7.0},
    [2, 3, 4, 5],
    {'low': 0.0, 'high': 0.4},
    [0, 1, 2, 3, 4, 5]
]

def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness Function for PyGAD.
    PyGAD aims to MAXIMIZE the fitness value. 
    Since we aim to MINIMIZE energy parameters (Heating Y1 + Cooling Y2),
    we return the inverse.
    """
    # model.predict expects shape (batch, features)
    input_features = np.expand_dims(solution, axis=0)
    
    # Predict Y1 and Y2 locally
    pred = model.predict(input_features, verbose=0)[0]
    heating_load = pred[0]
    cooling_load = pred[1]
    
    # Combined load to minimize
    combined_load = heating_load + cooling_load
    
    # Avoid div-by-zero
    fitness = 1.0 / (combined_load + 1e-6)
    return fitness

# Setup PyGAD instance
print("Initializing Generative Engine (PyGAD)...")
ga_instance = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50,
    num_genes=8,
    gene_space=gene_space,
    mutation_percent_genes=15,
    crossover_type="random", # Mix properties robustly
    mutation_type="random",
    suppress_warnings=True
)

print("Starting evolutionary optimization (this might take a minute)...")
ga_instance.run()

# Evolve the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()

print("\n--- OPTIMIZATION COMPLETE ---")
print(f"Optimal Building Parameters Details:")
print(f"Relative Compactness: {solution[0]:.2f}")
print(f"Surface Area: {solution[1]:.2f}")
print(f"Wall Area: {solution[2]:.2f}")
print(f"Roof Area: {solution[3]:.2f}")
print(f"Overall Height: {solution[4]:.2f}")
print(f"Orientation (2-5): {solution[5]}")
print(f"Glazing Area: {solution[6]:.2f}")
print(f"Glazing Area Distribution: {solution[7]}")

# Reverse calculate energy loads
optimal_load = (1.0 / solution_fitness) - 1e-6
print(f"\nMinimum Optimal Combined Energy Prediction (Y1+Y2): {optimal_load:.2f} kWh/m2")

ga_instance.plot_fitness(save_dir=os.path.join('outputs', 'fitness_evolution.png'))
print(f"View the fitness evolution plot at outputs/fitness_evolution.png")
