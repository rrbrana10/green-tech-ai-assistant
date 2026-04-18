import streamlit as st
import numpy as np
import tensorflow as tf
import pygad
import os
from PIL import Image

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(page_title="Green-Tech AI Design", layout="wide")

st.sidebar.title("Green-Tech AI Assistant")
st.sidebar.write("This tool uses Deep Learning and Genetic Algorithms to act as an energy expert for building design.")

# Load Model
@st.cache_resource
def load_surrogate_model():
    model_path = os.path.join('models', 'surrogate_model.keras')
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_surrogate_model()

if model is None:
    st.error("Model not found! Please run `python src/train_model.py` first.")
    st.stop()

st.title("🏡 Green-Tech AI Design Assistant")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🧬 Optimization", "🧠 Explainability"])

# ----------------- TAB 1: PREDICTION -----------------
with tab1:
    st.header("Manual Parameter Prediction")
    st.write("Tweak the building parameters below to see the instant Heating and Cooling load predictions.")
    
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.slider("Relative Compactness", 0.62, 0.98, 0.74, step=0.01)
        x2 = st.slider("Surface Area", 514.5, 808.5, 686.0, step=0.5)
        x3 = st.slider("Wall Area", 245.0, 416.5, 245.0, step=0.5)
        x4 = st.slider("Roof Area", 110.25, 220.5, 220.5, step=0.25)
    with col2:
        x5 = st.slider("Overall Height", 3.5, 7.0, 3.5, step=0.5)
        x6 = st.selectbox("Orientation", [2, 3, 4, 5])
        x7 = st.slider("Glazing Area", 0.0, 0.4, 0.0, step=0.05)
        x8 = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4, 5])

    if st.button("Predict Energy Load", type="primary"):
        input_data = np.array([[x1, x2, x3, x4, x5, x6, x7, x8]])
        prediction = model.predict(input_data, verbose=0)[0]
        
        st.success("Prediction Complete!")
        m1, m2 = st.columns(2)
        m1.metric("🔥 Heating Load", f"{prediction[0]:.2f} kWh/m²")
        m2.metric("❄️ Cooling Load", f"{prediction[1]:.2f} kWh/m²")


# ----------------- TAB 2: OPTIMIZATION -----------------
with tab2:
    st.header("PyGAD Generative Design")
    st.write("Let the AI evolve the absolute best building parameters to **minimize** combined energy loads.")
    
    if st.button("🚀 Run PyGAD Optimizer", type="primary"):
        with st.spinner("Evolving architectures... (This takes a few seconds)"):
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
                pred = model.predict(np.expand_dims(solution, axis=0), verbose=0)[0]
                return 1.0 / (pred[0] + pred[1] + 1e-6)

            ga_instance = pygad.GA(
                num_generations=50,
                num_parents_mating=10,
                fitness_func=fitness_func,
                sol_per_pop=50,
                num_genes=8,
                gene_space=gene_space,
                mutation_percent_genes=15,
                crossover_type="random", 
                mutation_type="random",
                suppress_warnings=True
            )
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            
            optimal_load = (1.0 / solution_fitness) - 1e-6
            
            st.success("Evolution Complete!")
            st.write(f"### Minimal Combined Energy Projection: **{optimal_load:.2f} kWh/m²**")
            
            st.write("### The Greenest Parameters:")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Compactness", f"{solution[0]:.2f}")
            c2.metric("Surface Area", f"{solution[1]:.2f}")
            c3.metric("Wall Area", f"{solution[2]:.2f}")
            c4.metric("Roof Area", f"{solution[3]:.2f}")
            c1.metric("Overall Height", f"{solution[4]:.2f}")
            c2.metric("Orientation", f"{int(solution[5])}")
            c3.metric("Glazing Area", f"{solution[6]:.2f}")
            c4.metric("Glazing Dist", f"{int(solution[7])}")

# ----------------- TAB 3: EXPLAINABILITY -----------------
with tab3:
    st.header("SHAP Explainable AI Insights")
    st.write("Understanding **why** the AI makes its decisions is crucial for 'Green' certification.")
    
    heating_img_path = os.path.join('outputs', 'shap_heating_load.png')
    cooling_img_path = os.path.join('outputs', 'shap_cooling_load.png')
    
    if os.path.exists(heating_img_path) and os.path.exists(cooling_img_path):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Heating Load Drivers")
            st.image(Image.open(heating_img_path), use_container_width=True)
            st.caption("A higher relative compactness / larger area drives the heating load differently.")
        with c2:
            st.subheader("Cooling Load Drivers")
            st.image(Image.open(cooling_img_path), use_container_width=True)
            st.caption("How varying Glazing Area distributions impact modern cooling loads.")
    else:
        st.warning("SHAP visualizations not found. Please run `python src/explain_model.py` to generate them.")
