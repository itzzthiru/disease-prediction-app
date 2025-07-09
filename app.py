import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

#Predict Function (used by all 3)
def predict(model, scaler, input_data):
    x = np.array(input_data).reshape(1, -1)
    x_scaled = scaler.transform(x)
    return model.predict(x_scaled)[0]


# Load all models and scalers
@st.cache_resource
def load_models():
    return {
        "Parkinsons": {
            "model": pickle.load(open("models/parkinsons_model.pkl", "rb")),
            "scaler": pickle.load(open("models/parkinsons_scaler.pkl", "rb"))
        },
        "Kidney": {
            "model": pickle.load(open("models/kidney_model.pkl", "rb")),
            "scaler": pickle.load(open("models/kidney_scaler.pkl", "rb"))
        },
        "Liver": {
            "model": pickle.load(open("models/liver_model.pkl", "rb")),
            "scaler": pickle.load(open("models/liver_scaler.pkl", "rb"))
        }
    }

models = load_models()

#Sidebar Drawer Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Multiple Disease Prediction",
        options=["🏠 Project Intro", "🧠 Parkinson’s", "🩸 Kidney", "🧪 Liver", "👨‍🔬 Creator Info"],
        icons=["house", "activity", "droplet", "flask", "person-circle"],
        default_index=0
    )

#Project Intro Page
if selected == "🏠 Project Intro":
    st.title("🧬 Multiple Disease Prediction System")
    st.write("""
    Welcome to the Multiple Disease Prediction App!  
    This tool allows you to predict the likelihood of three major diseases:
    
    - 🧠 Parkinson's Disease  
    - 🩸 Chronic Kidney Disease  
    - 🧪 Liver Disease

    Built using **Machine Learning**, **XGBoost**, and **Streamlit** 💻.
    """)

#Parkinson’s Page
elif selected == "🧠 Parkinson’s":
    st.title("🧠 Parkinson’s Disease Prediction")
    st.subheader("🧠 Parkinson’s Disease Prediction")

    features = [
        st.slider("MDVP:Fo(Hz)", 60.0, 300.0, 120.0),  # slider
        st.slider("MDVP:Fhi(Hz)", 80.0, 400.0, 160.0),  # slider
        st.number_input("MDVP:Flo(Hz)", min_value=60.0, max_value=300.0, value=110.0),  # number input
        st.number_input("MDVP:Jitter(%)", min_value=0.0001, max_value=0.2, value=0.01),  # number input
        st.slider("MDVP:Shimmer", 0.0, 0.2, 0.03),  # slider
        st.number_input("NHR (Noise to Harmonic Ratio)", 0.0, 1.0, 0.02),  # number input
        st.slider("HNR (Harmonics to Noise Ratio)", 0.0, 50.0, 20.0),  # slider
        st.slider("RPDE", 0.1, 1.0, 0.5),  # slider
        st.number_input("DFA", 0.4, 1.0, 0.7),  # number input
        st.slider("Spread1", -7.0, -1.0, -4.5),  # slider
        st.number_input("PPE (Pitch Period Entropy)", 0.0, 1.0, 0.3)  # number input
    ]

    if st.button("🔍 Predict"):
        out = predict(models["Parkinsons"]["model"], models["Parkinsons"]["scaler"], features)
        st.success("✅ No Parkinson's Disease" if out == 0 else "❌ Parkinson's Detected")

#Kidney Page
elif selected == "🩸 Kidney":
    st.title("🩸 Kidney Disease Prediction")
    st.subheader("🩸 Kidney Disease Prediction")

    features = [
        st.slider("Age", 1, 100, 45),  # slider
        st.number_input("Blood Pressure (mmHg)", 50, 200, 80),  # number input
        st.slider("Specific Gravity", 1.005, 1.030, 1.015),  # slider
        st.selectbox("Albumin", options=[0, 1, 2, 3, 4, 5]),  # dropdown
        st.selectbox("Sugar", options=[0, 1, 2, 3, 4, 5]),  # dropdown
        1 if st.radio("Red Blood Cells", ["Normal", "Abnormal"]) == "Normal" else 0,  # radio
        1 if st.radio("Pus Cells", ["Normal", "Abnormal"]) == "Normal" else 0,  # radio
        1 if st.radio("Pus Cell Clumps", ["Present", "Not Present"]) == "Present" else 0,  # radio
        1 if st.radio("Bacteria", ["Present", "Not Present"]) == "Present" else 0,  # radio
        st.number_input("Blood Glucose Random (mg/dL)", 50, 500, 120),  # number input
        st.slider("Blood Urea (mg/dL)", 10, 300, 40),  # slider
        st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 1.2)  # number input
    ]  

    if st.button("🔍 Predict"):
        out = predict(models["Kidney"]["model"], models["Kidney"]["scaler"], features)
        st.success("✅ No Kidney Disease" if out == 0 else "❌ Kidney Disease Detected")

#Liver Page
elif selected == "🧪 Liver":
    st.title("🧪 Liver Disease Prediction")
    st.subheader("🧪 Liver Disease Prediction")
    
    age = st.slider("Age", 1, 100, 45)  # slider
    gender = st.radio("Gender", ["Male", "Female"])  # radio → encoded later
    tb = st.number_input("Total Bilirubin (mg/dL)", 0.0, 75.0, 1.0)  # number input
    db = st.slider("Direct Bilirubin (mg/dL)", 0.0, 20.0, 0.5)  # slider
    alt = st.slider("ALT (SGPT)", 1, 2000, 30)  # slider
    ast = st.slider("AST (SGOT)", 1, 3000, 40)  # slider
    tp = st.number_input("Total Proteins (g/dL)", 1.0, 10.0, 6.5)  # number input 
    alb = st.slider("Albumin (g/dL)", 0.5, 6.0, 3.5)  # slider
    agr = st.number_input("Albumin/Globulin Ratio", 0.1, 3.0, 1.1)  # number input

    # Calculated feature
    ast_alt_ratio = ast / (alt + 1)

    features = [
        age,
        1 if gender == "Male" else 0,
        tb, db, alt, ast,
        tp, alb, agr,
        ast_alt_ratio
    ]

    if st.button("🔍 Predict"):
        out = predict(models["Liver"]["model"], models["Liver"]["scaler"], features)
        st.success("✅ No Liver Disease" if out == 0 else "❌ Liver Disease Detected")

#Creator Info Page
elif selected == "👨‍🔬 Creator Info":
    st.title("👨‍💻 Project Created By")
    st.write("""
    **Name:** Thirukumaran    
    **GitHub:** https://github.com/itzzthiru    
    """)


    
