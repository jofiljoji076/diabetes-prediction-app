import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# Page Configuration
st.set_page_config(page_title="Diabetes Risk Assessment App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Prediction App")
st.write("This app uses machine learning (Random Forest) trained on the Pima Indians Diabetes Dataset to assess diabetes risk based on medical metrics.")

# -------------------------------------------------------------
# CACHED DATA LOADING & MODEL TRAINING
# -------------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv(url, names=column_names)
    
    # Preprocessing: Replace 0s with median
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        df[col].fillna(df[col].median(), inplace=True)

    X = df[['Glucose', 'BMI', 'Insulin', 'Age', 'Pregnancies', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction']]
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Calculate performance metrics for display
    y_pred_rf = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    recall = recall_score(y_test, y_pred_rf)

    return rf_model, accuracy, recall

rf_model, model_accuracy, model_recall = load_and_train_model()

# Sidebar metrics info
st.sidebar.header("📊 Model Performance")
st.sidebar.metric(label="Random Forest Accuracy", value=f"{model_accuracy * 100:.2f}%")
st.sidebar.metric(label="Random Forest Recall", value=f"{model_recall * 100:.2f}%")

# -------------------------------------------------------------
# INTERACTIVE USER INPUT FORM
# -------------------------------------------------------------
st.subheader("Enter Patient Medical Details:")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
        glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
        
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=900.0, value=85.0)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.5)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age (Years)", min_value=1, max_value=120, value=35)

    submitted = st.form_submit_button("Predict Risk")

# -------------------------------------------------------------
# PREDICTION & RESULTS DISPLAY
# -------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame([[
        glucose, bmi, insulin, age, pregnancies, blood_pressure, skin_thickness, dpf
    ]], columns=['Glucose', 'BMI', 'Insulin', 'Age', 'Pregnancies', 'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction'])

    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)

    st.markdown("---")
    st.subheader("📋 Diagnosis Result")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **Result: Positive for Diabetes (Risk Detected)**")
        st.write(f"**Confidence:** {probability[0][1]*100:.2f}%")
    else:
        st.success(f"✅ **Result: Negative for Diabetes (Low Risk)**")
        st.write(f"**Confidence:** {probability[0][0]*100:.2f}%")
