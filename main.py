from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "ipynd_checkpoints"
    / "best_diabetes_model.joblib"
)

FEATURES = {
    "age": ("Age", "Age feature, standardized as in the training dataset."),
    "sex": ("Sex", "Sex feature, standardized as in the training dataset."),
    "bmi": ("BMI", "Body mass index feature, standardized."),
    "bp": ("Blood pressure", "Average blood pressure feature, standardized."),
    "s1": ("Total cholesterol (s1)", "Total serum cholesterol feature, standardized."),
    "s2": ("LDL (s2)", "Low-density lipoproteins feature, standardized."),
    "s3": ("HDL (s3)", "High-density lipoproteins feature, standardized."),
    "s4": ("Cholesterol ratio (s4)", "Total cholesterol / HDL feature, standardized."),
    "s5": ("Triglycerides (s5)", "Log of serum triglycerides feature, standardized."),
    "s6": ("Blood sugar (s6)", "Blood sugar feature, standardized."),
}


@st.cache_resource
def load_model() -> Pipeline:
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Diabetes Progression Estimator", page_icon="🩺")
st.title("Diabetes Progression Estimator")
st.write(
    "Enter the ten features used by the trained model to estimate a quantitative "
    "measure of diabetes disease progression one year after baseline."
)

st.info(
    "The notebook uses scikit-learn's diabetes dataset, whose input features are "
    "standardized and unitless. Enter values on that same scale; these are not "
    "raw clinical measurements."
)

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()
model_features = tuple(getattr(model, "feature_names_in_", ()))
if model_features != tuple(FEATURES):
    st.error(
        "The model's feature schema does not match this app. "
        f"Expected {tuple(FEATURES)}, got {model_features}."
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Patient features")
    columns = st.columns(2)
    values: dict[str, float] = {}

    for index, (feature, (label, help_text)) in enumerate(FEATURES.items()):
        with columns[index % len(columns)]:
            values[feature] = st.number_input(
                label,
                value=0.0,
                step=0.001,
                format="%.4f",
                help=help_text,
            )

    submitted = st.form_submit_button("Estimate progression")

if submitted:
    input_data = pd.DataFrame([values], columns=model_features)
    prediction = float(model.predict(input_data)[0])
    st.subheader("Model estimate")
    st.metric("Estimated progression score", f"{prediction:.2f}")
    st.caption(
        "This is a regression score, not a diabetes diagnosis, risk probability, "
        "or substitute for medical advice."
    )
