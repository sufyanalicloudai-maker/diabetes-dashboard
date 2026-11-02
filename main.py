from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.pipeline import Pipeline


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "ipynd_checkpoints"
    / "best_diabetes_model.joblib"
)
DIABETES_DATA = load_diabetes(as_frame=True)
RAW_DIABETES_DATA = load_diabetes(as_frame=True, scaled=False)

FEATURES = {
    "age": ("Age", "Age feature, standardized as in the training dataset."),
    "sex": ("Sex", "Select the patient's sex."),
    "bmi": ("BMI", "Body mass index feature, standardized."),
    "bp": ("Blood pressure", "Average blood pressure feature, standardized."),
    "s1": ("Total cholesterol (s1)", "Total serum cholesterol feature, standardized."),
    "s2": ("LDL (s2)", "Low-density lipoproteins feature, standardized."),
    "s3": ("HDL (s3)", "High-density lipoproteins feature, standardized."),
    "s4": ("Cholesterol ratio (s4)", "Total cholesterol / HDL feature, standardized."),
    "s5": ("Triglycerides (s5)", "Log of serum triglycerides feature, standardized."),
    "s6": ("Blood sugar (s6)", "Blood sugar feature, standardized."),
}

SEX_VALUES = {
    "Male": float(
        DIABETES_DATA.data.loc[RAW_DIABETES_DATA.data["sex"] == 1, "sex"].iloc[0]
    ),
    "Female": float(
        DIABETES_DATA.data.loc[RAW_DIABETES_DATA.data["sex"] == 2, "sex"].iloc[0]
    ),
}
SAMPLE_VALUES = DIABETES_DATA.data.iloc[0].to_dict()
SAMPLE_SEX = (
    "Male"
    if RAW_DIABETES_DATA.data.iloc[0]["sex"] == 1
    else "Female"
)


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
    selected_sex = SAMPLE_SEX

    for index, (feature, (label, help_text)) in enumerate(FEATURES.items()):
        with columns[index % len(columns)]:
            if feature == "sex":
                selected_sex = st.selectbox(
                    label,
                    options=tuple(SEX_VALUES),
                    index=tuple(SEX_VALUES).index(SAMPLE_SEX),
                    help=help_text,
                )
                values[feature] = SEX_VALUES[selected_sex]
            else:
                values[feature] = st.number_input(
                    label,
                    value=float(SAMPLE_VALUES[feature]),
                    step=0.001,
                    format="%.4f",
                    help=help_text,
                )

    submitted = st.form_submit_button("Estimate progression")

input_data = pd.DataFrame([values], columns=model_features)
prediction = float(model.predict(input_data)[0])

st.subheader("Patient report dashboard" if submitted else "Sample report dashboard")
metric_columns = st.columns(3)
metric_columns[0].metric("Estimated progression score", f"{prediction:.2f}")
metric_columns[1].metric("Sex", selected_sex)
metric_columns[2].metric("Features provided", str(len(values)))

profile = pd.DataFrame(
    {
        "Feature": [FEATURES[feature][0] for feature in model_features],
        "Standardized value": [values[feature] for feature in model_features],
    }
)
st.caption("Feature profile (standardized model inputs)")
st.bar_chart(profile.set_index("Feature"))
with st.expander("View entered feature values"):
    st.dataframe(profile, hide_index=True, width="stretch")

st.caption(
    "This is a regression score, not a diabetes diagnosis, risk probability, "
    "or substitute for medical advice."
)
