import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Body Fat Prediction", layout="centered")

# ------------------------------ #
# Load Model Artifacts
# ------------------------------ #
@st.cache_resource
def load_artifacts():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return model, scaler, feature_columns


model, scaler, feature_columns = load_artifacts()

# ------------------------------ #
# Page Title
# ------------------------------ #
st.markdown(
    """
    <h1 style="text-align:center; color:#2E8B57;">
         Body Fat Prediction
    </h1>
    <p style="text-align:center; font-size:18px; color:gray;">
        Enter your measurements to estimate body fat percentage
    </p>
    <br>
    """,
    unsafe_allow_html=True,
)

# ------------------------------ #
# Input Form
# ------------------------------ #
st.subheader("Enter Input Features")

col1, col2 = st.columns(2)

user_data = []

for i, col in enumerate(feature_columns):
    container = col1 if i % 2 == 0 else col2
    with container:
        value = st.number_input(f"{col}", min_value=0.0, step=0.01)
        user_data.append(value)

# Convert input to array
input_array = np.array(user_data).reshape(1, -1)

# Scale
scaled_input = scaler.transform(input_array)

# ------------------------------ #
# Predict Button
# ------------------------------ #
st.write("")
center_btn = st.columns(3)
with center_btn[1]:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

if predict_btn:
    prediction = model.predict(scaled_input)[0]

    st.markdown(
        f"""
        <div style="
            margin-top: 25px;
            background: #2E8B57;
            padding: 20px;
            border-radius: 12px;
            color: white;
            font-size: 22px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        ">
            Predicted Body Fat: <b>{prediction:.2f}%</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


