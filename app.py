import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# PATHS
# ======================================================

# 1. Dynamically find the root directory of your project
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Define the absolute paths to your models and images
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
COLUMN_PATH = BASE_DIR / "models" / "model_columns.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
IMAGE_DIR = BASE_DIR / "images"

# 3. Load the machine learning files safely by converting Path objects to strings
model = joblib.load(str(MODEL_PATH))
model_columns = joblib.load(str(COLUMN_PATH))
scaler = joblib.load(str(SCALER_PATH))

# ======================================================
# LOAD MODEL & PREPROCESSORS
# ======================================================

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMN_PATH)
scaler = joblib.load(SCALER_PATH)

# ======================================================
# HEADER
# ======================================================

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
Predict whether a telecom customer is likely to churn using a trained
Machine Learning model.

---
""")

# ======================================================
# MODEL INFORMATION
# ======================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Algorithm",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Accuracy",
        "80.45%"
    )

with col3:
    st.metric(
        "Features",
        len(model_columns)
    )

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Customer Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    ["No", "Yes"]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.sidebar.slider(
    "Tenure Months",
    0,
    72,
    24
)

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes",
        "No phone service"
    ]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    [
        "No",
        "Yes",
        "No internet service"
    ]
)

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    [
        "No",
        "Yes"
    ]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

predict = st.sidebar.button(
    "🔍 Predict Customer Churn"
)

# ======================================================
# PREDICTION
# ======================================================

if predict:

    # Create dataframe with all model columns
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=model_columns
    )

    # ----------------------------
    # Numerical Features
    # ----------------------------

    numeric_features = {
        "Tenure Months": tenure,
        "Monthly Charges": monthly,
        "Total Charges": total
    }

    for col, value in numeric_features.items():
        if col in input_df.columns:
            input_df[col] = value

    # ----------------------------
    # Categorical Features
    # ----------------------------

    user_values = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment
    }

    # Automatically create encoded columns
    for feature, value in user_values.items():

        column_name = f"{feature}_{value}"

        if column_name in input_df.columns:
            input_df[column_name] = 1

    # ----------------------------
    # Scale Numerical Features
    # ----------------------------
    numeric_columns = ["Tenure Months", "Monthly Charges", "Total Charges"]
    
    if all(col in input_df.columns for col in numeric_columns):
        scaled_values = scaler.transform(input_df[numeric_columns].values)
        input_df.loc[:, numeric_columns] = scaled_values

    # ----------------------------
    # Prediction
    # ----------------------------

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    # ======================================================
    # RESULTS
    # ======================================================

    st.markdown("---")

    st.header("Prediction Result")

    left, right = st.columns(2)

    with left:

        st.metric(
            "Churn Probability",
            f"{probability:.2%}"
        )

        st.progress(float(probability))

    with right:

        if prediction == 1:

            st.error(
                "⚠️ Customer is likely to churn."
            )

        else:

            st.success(
                "✅ Customer is likely to stay."
            )

    st.markdown("---")

    # ======================================================
    # CUSTOMER SUMMARY
    # ======================================================

    st.subheader("Customer Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Gender",
            "Senior Citizen",
            "Partner",
            "Dependents",
            "Tenure",
            "Internet",
            "Contract",
            "Monthly Charges",
            "Total Charges"
        ],
        "Value": [
            gender,
            senior,
            partner,
            dependents,
            tenure,
            internet,
            contract,
            monthly,
            total
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True
    )

    # ======================================================
    # DEBUG (Optional)
    # ======================================================

    with st.expander("View Model Input"):

        st.dataframe(input_df)

# ======================================================
# MODEL PERFORMANCE
# ======================================================

st.markdown("---")
st.header("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Confusion Matrix")

    cm_path = IMAGE_DIR / "confusion_matrix.png"

    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Run train.py to generate the confusion matrix.")

with col2:

    st.subheader("ROC Curve")

    roc_path = IMAGE_DIR / "roc_curve.png"

    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)
    else:
        st.info("Run train.py to generate the ROC curve.")

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

st.markdown("---")

st.header("⭐ Top Feature Importance")

feature_path = IMAGE_DIR / "feature_importance.png"

if feature_path.exists():
    st.image(str(feature_path), use_container_width=True)
else:
    st.info("Feature importance is only available for tree-based models such as Random Forest.")

# ======================================================
# ABOUT PROJECT
# ======================================================

st.markdown("---")

st.header("📘 About This Project")

st.markdown("""
### Customer Churn Prediction using Machine Learning

This project predicts whether a telecom customer is likely to churn based on customer demographics, service usage, and billing information.

### Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

### Machine Learning Models Compared

- Logistic Regression
- Decision Tree
- Random Forest

### Best Model

**Logistic Regression**

**Accuracy:** 80.45%

### Dataset

IBM Telco Customer Churn Dataset
""")

# ======================================================
# DATASET INFORMATION
# ======================================================

st.markdown("---")

st.header("📊 Dataset Statistics")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Customers", "7043")

with c2:
    st.metric("Features", "19")

with c3:
    st.metric("Target", "Churn")

with c4:
    st.metric("Best Accuracy", "80.45%")

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;color:gray;'>

    Developed as a Machine Learning Project using the
    <b>IBM Telco Customer Churn Dataset</b>.

    </div>
    """,
    unsafe_allow_html=True
)