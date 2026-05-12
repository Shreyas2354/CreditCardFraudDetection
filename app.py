# ==========================================
# Credit Card Fraud Detection System
# Professional Streamlit Web App
# ==========================================

# ------------------------------------------
# Import Libraries
# ------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------------
# Load Saved Model
# ------------------------------------------

model = joblib.load("models/fraud_model.pkl")

# ------------------------------------------
# App Title
# ------------------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
    Upload a transaction CSV file to detect fraudulent transactions
    using Machine Learning.
    """
)

# ------------------------------------------
# Sidebar
# ------------------------------------------

st.sidebar.header("📂 Upload Transaction File")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# ------------------------------------------
# Main Prediction Process
# ------------------------------------------

if uploaded_file is not None:

    # --------------------------------------
    # Read Uploaded CSV
    # --------------------------------------

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset")

    st.dataframe(df.head())

    # --------------------------------------
    # Save Original Data
    # --------------------------------------

    original_df = df.copy()

    # --------------------------------------
    # Remove Target Column if Exists
    # --------------------------------------

    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    # --------------------------------------
    # Feature Scaling
    # --------------------------------------

    scaler = StandardScaler()

    df["scaled_amount"] = scaler.fit_transform(df[["Amount"]])

    # Remove original Amount column
    df = df.drop("Amount", axis=1)

    # --------------------------------------
    # Final Input Features
    # --------------------------------------

    X = df

    # --------------------------------------
    # Make Predictions
    # --------------------------------------

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # --------------------------------------
    # Add Results to Original Dataset
    # --------------------------------------

    original_df["Prediction"] = predictions

    original_df["Fraud_Probability"] = probabilities[:, 1] * 100

    # --------------------------------------
    # Fraud Statistics
    # --------------------------------------

    fraud_count = int(original_df["Prediction"].sum())

    total_transactions = len(original_df)

    genuine_count = total_transactions - fraud_count

    fraud_percentage = (
        fraud_count / total_transactions
    ) * 100

    # --------------------------------------
    # Dashboard Metrics
    # --------------------------------------

    st.subheader("📊 Fraud Detection Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraud Transactions",
        fraud_count
    )

    col3.metric(
        "Genuine Transactions",
        genuine_count
    )

    col4.metric(
        "Fraud Percentage",
        f"{fraud_percentage:.2f}%"
    )

    # --------------------------------------
    # Final Prediction Result
    # --------------------------------------

    st.subheader("🔍 Final Fraud Analysis")

    if fraud_count > 0:

        st.error(
            f"⚠ ALERT: {fraud_count} Fraudulent Transactions Detected"
        )

    else:

        st.success(
            "✅ All Transactions are Genuine"
        )

    # --------------------------------------
    # Risk Level
    # --------------------------------------

    if fraud_percentage > 20:

        st.error("🔴 Risk Level: HIGH")

    elif fraud_percentage > 5:

        st.warning("🟠 Risk Level: MEDIUM")

    else:

        st.success("🟢 Risk Level: LOW")

    # --------------------------------------
    # Show Fraud Transactions
    # --------------------------------------

    fraud_transactions = original_df[
        original_df["Prediction"] == 1
    ]

    st.subheader("⚠ Fraudulent Transactions")

    if len(fraud_transactions) > 0:

        st.dataframe(fraud_transactions)

    else:

        st.info("No Fraudulent Transactions Found")

    # --------------------------------------
    # Prediction Results Table
    # --------------------------------------

    st.subheader("📋 Prediction Results")

    st.dataframe(original_df.head(20))

    # --------------------------------------
    # Download Results
    # --------------------------------------

    csv = original_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Prediction Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

# ------------------------------------------
# Footer
# ------------------------------------------

st.markdown("---")

st.markdown(
    """
    🚀 Machine Learning Based Fraud Detection System  
    Built using Streamlit, Random Forest & SMOTE
    """
)