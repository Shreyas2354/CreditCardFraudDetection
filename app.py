
# ==========================================
# FraudShield AI - Advanced Fraud Detection
# Professional Streamlit UI
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/fraud_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #060B26;
    color: white;
}

/* MAIN AREA */
.main {
    background: linear-gradient(to right, #050816, #0b1437);
    color: white;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1026, #111C44);
    border-right: 1px solid rgba(255,255,255,0.1);
}

.sidebar-title {
    font-size: 30px;
    font-weight: bold;
    color: white;
}

.sidebar-sub {
    color: #9CA3AF;
    font-size: 14px;
}

/* TITLE */
.main-title {
    font-size: 65px;
    font-weight: 800;
    line-height: 1.1;
    color: white;
}

.gradient-text {
    background: linear-gradient(90deg,#4FACFE,#C471ED);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CARD */
.custom-card {
    background: rgba(255,255,255,0.04);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

/* METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    text-align: center;
}

.metric-title {
    color: #9CA3AF;
    font-size: 14px;
}

.metric-value {
    font-size: 38px;
    font-weight: bold;
    color: white;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    height: 65px;
    border-radius: 16px;
    border: none;
    font-size: 22px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg,#9333EA,#3B82F6);
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg,#A855F7,#2563EB);
}

/* INPUTS */
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stSlider {
    background-color: rgba(255,255,255,0.05) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* RESULT BOX */
.result-box {
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

.high-risk {
    background: rgba(239,68,68,0.15);
    border: 2px solid #EF4444;
    color: #FF4D4D;
}

.low-risk {
    background: rgba(34,197,94,0.15);
    border: 2px solid #22C55E;
    color: #4ADE80;
}

.footer {
    text-align:center;
    color:#9CA3AF;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-title">
    🛡 FraudShield
    </div>

    <div class="sidebar-sub">
     Fraud Detection System
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Dashboard")
    # st.markdown("### 🧾 New Prediction")
    # st.markdown("### 📈 Analytics")
    # st.markdown("### ⚙ Settings")
    # st.markdown("### ℹ About")

# ==========================================
# HEADER
# ==========================================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown("""
    <div class="main-title">
    Credit Card <br>
    <span class="gradient-text">
    Fraud Detection
    </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Detect suspicious transactions in real-time
    using Machine Learning models.
    """)

with col2:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2489/2489756.png",
        width=280
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# METRICS
# ==========================================

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="Total Predictions",
        value="2,548",
        delta="+12%"
    )

with m2:
    st.metric(
        label="Fraud Detected",
        value="142",
        delta="+8%"
    )

with m3:
    st.metric(
        label="Model Accuracy",
        value="99%",
        delta="+2%"
    )

# ==========================================
# MAIN LAYOUT
# ==========================================

left, right = st.columns([2,1])

# ==========================================
# INPUT SECTION
# ==========================================

with left:

    st.markdown("""
    <div class="custom-card">
    <h2>📝 Transaction Details</h2>
    </div>
    """, unsafe_allow_html=True)

    amount = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=150.0
    )

    transaction_hour = st.slider(
        "Transaction Hour",
        0,
        23,
        14
    )
    merchant_category = st.selectbox(
    "Merchant Category",
    
      [
    "Electronics",
    "Travel",
    "Food",
    'Clothing',
    'Grocery' 
    ]
    )

    foreign_transaction = st.selectbox(
        "Foreign Transaction",
        [0,1]
    )

    location_mismatch = st.selectbox(
        "Location Mismatch",
        [0,1]
    )

    device_trust_score = st.slider(
        "Device Trust Score",
        0,
        100,
        75
    )

    velocity_last_24h = st.slider(
        "Transactions in Last 24 Hours",
        0,
        20,
        2
    )

    cardholder_age = st.slider(
        "Cardholder Age",
        18,
        100,
        30
    )

    predict_button = st.button("🔍 Predict Transaction")

# ==========================================
# PREDICTION SECTION
# ==========================================

with right:

    st.markdown("""
    <div class="custom-card">
    <h2>📊 Prediction Result</h2>
    </div>
    """, unsafe_allow_html=True)

    if predict_button:

        # ----------------------------------
        # Encode Category
        # ----------------------------------

        merchant_encoded = encoder.transform(
            [merchant_category]
        )[0]

        # ----------------------------------
        # Prepare Input Data
        # ----------------------------------

        input_data = pd.DataFrame([[
            amount,
            transaction_hour,
            merchant_encoded,
            foreign_transaction,
            location_mismatch,
            device_trust_score,
            velocity_last_24h,
            cardholder_age
        ]], columns=[
            'amount',
            'transaction_hour',
            'merchant_category',
            'foreign_transaction',
            'location_mismatch',
            'device_trust_score',
            'velocity_last_24h',
            'cardholder_age'
        ])

        # ----------------------------------
        # Model Prediction
        # ----------------------------------

        prediction = model.predict(input_data)

        probability = model.predict_proba(
            input_data
        )

        fraud_probability = probability[0][1] * 100

        # ----------------------------------
        # Risk Analysis
        # ----------------------------------

        risk_factors = []

        if amount > 10000:
            risk_factors.append(
                "💰 High Transaction Amount"
            )

        if foreign_transaction == 1:
            risk_factors.append(
                "🌍 Foreign Transaction"
            )

        if location_mismatch == 1:
            risk_factors.append(
                "📍 Location Mismatch"
            )

        if velocity_last_24h > 10:
            risk_factors.append(
                "⚡ High Transaction Velocity"
            )

        if device_trust_score < 40:
            risk_factors.append(
                "📱 Low Device Trust Score"
            )

        if transaction_hour < 5:
            risk_factors.append(
                "🕒 Suspicious Transaction Hour"
            )

        # ----------------------------------
        # FRAUD RESULT
        # ----------------------------------

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.progress(
            min(int(fraud_probability), 100)
        )

        if prediction[0] == 1:

            st.markdown("""
            <div class="result-box high-risk">
            ⚠ FRAUDULENT TRANSACTION
            </div>
            """, unsafe_allow_html=True)

            st.error("🔴 HIGH RISK")

        else:

            st.markdown("""
            <div class="result-box low-risk">
            ✅ LEGITIMATE TRANSACTION
            </div>
            """, unsafe_allow_html=True)

            st.success("🟢 LOW RISK")

        # ----------------------------------
        # DYNAMIC RISK FACTORS
        # ----------------------------------

        st.markdown("---")

        st.subheader("🔥 Risk Factors")

        if len(risk_factors) > 0:

            for factor in risk_factors:

                st.warning(factor)

        else:

            st.success(
                "✅ No Major Risk Factors Found"
            )

        # ----------------------------------
        # DYNAMIC RECOMMENDATIONS
        # ----------------------------------

        st.markdown("---")

        st.subheader("🛡 Recommendations")

        if prediction[0] == 1:

            st.error("""
• Verify transaction manually

• Contact cardholder immediately

• Temporarily block suspicious transaction

• Enable multi-factor authentication

• Review recent account activity
""")

        else:

            st.success("""
• Transaction appears safe

• Continue normal monitoring

• Maintain device security

• Enable banking alerts for safety
""")

        # ----------------------------------
        # TRANSACTION SUMMARY
        # ----------------------------------

        st.markdown("---")

        st.subheader("📋 Transaction Summary")

        st.info(f"""
Transaction Amount: ${amount}

Merchant Category: {merchant_category}

Transaction Hour: {transaction_hour}:00

Foreign Transaction: {"Yes" if foreign_transaction == 1 else "No"}

Location Mismatch: {"Yes" if location_mismatch == 1 else "No"}

Device Trust Score: {device_trust_score}

Transactions in Last 24 Hours: {velocity_last_24h}

Cardholder Age: {cardholder_age}
""")