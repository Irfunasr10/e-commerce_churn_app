import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# -----------------------------
# Load model, scaler, columns
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("Random_Forest.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, model_columns = load_artifacts()

# Columns that were scaled with StandardScaler in training
SCALE_COLS = [
    'Tenure', 'WarehouseToHome', 'HourSpendOnApp',
    'OrderAmountHikeFromlastYear', 'CouponUsed',
    'OrderCount', 'DaySinceLastOrder',
    'CashbackAmount', 'NumberOfDeviceRegistered',
    'SatisfactionScore', 'NumberOfAddress'
]

st.title("📉 E-Commerce Customer Churn Predictor")
st.write(
    "Fill in the customer details below to predict the likelihood of churn "
    "using a trained Random Forest model."
)

st.divider()

# -----------------------------
# Input form
# -----------------------------
with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        warehouse_to_home = st.number_input("Warehouse To Home (distance)", min_value=0, max_value=200, value=15)
        hour_spend_on_app = st.number_input("Hours Spend On App", min_value=0.0, max_value=10.0, value=3.0)
        number_of_device_registered = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=3)
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
        number_of_address = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)
        complain = st.selectbox("Complain Raised?", ["No", "Yes"])

    with col2:
        order_amount_hike = st.number_input("Order Amount Hike From Last Year (%)", min_value=0, max_value=100, value=15)
        coupon_used = st.number_input("Coupons Used", min_value=0, max_value=50, value=1)
        order_count = st.number_input("Order Count", min_value=0, max_value=50, value=2)
        day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=100, value=5)
        cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=1000.0, value=150.0)

    st.markdown("##### Categorical Details")
    col3, col4, col5 = st.columns(3)

    with col3:
        login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    with col4:
        payment_mode = st.selectbox(
            "Preferred Payment Mode",
            ["Debit Card", "Credit Card", "E wallet", "UPI", "COD"]
        )
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col5:
        category = st.selectbox(
            "Preferred Order Category",
            ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"]
        )
        city_tier = st.selectbox("City Tier", [1, 2, 3])

    submitted = st.form_submit_button("Predict Churn")

# -----------------------------
# Prediction logic
# -----------------------------
if submitted:
    # Build raw input dict matching original (pre-dummy) feature names
    raw_input = {
        'Tenure': tenure,
        'WarehouseToHome': warehouse_to_home,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_device_registered,
        'SatisfactionScore': satisfaction_score,
        'NumberOfAddress': number_of_address,
        'Complain': 1 if complain == "Yes" else 0,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount,
        'Is_mobile_phone': 1 if login_device == "Mobile Phone" else 0,
        'Is_male': 1 if gender == "Male" else 0,
        'Payment_mode': payment_mode,
        'Category': category,
        'MaritalStatus': marital_status,
        'CityTier': city_tier,
    }

    input_df = pd.DataFrame([raw_input])

    # One-hot encode the same categorical columns used in training
    input_df = pd.get_dummies(
        input_df, columns=['Payment_mode', 'Category', 'MaritalStatus', 'CityTier']
    )

    # Convert any bool columns to int
    for c in input_df.columns:
        if input_df[c].dtype == 'bool':
            input_df[c] = input_df[c].astype(int)

    # Align with training columns (add missing dummy cols as 0, drop extras, keep order)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Apply the same scaler to the numeric columns
    input_df[SCALE_COLS] = scaler.transform(input_df[SCALE_COLS])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn**.")
    else:
        st.success(f"✅ This customer is **likely to stay**.")

    st.metric("Churn Probability", f"{probability * 100:.2f}%")
    st.progress(min(int(probability * 100), 100))

    with st.expander("See processed input fed to the model"):
        st.dataframe(input_df)

st.divider()
st.caption("Model: Random Forest (GridSearchCV tuned) | Built with Streamlit")