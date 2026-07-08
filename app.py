from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "Random_Forest.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
COLUMNS_PATH = ARTIFACT_DIR / "columns.pkl"

NUMERIC_COLUMNS = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns


def build_feature_row(values, columns, scaler):
    row = {column: 0 for column in columns}

    row["Tenure"] = values["tenure"]
    row["Is_mobile_phone"] = 1 if values["login_device"] == "Mobile Phone" else 0
    row["WarehouseToHome"] = values["warehouse_to_home"]
    row["Is_male"] = 1 if values["gender"] == "Male" else 0
    row["HourSpendOnApp"] = values["hours_on_app"]
    row["NumberOfDeviceRegistered"] = values["devices_registered"]
    row["SatisfactionScore"] = values["satisfaction_score"]
    row["NumberOfAddress"] = values["number_of_addresses"]
    row["Complain"] = 1 if values["complain"] == "Yes" else 0
    row["OrderAmountHikeFromlastYear"] = values["order_hike"]
    row["CouponUsed"] = values["coupons_used"]
    row["OrderCount"] = values["order_count"]
    row["DaySinceLastOrder"] = values["days_since_last_order"]
    row["CashbackAmount"] = values["cashback_amount"]

    categorical_columns = {
        f"Payment_mode_{values['payment_mode']}": 1,
        f"Category_{values['category']}": 1,
        f"MaritalStatus_{values['marital_status']}": 1,
        f"CityTier_{values['city_tier']}": 1,
    }

    for column, value in categorical_columns.items():
        if column in row:
            row[column] = value

    feature_df = pd.DataFrame([row], columns=columns)
    feature_df[NUMERIC_COLUMNS] = scaler.transform(feature_df[NUMERIC_COLUMNS])

    return feature_df


st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    layout="wide"
)

model, scaler, columns = load_artifacts()

st.title("E-Commerce Customer Churn Predictor")
st.write("Enter customer details below to predict whether the customer is likely to churn.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure", min_value=0, max_value=100, value=10)
        warehouse_to_home = st.number_input(
            "Warehouse to Home Distance",
            min_value=0,
            max_value=200,
            value=15,
        )
        hours_on_app = st.number_input(
            "Hours Spent on App",
            min_value=0,
            max_value=24,
            value=3,
        )
        devices_registered = st.number_input(
            "Number of Devices Registered",
            min_value=1,
            max_value=20,
            value=3,
        )
        number_of_addresses = st.number_input(
            "Number of Addresses",
            min_value=1,
            max_value=30,
            value=3,
        )

    with col2:
        satisfaction_score = st.select_slider(
            "Satisfaction Score",
            options=[1, 2, 3, 4, 5],
            value=3,
        )
        complain = st.radio(
            "Customer Complained?",
            ["No", "Yes"],
            horizontal=True,
        )
        order_hike = st.number_input(
            "Order Amount Hike From Last Year (%)",
            min_value=0,
            max_value=100,
            value=15,
        )
        coupons_used = st.number_input(
            "Coupons Used",
            min_value=0,
            max_value=100,
            value=1,
        )
        order_count = st.number_input(
            "Order Count",
            min_value=0,
            max_value=100,
            value=2,
        )
        days_since_last_order = st.number_input(
            "Days Since Last Order",
            min_value=0,
            max_value=365,
            value=5,
        )
        cashback_amount = st.number_input(
            "Cashback Amount",
            min_value=0.0,
            max_value=1000.0,
            value=150.0,
        )

    with col3:
        login_device = st.selectbox(
            "Preferred Login Device",
            ["Mobile Phone", "Computer"],
        )
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )
        payment_mode = st.selectbox(
            "Preferred Payment Mode",
            ["COD", "Credit Card", "Debit Card", "E wallet", "UPI"],
        )
        category = st.selectbox(
            "Preferred Order Category",
            ["Fashion", "Grocery", "Laptop & Accessory", "Mobile Phone", "Others"],
        )
        marital_status = st.selectbox(
            "Marital Status",
            ["Divorced", "Married", "Single"],
        )
        city_tier = st.selectbox(
            "City Tier",
            [1, 2, 3],
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    values = {
        "tenure": tenure,
        "warehouse_to_home": warehouse_to_home,
        "hours_on_app": hours_on_app,
        "devices_registered": devices_registered,
        "number_of_addresses": number_of_addresses,
        "satisfaction_score": satisfaction_score,
        "complain": complain,
        "order_hike": order_hike,
        "coupons_used": coupons_used,
        "order_count": order_count,
        "days_since_last_order": days_since_last_order,
        "cashback_amount": cashback_amount,
        "login_device": login_device,
        "gender": gender,
        "payment_mode": payment_mode,
        "category": category,
        "marital_status": marital_status,
        "city_tier": city_tier,
    }

    features = build_feature_row(values, columns, scaler)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is not likely to churn.")

    st.metric("Churn Probability", f"{probability:.2%}")

    with st.expander("View Model Input Data"):
        st.dataframe(features)
