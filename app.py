import streamlit as st
import pandas as pd
import joblib

# Configure the page
st.set_page_config(
    page_title="Olist Late Delivery Risk",
    page_icon="📦",
    layout="centered"
)

# Load the saved model bundle
@st.cache_resource
def load_model_bundle():
    return joblib.load("olist_late_delivery_model_final.joblib")

bundle = load_model_bundle()

model = bundle["model"]
threshold = bundle["threshold"]

# Application title and description
st.title("Olist Late Delivery Risk Prediction")

st.write(
    "Enter the order features below to estimate the model's "
    "predicted late-delivery probability."
)

st.warning(
    "Demonstration only: this model was trained on delivered orders "
    "and has not been validated for operational predictions on "
    "orders currently in transit."
)

# Collect numeric features
st.subheader("Order Information")

purchase_month = st.selectbox(
    "Purchase month",
    list(range(1, 13)),
    index=0
)

purchase_day_of_week = st.selectbox(
    "Purchase day of week (0 = Monday)",
    list(range(7)),
    index=0
)

purchase_hour = st.slider(
    "Purchase hour",
    min_value=0,
    max_value=23,
    value=12
)

promised_delivery_days = st.number_input(
    "Promised delivery days",
    min_value=0.0,
    value=15.0,
    step=1.0
)

item_count = st.number_input(
    "Number of items",
    min_value=1,
    value=1,
    step=1
)

total_item_price = st.number_input(
    "Total item price",
    min_value=0.0,
    value=100.0,
    step=10.0
)

total_freight_value = st.number_input(
    "Total freight value",
    min_value=0.0,
    value=20.0,
    step=5.0
)

unique_sellers = st.number_input(
    "Number of unique sellers",
    min_value=1,
    value=1,
    step=1
)

unique_products = st.number_input(
    "Number of unique products",
    min_value=1,
    value=1,
    step=1
)

# Collect categorical features
st.subheader("Product and Seller Information")

main_product_category = st.text_input(
    "Main product category",
    value="bed_bath_table",
    help="Enter a category value represented in the training data."
)

main_seller_state = st.text_input(
    "Main seller state",
    value="SP",
    help="Enter a seller-state code represented in the training data."
)

# Make a prediction when the user clicks the button
if st.button("Predict Delivery Risk"):

    # Create a single-row DataFrame using the expected feature names
    input_data = pd.DataFrame([{
        "purchase_month": purchase_month,
        "purchase_day_of_week": purchase_day_of_week,
        "purchase_hour": purchase_hour,
        "promised_delivery_days": promised_delivery_days,
        "item_count": item_count,
        "total_item_price": total_item_price,
        "total_freight_value": total_freight_value,
        "unique_sellers": unique_sellers,
        "unique_products": unique_products,
        "main_product_category": main_product_category,
        "main_seller_state": main_seller_state
    }])

    # Ensure the feature order matches the saved schema
    input_data = input_data[bundle["feature_columns"]]

    # Calculate the probability of late delivery
    late_probability = model.predict_proba(input_data)[0, 1]

    # Apply the threshold selected during validation
    predicted_late = late_probability >= threshold

    st.subheader("Prediction Result")

    st.metric(
        "Predicted late-delivery probability",
        f"{late_probability:.1%}"
    )

    st.caption(f"Classification threshold: {threshold:.2f}")

    if predicted_late:
        st.error("The model classifies this order as late-delivery risk.")
    else:
        st.success("The model classifies this order as not flagged for late-delivery risk.")

    st.progress(float(late_probability))

    st.caption(
        "This is a model estimate, not a guarantee that an order "
        "will arrive late or on time."
    )
