import streamlit as st
import pandas as pd
import joblib
import hmac

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Olist Late Delivery Risk",
    page_icon="📦",
    layout="centered"
)

# ==================================================
# 2. LOGIN SYSTEM
# ==================================================

def login():
    st.title("📦 Olist Analytics")
    st.subheader("Late Delivery Risk Prediction")
    st.write("Please log in to access the prediction dashboard.")

    st.divider()

    with st.form("login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        submitted = st.form_submit_button(
            "Login",
            use_container_width=True
        )

        if submitted:
            try:
                correct_username = st.secrets["auth"]["username"]
                correct_password = st.secrets["auth"]["password"]

                username_ok = hmac.compare_digest(
                    username,
                    correct_username
                )

                password_ok = hmac.compare_digest(
                    password,
                    correct_password
                )

                if username_ok and password_ok:
                    st.session_state["authenticated"] = True
                    st.rerun()

                else:
                    st.error("Incorrect username or password.")

            except Exception:
                st.error(
                    "Login configuration is unavailable. "
                    "Please check the app's Secrets settings."
                )


# ==================================================
# 3. AUTHENTICATION CHECK
# ==================================================

if not st.session_state.get("authenticated", False):
    login()
    st.stop()


# ==================================================
# 4. LOAD THE SAVED PREDICTION MODEL
# ==================================================

@st.cache_resource
def load_model_bundle():
    return joblib.load("olist_late_delivery_model_final.joblib")


try:
    bundle = load_model_bundle()
    model = bundle["model"]
    threshold = bundle["threshold"]

except Exception:
    st.error(
        "The prediction model could not be loaded. "
        "Please check the model file and deployment logs."
    )
    st.stop()


# ==================================================
# 5. DASHBOARD HEADER
# ==================================================

header_col, logout_col = st.columns([4, 1])

with header_col:
    st.title("Olist Late Delivery Risk Prediction")

with logout_col:
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

st.write(
    "Enter the order features below to estimate the model's "
    "predicted late-delivery probability."
)

st.warning(
    "Demonstration only: this model was trained on delivered orders "
    "and has not been validated for operational predictions on "
    "orders currently in transit."
)

st.divider()


# ==================================================
# 6. ORDER INFORMATION
# ==================================================

st.subheader("📋 Order Information")

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


# ==================================================
# 7. PRODUCT AND SELLER INFORMATION
# ==================================================

st.subheader("🛍️ Product and Seller Information")

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


# ==================================================
# 8. MAKE A PREDICTION
# ==================================================

if st.button("Predict Delivery Risk", use_container_width=True):

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

    try:
        # Match the exact feature order expected by the model.
        input_data = input_data[bundle["feature_columns"]]

        # Calculate predicted late-delivery probability.
        late_probability = model.predict_proba(input_data)[0, 1]

        # Apply the saved classification threshold.
        predicted_late = late_probability >= threshold

        st.divider()
        st.subheader("📊 Prediction Result")

        st.metric(
            "Predicted late-delivery probability",
            f"{late_probability:.1%}"
        )

        st.caption(
            f"Classification threshold: {threshold:.2f}"
        )

        if predicted_late:
            st.error(
                "⚠️ The model classifies this order "
                "as late-delivery risk."
            )
        else:
            st.success(
                "✅ The model classifies this order "
                "as not flagged for late-delivery risk."
            )

        st.progress(float(late_probability))

        st.caption(
            "This is a model estimate, not a guarantee that an order "
            "will arrive late or on time."
        )

    except Exception:
        st.error(
            "Prediction could not be completed. Check that the "
            "input values and feature schema match the saved model."
        )
