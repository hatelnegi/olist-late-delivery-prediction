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
    if
