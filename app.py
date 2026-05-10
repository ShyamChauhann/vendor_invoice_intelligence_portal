import streamlit as st
import pandas as pd
import numpy as np
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📦",
    layout="wide"
)

# ----------------------------------------
# Caching Model Calls
# ----------------------------------------
@st.cache_data
def get_freight_prediction(data):
    return predict_freight_cost(data)

@st.cache_data
def get_flag_prediction(data):
    return predict_invoice_flag(data)

# ----------------------------------------
# Header
# ----------------------------------------
st.title("📦 Vendor Invoice Intelligence Portal")
st.caption("AI-powered decision support for finance operations")

st.markdown("""
This portal helps finance teams to:
- 📊 Predict freight costs
- 🚨 Detect risky invoices
- ⚙ Improve operational efficiency
""")

st.divider()

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("🔍 Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
---
### 💼 Business Impact
- Better forecasting
- Fraud detection
- Faster approvals
""")

# ==================================================
# Freight Cost Prediction
# ==================================================
if selected_model == "Freight Cost Prediction":

    st.subheader("🚛 Freight Cost Prediction")

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input("📦 Quantity", min_value=1, value=1200)

        with col2:
            dollars = st.number_input("💰 Invoice Dollars", min_value=1.0, value=18500.0)

        submit_freight = st.form_submit_button("🔮 Predict Freight Cost")

    if submit_freight:

        # Validation
        if dollars < quantity:
            st.warning("⚠ Invoice dollars seem unusually low compared to quantity.")

        input_data = {
            "Quantity": [quantity],
            "Dollars": [dollars]
        }

        prediction = get_freight_prediction(input_data)['Predicted_Freight']
        freight_cost = prediction[0]

        freight_ratio = freight_cost / dollars

        st.success("Prediction completed successfully.")

        # Metrics Display
        col1, col2 = st.columns(2)

        col1.metric("📊 Estimated Freight Cost", f"${freight_cost:,.2f}")
        col2.metric("📉 Freight % of Invoice", f"{freight_ratio:.2%}")

        # Interpretation
        st.info(f"""
        **Insight:**
        - Freight cost is approximately **{freight_ratio:.2%}** of the invoice value.
        - Higher quantity and invoice value generally increase freight.
        """)

        # Progress Bar (visual cue)
        st.progress(min(int(freight_ratio * 100), 100))

        # Download Results
        result_df = pd.DataFrame({
            "Quantity": [quantity],
            "Dollars": [dollars],
            "Predicted Freight": [freight_cost],
            "Freight %": [freight_ratio]
        })

        st.download_button(
            "⬇ Download Result",
            result_df.to_csv(index=False),
            file_name="freight_prediction.csv"
        )

# ==================================================
# Invoice Risk Prediction
# ==================================================
else:

    st.subheader("🚨 Invoice Manual Approval Prediction")

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)

        with col2:
            invoice_dollars = st.number_input("Invoice Dollars", min_value=1.0, value=352.95)
            total_item_quantity = st.number_input("Total Item Quantity", min_value=1, value=162)

        with col3:
            total_item_dollars = st.number_input("Total Item Dollars", min_value=1.0, value=2476.0)

        submit_flag = st.form_submit_button("🧠 Evaluate Invoice Risk")

    if submit_flag:

        # Derived metric
        cost_per_unit = invoice_dollars / invoice_quantity

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        flag_prediction = get_flag_prediction(input_data)['Predicted_Flag']
        is_flagged = bool(flag_prediction[0])

        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("💲 Cost per Unit", f"${cost_per_unit:.2f}")
        col2.metric("🚚 Freight", f"${freight:.2f}")

        # Result
        if is_flagged:
            st.error("🚨 Invoice requires **MANUAL APPROVAL**")
            risk_level = "High Risk 🔴"

            st.markdown("""
            **Why flagged? Possible reasons:**
            - Unusual freight-to-cost ratio
            - High invoice value for low quantity
            - Mismatch in totals
            """)

        else:
            st.success("✅ Invoice is **SAFE for Auto-Approval**")
            risk_level = "Low Risk 🟢"

        st.metric("Risk Level", risk_level)

        # Download Results
        result_df = pd.DataFrame({
            "Invoice Quantity": [invoice_quantity],
            "Invoice Dollars": [invoice_dollars],
            "Freight": [freight],
            "Total Item Quantity": [total_item_quantity],
            "Total Item Dollars": [total_item_dollars],
            "Cost per Unit": [cost_per_unit],
            "Flagged": [is_flagged]
        })

        st.download_button(
            "⬇ Download Result",
            result_df.to_csv(index=False),
            file_name="invoice_risk_result.csv"
        )