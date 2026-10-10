<<<<<<< HEAD
#display result 
=======
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Aura Predictive", layout="wide")

# Starter shell. The triage queue, deep dive and decision matrix replace this.
SAMPLE_SHIPMENT = {
    "Shipment_ID": "BIO-8841-KL",
    "Place_of_Origin": "Frankfurt CargoCity",
    "Place_of_Destination": "Kuala Lumpur General Hospital",
    "Distance": 9850,
    "Transport_Mode": "Multi-Modal Air-to-Land",
    "Lead-Time in days": 3,
    "Transit_Duration": 17.5,
    "Stock_Volume": 1200,
    "Geopolitical_Risk": 6.5,
    "Fuel_Price_Index": 1.15,
    "Carrier_Reliability_Score": 0.42,
    "Buffer_of_Package": 2.8,
    "Storage_standard": "2°C to 8°C",
    "Cargo_Value_USD": 85000,
    "Hospital_Urgency_Tier": "Critical Emergency",
}

st.title("Aura")
st.caption("Cold-chain shipment failure risk, caught before the breach.")

st.subheader("Sample shipment")
st.dataframe(
    pd.DataFrame([SAMPLE_SHIPMENT]).astype(str).T.rename(columns={0: "Value"}),
    width="stretch",
)
>>>>>>> zhetein
