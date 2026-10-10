#defines share vocabulary
SHIPMENT_COLUMNS = [
    "Shipment_ID",
    "Place_of_Origin",
    "Place_of_Destination",
    "Distance",
    "Transport_Mode",
    "Lead-Time in days",
    "Transit_Duration",
    "Stock_Volume",
    "Geopolitical_Risk",
    "Fuel_Price_Index",
    "Carrier_Reliability_Score",
    "Buffer_of_Package",
    "Storage_standard",
    "Cargo_Value_USD",
    "Hospital_Urgency_Tier",
]

TARGET_COLUMN = "Shipment_Failure"

NUMERIC_COLUMNS = [
    "Distance",
    "Lead-Time in days",
    "Transit_Duration",
    "Stock_Volume",
    "Geopolitical_Risk",
    "Fuel_Price_Index",
    "Carrier_Reliability_Score",
    "Buffer_of_Package",
    "Cargo_Value_USD",
]

TRANSPORT_MODES = [
    "Air",
    "Land",
    "Multi-Modal Air-to-Land",
]

STORAGE_STANDARDS = [
    "2°C to 8°C",
    "-20°C",
    "Deep Frozen / Dry Ice",
]

URGENCY_TIERS = [
    "Routine Restock",
    "Elevated Procedure",
    "Critical Emergency",
]

URGENCY_WEIGHTS = dict(zip(URGENCY_TIERS, [1.0, 1.5, 2.5]))

MODEL_NUMERIC_COLUMNS = [
    "Distance",
    "Lead-Time in days",
    "Transit_Duration",
    "Stock_Volume",
    "Geopolitical_Risk",
    "Fuel_Price_Index",
    "Carrier_Reliability_Score",
    "Buffer_of_Package",
    "remaining_sla_hours",
    "thermal_margin_to_deadline",
    "elapsed_sla_ratio",
    "buffer_to_deadline_ratio",
    "multimodal_flag",
]

MODEL_CATEGORICAL_COLUMNS = [
    "Transport_Mode",
    "Storage_standard",
]

MODEL_FEATURE_COLUMNS = (
    MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS
)