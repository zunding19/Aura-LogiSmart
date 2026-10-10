"""Recorded disruption inputs; optional simulated context never enters the model."""
FEATURE_SCHEMA_VERSION = "aura-disruption-v2"
DATASET_VERSION = "aura-context-v2"
TARGET_COLUMN = "Disruption_Occurred"

RAW_INPUT_COLUMNS = [
    "Shipment_ID", "Date", "Origin_Port", "Destination_Port", "Transport_Mode",
    "Product_Category", "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Weather_Condition", "Carrier_Reliability_Score",
    "Lead_Time_Days",
]
SHIPMENT_COLUMNS = list(RAW_INPUT_COLUMNS)
OUTCOME_COLUMNS = [TARGET_COLUMN]

NUMERIC_COLUMNS = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index", "Geopolitical_Risk_Score",
    "Carrier_Reliability_Score", "Lead_Time_Days",
]

# The user confirms Lead_Time_Days is a planned estimate known before disruption.
MODEL_NUMERIC_COLUMNS = NUMERIC_COLUMNS + ["month_sin", "month_cos"]
MODEL_CATEGORICAL_COLUMNS = [
    "Origin_Port", "Destination_Port", "Transport_Mode",
    "Product_Category", "Weather_Condition",
]
MODEL_FEATURE_COLUMNS = MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS

TRANSPORT_MODES = ["Air", "Sea", "Road", "Rail"]
PRODUCT_CATEGORIES = [
    "Electronics", "Textiles", "Perishables", "Pharmaceuticals", "Automotive",
]
WEATHER_CONDITIONS = ["Clear", "Rain", "Fog", "Storm", "Hurricane"]

FLAG_COLUMNS = [
    "Flag_Land_Route_Across_Ocean",
    "Flag_Lead_Time_At_Minimum",
    "Flag_Speed_Impossible",
]
CONTEXT_COLUMNS = [
    "Elapsed_Transit_Hours_Simulated",
    "Storage_Standard_Simulated",
    "Remaining_Buffer_Hours_Simulated",
    "Cargo_Value_USD_Simulated",
    "Hospital_Urgency_Tier_Simulated",
]

# These constants support optional context displays, not recorded model features.
STORAGE_STANDARDS = ["Ambient", "2°C to 8°C", "-20°C", "Deep Frozen / Dry Ice"]
URGENCY_TIERS = [
    "Not Applicable", "Routine Restock", "Elevated Procedure", "Critical Emergency",
]
URGENCY_WEIGHTS = dict(zip(URGENCY_TIERS, [1.0, 1.0, 1.5, 2.5]))

DATA_SOURCE = "Supplied supply-chain dataset; source authenticity unverified"
LABEL_PROVENANCE = (
    "Disruption_Occurred supplied in source file; authenticity unverified"
)
CONTEXT_SOURCE = "Simulated operational context; assumptions only"