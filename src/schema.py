#defines share vocabulary
FEATURE_SCHEMA_VERSION = "aura-eta-v1"
DATASET_VERSION = "aura-simulation-v1"
TARGET_COLUMN = "Shipment_Failure"

RAW_INPUT_COLUMNS = [
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

SHIPMENT_COLUMNS = RAW_INPUT_COLUMNS + [
    "Planned_Transit_Hours",
    "Handover_Count",
]

OUTCOME_COLUMNS = [
    "Actual_Transit_Hours",
    "Late_Delivery",
    "Temperature_Excursion",
    "Handling_Damage",
    TARGET_COLUMN,
]

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
    "Planned_Transit_Hours",
    "Handover_Count",
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

URGENCY_WEIGHTS = dict(
    zip(URGENCY_TIERS, [1.0, 1.5, 2.5])
)

MODEL_NUMERIC_COLUMNS = [
    column
    for column in NUMERIC_COLUMNS
    if column != "Cargo_Value_USD"
] + [
    "known_delay_hours",
    "deadline_margin_hours",
    "thermal_margin_hours",
    "eta_to_planned_ratio",
    "eta_to_deadline_ratio",
    "buffer_to_eta_ratio",
    "multimodal_flag",
]

MODEL_CATEGORICAL_COLUMNS = [
    "Transport_Mode",
    "Storage_standard",
]

MODEL_FEATURE_COLUMNS = (
    MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS
)