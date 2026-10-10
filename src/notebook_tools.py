"""Compatibility helpers that always use the saved Notebook 05 model."""
import json
from pathlib import Path

from src.data_loader import read_raw_shipments
from src.model import dataset_fingerprint, evaluate_model, load_model

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data/processed/aura_shipments.csv"
MODEL_PATH = ROOT / "models/aura_final_model.pkl"


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def notebook_data(data_path="", mapping_path=""):
    path = _project_path(data_path) if data_path else PROCESSED_PATH

    mapping = (
        json.loads(
            _project_path(mapping_path).read_text(encoding="utf-8")
        )
        if mapping_path
        else None
    )

    raw = read_raw_shipments(path, column_mapping=mapping)
    source = "; ".join(
        raw["Data_Source"].dropna().astype(str).unique()
    )

    return raw, source


def data_fingerprint(clean):
    return dataset_fingerprint(clean)


def notebook_train(clean, source):
    raise ValueError(
        "Train and export in Notebook 05. "
        "Backend helpers never replace aura_final_model.pkl."
    )


def notebook_evaluate(clean, source=""):
    model = load_model(MODEL_PATH)
    model.metrics = evaluate_model(clean, model)
    return model