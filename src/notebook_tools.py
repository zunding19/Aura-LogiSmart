"""Backend helpers for the notebook teammate's disruption-v2 migration."""
import json
from pathlib import Path

from src.data_loader import read_raw_shipments
from src.model import (
    dataset_fingerprint,
    load_model,
    save_model,
    train_model,
)
from src.schema import TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data/processed/aura_shipments.csv"
MODEL_PATH = ROOT / "models/aura_disruption_model.pkl"


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
    source = (
        "; ".join(raw["Data_Source"].dropna().astype(str).unique())
        if "Data_Source" in raw
        else "Supplied supply-chain records; authenticity unverified"
    )
    return raw, source


def data_fingerprint(clean):
    return dataset_fingerprint(clean)


def notebook_train(clean, source):
    bundle = train_model(
        clean,
        target=TARGET_COLUMN,
        data_source=source,
    )
    save_model(bundle, MODEL_PATH)
    return bundle, MODEL_PATH


def notebook_evaluate(clean, source):
    fingerprint = dataset_fingerprint(clean)

    if MODEL_PATH.is_file():
        try:
            bundle = load_model(MODEL_PATH)
        except ValueError:
            bundle = None

        if (
            bundle is not None
            and bundle.metadata.get("dataset_fingerprint") == fingerprint
        ):
            return bundle

    return train_model(
        clean,
        target=TARGET_COLUMN,
        data_source=source,
    )