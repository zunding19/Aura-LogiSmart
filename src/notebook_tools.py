from hashlib import sha256
import json
from pathlib import Path

from src.data_loader import (
    generate_demo_shipments,
    read_raw_shipments,
)
from src.model import (
    load_model,
    save_model,
    train_model,
)
from src.schema import TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[1]


def notebook_data(data_path="", mapping_path=""):
    if not data_path:
        return (
            generate_demo_shipments(n=400, seed=42),
            "synthetic demo (seed=42)",
        )

    path = Path(data_path).expanduser()
    path = path if path.is_absolute() else ROOT / path
    mapping = None

    if mapping_path:
        mapping_file = Path(mapping_path).expanduser()
        mapping_file = (
            mapping_file
            if mapping_file.is_absolute()
            else ROOT / mapping_file
        )
        mapping = json.loads(mapping_file.read_text())

    raw = read_raw_shipments(
        path, column_mapping=mapping
    )

    marked = (
        raw["Shipment_ID"]
        .astype(str)
        .str.upper()
        .str.startswith("SYNTHETIC-")
        .any()
        if "Shipment_ID" in raw
        else False
    )

    if "Data_Source" in raw:
        marked = marked or (
            raw["Data_Source"]
            .astype(str)
            .str.contains(
                "synthetic", case=False, na=False
            )
            .any()
        )

    source = (
        f"synthetic-marked CSV: {path.name}"
        if marked
        else f"user CSV: {path.name}"
    )
    return raw, source


def data_fingerprint(clean):
    return sha256(
        clean.to_csv(index=True).encode()
    ).hexdigest()


def notebook_train(clean, source):
    bundle = train_model(
        clean,
        target=TARGET_COLUMN,
        data_source=source,
    )
    bundle.metadata["notebook_data_sha256"] = (
        data_fingerprint(clean)
    )

    filename = (
        "aura_demo_model.pkl"
        if bundle.metadata["is_synthetic"]
        else "aura_model.pkl"
    )
    path = ROOT / "models" / filename
    save_model(bundle, path)

    return bundle, path


def notebook_evaluate(clean, source):
    fingerprint = data_fingerprint(clean)

    for filename in (
        "aura_demo_model.pkl", "aura_model.pkl"
    ):
        path = ROOT / "models" / filename

        if path.exists():
            bundle = load_model(path)
            if (
                bundle.metadata.get("notebook_data_sha256")
                == fingerprint
            ):
                return bundle

    return train_model(
        clean,
        target=TARGET_COLUMN,
        data_source=source,
    )