"""Shared notebook loading, training and artifact reuse."""
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
from src.schema import (
    FEATURE_SCHEMA_VERSION,
    TARGET_COLUMN,
)


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_PATH = (
    ROOT / "data/processed/aura_shipments.csv"
)


def _project_path(value):
    path = Path(value).expanduser()

    return (
        path
        if path.is_absolute()
        else ROOT / path
    )


def notebook_data(
    data_path="",
    mapping_path="",
):
    path = (
        _project_path(data_path)
        if data_path
        else PROCESSED_PATH
    )

    mapping = (
        json.loads(
            _project_path(mapping_path).read_text(
                encoding="utf-8"
            )
        )
        if mapping_path
        else None
    )

    raw = read_raw_shipments(
        path,
        column_mapping=mapping,
    )

    source = (
        "; ".join(
            raw["Data_Source"]
            .dropna()
            .astype(str)
            .unique()
        )
        if "Data_Source" in raw
        else ""
    )

    return (
        raw,
        source
        or (
            f"User CSV: {path.name}; "
            "label provenance unverified"
        ),
    )


def data_fingerprint(clean):
    return sha256(
        clean.to_csv(
            index=True
        ).encode("utf-8")
    ).hexdigest()


def notebook_train(clean, source):
    bundle = train_model(
        clean,
        target=TARGET_COLUMN,
        data_source=source,
    )

    bundle.metadata.update(
        notebook_data_sha256=data_fingerprint(
            clean
        ),
        feature_schema_version=(
            FEATURE_SCHEMA_VERSION
        ),
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
    fingerprint = data_fingerprint(
        clean
    )

    for filename in (
        "aura_demo_model.pkl",
        "aura_model.pkl",
    ):
        path = ROOT / "models" / filename

        if not path.is_file():
            continue

        try:
            bundle = load_model(path)
        except ValueError:
            continue

        if (
            bundle.metadata.get(
                "feature_schema_version"
            ) == FEATURE_SCHEMA_VERSION
            and bundle.metadata.get(
                "notebook_data_sha256"
            ) == fingerprint
        ):
            return bundle

    bundle = train_model(
        clean,
        target=TARGET_COLUMN,
        data_source=source,
    )

    bundle.metadata[
        "notebook_data_sha256"
    ] = fingerprint

    return bundle