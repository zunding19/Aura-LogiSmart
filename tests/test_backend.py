"""Backend data, model and scenario regression checks."""
from dataclasses import replace
from io import StringIO
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_dataset import build
from src.data_loader import (
    generate_demo_shipments,
    load_shipments,
    read_raw_shipments,
)
from src.explain import explain_prediction
from src.features import extract_features
from src.model import (
    load_model,
    predict_failure_risk,
    risk_level,
    save_model,
    train_model,
)
from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.prioritize import calculate_triage_queue
from src.protocols import (
    export_custody_audit,
    record_custody_handshake,
)
from src.schema import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    OUTCOME_COLUMNS,
)

import src.notebook_tools as notebooks
import src.recommendations as actions


@pytest.fixture(scope="module")
def full():
    return load_shipments(
        ROOT / "data/processed/aura_shipments.csv",
        require_target=True,
    )


@pytest.fixture(scope="module")
def balanced(full):
    return pd.concat(
        [
            full[
                full.Shipment_Failure.eq(label)
            ].sample(
                80,
                random_state=42,
            )
            for label in (0, 1)
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def bundle(balanced):
    return train_model(
        balanced,
        data_source="synthetic processed test sample",
    )


def test_builder_preserves_original_dataset():
    original = pd.read_csv(
        ROOT / "data/processed/aura_shipments.csv"
    )

    built = build(
        pd.read_csv(
            ROOT / "data/raw/global_supply_chain_pharma.csv"
        ),
        seed=42,
    )

    columns = [
        column
        for column in original
        if column not in {
            "Data_Source",
            "Simulation_Seed",
            "Dataset_Version",
        }
    ]

    assert len(built) == 5000
    assert len(columns) == 22
    assert built.Shipment_Failure.sum() == 787

    pd.testing.assert_frame_equal(
        built[columns],
        original[columns],
        check_dtype=False,
        rtol=1e-12,
        atol=1e-10,
    )

    assert built.Data_Source.str.contains(
        "synthetic",
        case=False,
    ).all()

    assert built.Simulation_Seed.eq(
        42
    ).all()

    assert (
        built.Dataset_Version.nunique()
        == 1
    )


def test_csv_text_missingness_and_provenance(tmp_path):
    raw = read_raw_shipments(
        StringIO(
            "Shipment_ID,Place_of_Origin,Buffer_of_Package\n"
            "0007,NA,\n"
        )
    )

    assert raw.iloc[0].Shipment_ID == "0007"
    assert raw.iloc[0].Place_of_Origin == "NA"
    assert pd.isna(
        raw.iloc[0].Buffer_of_Package
    )

    copied = (
        tmp_path / "renamed_shipments.csv"
    )
    copied.write_bytes(
        (
            ROOT / "data/processed/aura_shipments.csv"
        ).read_bytes()
    )

    assert read_raw_shipments(
        copied
    ).Data_Source.str.contains(
        "synthetic|simulat",
        case=False,
    ).all()

    with pytest.raises(
        ShipmentValidationError,
        match="Duplicate",
    ):
        read_raw_shipments(
            StringIO(
                "Shipment_ID,Shipment_ID\n"
                "a,b\n"
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("Handover_Count", 1.5),
        ("Handover_Count", -1),
        ("Handover_Count", np.nan),
        ("Storage_standard", "+20°C"),
    ],
)
def test_invalid_operational_inputs(full, field, value):
    row = full.iloc[[0]].copy()
    row[field] = value

    with pytest.raises(
        ShipmentValidationError,
        match=field,
    ):
        clean_raw_shipment_data(row)


def test_raw_mode_and_missing_outcomes(full):
    raw = full.iloc[[0]].drop(
        columns=[
            "Planned_Transit_Hours",
            "Handover_Count",
            *OUTCOME_COLUMNS,
        ]
    )

    raw["Hospital_Urgency_Tier"] = (
        "Critical Emergency / Direct Patient Treatment"
    )

    cleaned = clean_raw_shipment_data(
        raw,
        require_processed=False,
    )

    assert (
        cleaned.iloc[0].Hospital_Urgency_Tier
        == "Critical Emergency"
    )

    with pytest.raises(
        ShipmentValidationError,
        match="Planned_Transit_Hours",
    ):
        clean_raw_shipment_data(raw)

    with pytest.raises(
        ShipmentValidationError,
        match="Shipment_Failure",
    ):
        train_model(
            full.drop(
                columns=OUTCOME_COLUMNS
            ),
            data_source="synthetic",
        )


def test_eta_features_and_leakage_exclusions(full):
    row = full.loc[
        full.Shipment_ID.eq(
            "SC-10000"
        )
    ]
    features = extract_features(row)

    assert list(
        features
    ) == MODEL_FEATURE_COLUMNS

    assert len(
        features.columns
    ) == 19

    excluded = OUTCOME_COLUMNS + [
        "Shipment_ID",
        "Cargo_Value_USD",
        "Hospital_Urgency_Tier",
    ]

    assert not set(
        excluded
    ).intersection(features)

    assert features.index.equals(
        row.index
    )

    assert (
        features.iloc[0].thermal_margin_hours
        == pytest.approx(28.3)
    )

    assert (
        features.iloc[0].deadline_margin_hours
        == pytest.approx(35.28)
    )

    assert (
        features.iloc[0].known_delay_hours
        == pytest.approx(1.6)
    )


def test_demo_sampling_is_repeatable_and_without_replacement():
    first = generate_demo_shipments(
        160, 42
    )
    second = generate_demo_shipments(
        160, 42
    )

    pd.testing.assert_frame_equal(
        first, second
    )

    assert first.Shipment_ID.is_unique

    assert first.Data_Source.str.contains(
        "synthetic|simulat",
        case=False,
    ).all()

    with pytest.raises(
        ValueError,
        match="without replacement",
    ):
        generate_demo_shipments(5001)


@pytest.mark.parametrize(
    "probability,band",
    [
        (0, "Low"),
        (.3, "Low"),
        (.6, "Medium"),
        (.8, "High"),
        (1, "Critical"),
    ],
)
def test_risk_band_boundaries(probability, band):
    assert risk_level(
        probability
    ) == band


def test_training_only_scaler_and_holdout(balanced, bundle):
    train = bundle.metadata[
        "train_positions"
    ]
    test = bundle.metadata[
        "test_positions"
    ]

    assert set(train).isdisjoint(test)
    assert (
        len(train) + len(test)
        == len(balanced)
    )

    features = extract_features(
        balanced
    )

    scaler = (
        bundle.pipeline
        .named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )

    np.testing.assert_allclose(
        scaler.mean_,
        features.iloc[train][
            MODEL_NUMERIC_COLUMNS
        ].mean(),
    )

    assert (
        bundle.metadata["feature_schema_version"]
        == FEATURE_SCHEMA_VERSION
    )
    assert bundle.metadata["is_synthetic"]

    assert (
        bundle.metrics["n_test"]
        == len(test)
    )
    assert sum(
        map(
            sum,
            bundle.metrics["confusion_matrix"],
        )
    ) == len(test)


def test_repeated_shipments_stay_in_one_partition(balanced):
    groups = pd.concat(
        [
            balanced.iloc[:40],
            balanced.iloc[80:120],
        ],
        ignore_index=True,
    )

    repeated = pd.concat(
        [groups, groups],
        ignore_index=True,
    )

    model = train_model(
        repeated,
        data_source="synthetic grouped sample",
    )

    train = model.metadata[
        "train_positions"
    ]
    test = model.metadata[
        "test_positions"
    ]

    assert set(
        repeated.iloc[train].Shipment_ID
    ).isdisjoint(
        repeated.iloc[test].Shipment_ID
    )

    assert (
        model.metadata["cv_method"]
        == "StratifiedGroupKFold"
    )


def test_model_roundtrip_and_stale_schema_rejection(
    tmp_path,
    balanced,
    bundle,
):
    path = (
        tmp_path / "trusted_model.pkl"
    )
    save_model(bundle, path)

    features = extract_features(
        balanced.iloc[:5]
    )

    pd.testing.assert_frame_equal(
        predict_failure_risk(
            features, bundle
        ),
        predict_failure_risk(
            features, load_model(path)
        ),
    )

    stale = replace(
        bundle,
        metadata={
            **bundle.metadata,
            "feature_schema_version": "aura-old",
        },
    )

    joblib.dump(stale, path)

    with pytest.raises(
        ValueError,
        match="Retrain",
    ):
        load_model(path)


def test_triage_ranks_financial_stakes(full):
    rows = pd.concat(
        [full.iloc[[0]]] * 2,
        ignore_index=True,
    )

    rows["p_failure"] = .5
    rows["Cargo_Value_USD"] = [
        100, 1000
    ]

    queue = calculate_triage_queue(
        rows
    )

    assert queue.index.tolist() == [
        1, 0
    ]
    assert (
        queue.expected_loss_usd.tolist()
        == [500, 50]
    )


def test_monitor_diversion_and_adverse_scenario(
    full,
    bundle,
    monkeypatch,
):
    row = full.iloc[0].to_dict()

    for action, expected in [
        ("monitor", 0),
        ("emergency_stock_diversion", -750),
    ]:
        result = actions.simulate_intervention(
            row,
            action,
            bundle,
        )

        assert (
            result["after_probability"]
            == result["before_probability"]
        )
        assert result["changed_fields"] == {}
        assert (
            result["estimated_net_benefit_usd"]
            == expected
        )

    monkeypatch.setattr(
        actions,
        "predict_failure_risk",
        lambda features, model: pd.DataFrame(
            {
                "p_failure": np.where(
                    features.Buffer_of_Package.gt(
                        row["Buffer_of_Package"]
                    ),
                    .9,
                    .2,
                )
            },
            index=features.index,
        ),
    )

    result = actions.simulate_intervention(
        row,
        "refrigerant_refresh",
        bundle,
    )

    assert (
        result["after_probability"]
        > result["before_probability"]
    )
    assert (
        result["estimated_net_benefit_usd"]
        < 0
    )


def test_explanation_reports_actual_signed_sensitivity(
    full,
    bundle,
):
    result = explain_prediction(
        full.iloc[0],
        model=bundle,
    )

    assert (
        "model sensitivity"
        in result["method"]
    )
    assert (
        1 <= len(result["drivers"]) <= 3
    )

    for driver in result["drivers"]:
        assert np.isfinite(
            driver["contribution"]
        )
        assert (
            driver["contribution"]
            == pytest.approx(
                result["baseline_probability"]
                - driver["counterfactual_probability"]
            )
        )


def test_two_handler_custody_requires_confirmation():
    records = []

    record_custody_handshake(
        "SC-test",
        "Alice",
        "Bob",
        3,
        "Hub",
        True,
        records,
    )

    assert len(records) == 1

    assert (
        '"shipment_id": "SC-test"'
        in export_custody_audit(records)
    )

    for receiver, confirmed in [
        ("ALICE", True),
        ("Bob", False),
    ]:
        with pytest.raises(
            ShipmentValidationError
        ):
            record_custody_handshake(
                "SC-test",
                "Alice",
                receiver,
                3,
                "Hub",
                confirmed,
                records,
            )

    assert len(records) == 1


def test_notebook_full_data_raw_inspection_and_stale_artifact(
    full,
    balanced,
    bundle,
    tmp_path,
    monkeypatch,
):
    raw, source = notebooks.notebook_data()

    assert len(raw) == 5000
    assert any(
        word in source.casefold()
        for word in (
            "synthetic", "simulat"
        )
    )

    raw, source = notebooks.notebook_data(
        "data/raw/global_supply_chain_pharma.csv"
    )

    assert (
        "Planned_Transit_Hours"
        not in raw
    )
    assert len(raw) == 5000

    monkeypatch.setattr(
        notebooks,
        "ROOT",
        tmp_path,
    )

    (tmp_path / "models").mkdir()

    stale = replace(
        bundle,
        metadata={
            **bundle.metadata,
            "feature_schema_version": "aura-old",
        },
    )

    joblib.dump(
        stale,
        tmp_path / "models/aura_demo_model.pkl",
    )

    monkeypatch.setattr(
        notebooks,
        "train_model",
        lambda *args, **kwargs: bundle,
    )

    assert notebooks.notebook_evaluate(
        balanced,
        "synthetic test",
    ) is bundle