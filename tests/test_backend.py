"""Disruption-v2 regressions; supplied labels and recorded values stay intact."""
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
from src.clean_dataset import clean
from src.data_loader import (
    generate_demo_shipments,
    load_shipments,
    read_raw_shipments,
)
from src.explain import explain_prediction
from src.features import extract_features
from src.model import (
    dataset_fingerprint,
    load_model,
    predict_failure_risk,
    save_model,
    train_model,
)
from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.prioritize import calculate_triage_queue
from src.schema import (
    CONTEXT_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    SHIPMENT_COLUMNS,
    TARGET_COLUMN,
)
import src.notebook_tools as notebooks
import src.recommendations as actions

RAW_PATH = ROOT / "data/raw/global_supply_chain_risk_2026.csv"


@pytest.fixture(scope="module")
def records():
    return load_shipments(RAW_PATH, require_target=True)


@pytest.fixture(scope="module")
def sample(records):
    return records.sample(
        320, random_state=42
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def bundle(sample):
    return train_model(
        sample,
        data_source="supplied disruption records; authenticity unverified",
    )


def test_cleaning_preserves_recorded_values_and_labels(records):
    result = clean(read_raw_shipments(RAW_PATH))
    columns = SHIPMENT_COLUMNS + [TARGET_COLUMN]
    expected = records.sort_values(
        "Shipment_ID"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result[columns], expected[columns]
    )
    assert len(result) == 5000
    assert result[TARGET_COLUMN].sum() == 3063
    assert result.Needs_Review.sum() == 975
    assert set(result.Transport_Mode) == {"Air", "Sea", "Road", "Rail"}
    assert (result.Weight_MT % 1 != 0).any()
    assert result.iloc[0].Weight_MT == pytest.approx(197.42)
    assert result.Label_Provenance.str.contains(
        "unverified", case=False
    ).all()


def test_simulated_context_is_target_independent(records):
    original = build(records, seed=42)
    flipped = records.copy()
    flipped[TARGET_COLUMN] = 1 - flipped[TARGET_COLUMN]
    changed = build(flipped, seed=42)

    pd.testing.assert_frame_equal(
        original[SHIPMENT_COLUMNS + CONTEXT_COLUMNS],
        changed[SHIPMENT_COLUMNS + CONTEXT_COLUMNS],
    )
    assert original[TARGET_COLUMN].sum() == 3063
    assert changed[TARGET_COLUMN].sum() == 1937
    assert original.Context_Source.str.contains(
        "simulated", case=False
    ).all()

    pd.testing.assert_frame_equal(
        extract_features(original),
        extract_features(records),
    )
    assert not set(
        CONTEXT_COLUMNS + [TARGET_COLUMN, "Shipment_ID", "Needs_Review"]
    ).intersection(MODEL_FEATURE_COLUMNS)
    assert len(MODEL_FEATURE_COLUMNS) == 13
    assert "Lead_Time_Days" in MODEL_FEATURE_COLUMNS


def test_loader_preserves_raw_text_blank_values_and_cursor():
    stream = StringIO(
        "Shipment_ID,Origin_Port,Weight_MT\n0007,NA,\n"
    )
    stream.seek(5)
    raw = read_raw_shipments(stream)

    assert stream.tell() == 5
    assert raw.iloc[0].Shipment_ID == "0007"
    assert raw.iloc[0].Origin_Port == "NA"
    assert pd.isna(raw.iloc[0].Weight_MT)
    assert "unverified" in raw.iloc[0].Data_Source


@pytest.mark.parametrize("csv", [
    "<<<<<<< ours\nShipment_ID\na\n",
    "Shipment_ID\na\n=======\nb\n",
    "Shipment_ID\na\n>>>>>>> theirs\n",
])
def test_loader_rejects_conflict_markers_anywhere(csv):
    with pytest.raises(
        ShipmentValidationError, match="merge marker"
    ):
        read_raw_shipments(StringIO(csv))


@pytest.mark.parametrize("csv", [
    "Shipment_ID,Shipment_ID\na,b\n",
    "Origin_Port,Place_of_Origin\na,b\n",
    "Shipment_ID,Date\na\n",
])
def test_loader_rejects_duplicate_aliased_or_ragged_headers(csv):
    with pytest.raises(ShipmentValidationError):
        read_raw_shipments(StringIO(csv))


@pytest.mark.parametrize("field,value", [
    ("Weight_MT", True),
    ("Weight_MT", np.inf),
    ("Weight_MT", -1),
    ("Fuel_Price_Index", np.nan),
    ("Date", "2025-02-30"),
    ("Date", "16/10/2025"),
    ("Date", 123),
    ("Transport_Mode", "Land"),
    (TARGET_COLUMN, 2),
    (TARGET_COLUMN, True),
    (TARGET_COLUMN, np.nan),
])
def test_validation_rejects_invalid_recorded_inputs(
    records, field, value
):
    row = records.iloc[[0]].copy()
    row[field] = [value]

    with pytest.raises(
        ShipmentValidationError, match=field
    ):
        clean_raw_shipment_data(row, require_target=True)


def test_missing_labels_are_not_generated(records):
    inputs = records.drop(columns=TARGET_COLUMN)
    assert TARGET_COLUMN not in clean_raw_shipment_data(inputs)

    with pytest.raises(
        ShipmentValidationError, match=TARGET_COLUMN
    ):
        train_model(inputs)

    with pytest.raises(ValueError, match=TARGET_COLUMN):
        train_model(records, target="Shipment_Failure")


def test_sampling_preserves_supplied_labels_and_is_repeatable(
    records, monkeypatch
):
    import src.data_loader as loader

    monkeypatch.setattr(loader, "DEMO_PATH", RAW_PATH)
    first = generate_demo_shipments(320, 42)
    second = generate_demo_shipments(320, 42)

    pd.testing.assert_frame_equal(first, second)
    assert first.Shipment_ID.is_unique

    expected = records.set_index("Shipment_ID")[TARGET_COLUMN]
    assert first[TARGET_COLUMN].tolist() == expected.loc[
        first.Shipment_ID
    ].tolist()

    with pytest.raises(ValueError, match="without replacement"):
        generate_demo_shipments(5001)


def test_holdout_and_cv_keep_complete_dates_in_order(sample, bundle):
    train = bundle.metadata["train_positions"]
    test = bundle.metadata["test_positions"]

    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(sample)
    assert sample.iloc[train].Date.max() < sample.iloc[test].Date.min()
    assert set(sample.iloc[train].Date).isdisjoint(
        sample.iloc[test].Date
    )

    for fold in bundle.metadata["cv_splits"]:
        past = fold["train_positions"]
        future = fold["validation_positions"]

        assert set(past + future).issubset(train)
        assert set(past).isdisjoint(future)
        assert sample.iloc[past].Date.max() < sample.iloc[future].Date.min()
        assert set(sample.iloc[past].Date).isdisjoint(
            sample.iloc[future].Date
        )

    features = extract_features(sample).iloc[train][
        MODEL_NUMERIC_COLUMNS
    ]
    scaler = (
        bundle.pipeline.named_steps["preprocessor"]
        .named_transformers_["numeric"]
        .named_steps["scaler"]
    )

    np.testing.assert_allclose(
        scaler.mean_, features.mean(), atol=1e-12
    )

    assert (
        bundle.metadata["feature_schema_version"]
        == FEATURE_SCHEMA_VERSION
    )
    assert bundle.metadata["artifact_format_version"] == 3
    assert bundle.metadata["planned_lead_time_confirmed"]
    assert bundle.metadata["label_provenance"] == "unverified"
    assert not bundle.metadata["calibrated"]
    assert bundle.metadata["is_synthetic"] is None


def test_duplicate_ids_and_insufficient_dates_do_not_fall_back(sample):
    repeated = pd.concat(
        [sample, sample.iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Repeated Shipment_ID"):
        train_model(repeated)

    collapsed = sample.copy()
    collapsed["Date"] = pd.Timestamp("2025-01-01")
    with pytest.raises(ValueError, match="distinct record dates"):
        train_model(collapsed)


def test_prediction_alias_and_artifact_roundtrip(
    sample, bundle, tmp_path
):
    path = tmp_path / "trusted_disruption.pkl"
    save_model(bundle, path)

    features = extract_features(sample.iloc[:10])
    predicted = predict_failure_risk(
        features, load_model(path)
    )

    pd.testing.assert_frame_equal(
        predicted,
        predict_failure_risk(features, bundle),
    )
    np.testing.assert_array_equal(
        predicted.p_disruption, predicted.p_failure
    )
    assert predicted.p_disruption.between(0, 1).all()
    assert predicted.target_column.eq(TARGET_COLUMN).all()
    assert predicted.index.equals(features.index)


@pytest.mark.parametrize("key,value", [
    ("feature_schema_version", "aura-eta-v1"),
    ("artifact_format_version", 2),
    ("target_column", "Shipment_Failure"),
    ("sklearn_version", "0.0"),
])
def test_stale_artifact_contracts_are_rejected(
    bundle, tmp_path, key, value
):
    path = tmp_path / "stale.pkl"
    joblib.dump(
        replace(
            bundle,
            metadata={**bundle.metadata, key: value},
        ),
        path,
    )

    with pytest.raises(ValueError, match="retrain"):
        load_model(path)


def test_old_dictionary_and_invalid_pickle_are_rejected(tmp_path):
    path = tmp_path / "obsolete.pkl"
    joblib.dump(
        {"model": "old cold-chain notebook format"},
        path,
    )
    with pytest.raises(ValueError, match="retrain"):
        load_model(path)

    path.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="retrain"):
        load_model(path)


def test_triage_is_stable_probability_order_without_financial_ranking(
    records,
):
    rows = records.iloc[:3].copy()
    rows["p_disruption"] = [.4, .9, .9]
    rows["Cargo_Value_USD_Simulated"] = [1e12, 0, 1e9]

    queue = calculate_triage_queue(rows)

    assert queue.index.tolist() == [1, 2, 0]
    assert queue.risk_score.tolist() == [.9, .9, .4]
    assert "expected_loss_usd" not in queue
    assert queue.p_failure.equals(queue.p_disruption)

    rows["p_failure"] = [.5, .9, .9]
    with pytest.raises(
        ShipmentValidationError, match="disagrees"
    ):
        calculate_triage_queue(rows)


def test_scenarios_keep_adverse_probability_changes_and_unknown_costs(
    records, bundle, monkeypatch
):
    row = records.iloc[0].to_dict()
    row["Carrier_Reliability_Score"] = .3

    monkeypatch.setattr(
        actions,
        "predict_failure_risk",
        lambda features, model: pd.DataFrame(
            {
                "p_disruption": features.Carrier_Reliability_Score
            },
            index=features.index,
        ),
    )

    result = actions.simulate_intervention(
        row, "carrier_review", bundle
    )

    assert result["before_probability"] == .3
    assert result["after_probability"] == .95
    assert result["probability_change"] == pytest.approx(-.65)
    assert result["estimated_net_benefit_usd"] is None
    assert result["estimated_loss_reduction_usd"] is None
    assert result["action_cost_usd"] is None
    assert set(result["changed_fields"]) == {
        "Carrier_Reliability_Score"
    }

    monitor = actions.simulate_intervention(
        row, "monitor", bundle
    )
    assert monitor["probability_change"] == 0
    assert monitor["changed_fields"] == {}

    sla = actions.check_sla_status(.99, buffer_hours=.1)
    assert not sla["available"]
    assert not sla["breach_warning"]
    assert not sla["simulated_escrow_locked"]


def test_explanations_have_three_finite_signed_references(
    records, bundle
):
    row = records.iloc[0].to_dict()
    row.update(
        Weather_Condition="Storm",
        Carrier_Reliability_Score=.4,
        Geopolitical_Risk_Score=8,
    )

    result = explain_prediction(row, bundle)

    assert result["target"] == TARGET_COLUMN
    assert result["category"] == "Combined exposure"
    assert len(result["drivers"]) == 3
    assert "model sensitivity" in result["method"]

    contributions = [
        abs(driver["contribution"])
        for driver in result["drivers"]
    ]
    assert contributions == sorted(
        contributions, reverse=True
    )

    for driver in result["drivers"]:
        assert np.isfinite(driver["contribution"])
        assert driver["contribution"] == pytest.approx(
            result["baseline_probability"]
            - driver["counterfactual_probability"]
        )


def test_fingerprint_ignores_simulated_context_and_helper_skips_old_model(
    sample, bundle, tmp_path, monkeypatch
):
    contextual = sample.assign(
        Cargo_Value_USD_Simulated=1e12,
        Remaining_Buffer_Hours_Simulated=0,
    )
    assert (
        dataset_fingerprint(contextual)
        == dataset_fingerprint(sample)
    )

    path = tmp_path / "old.pkl"
    joblib.dump({"model": "obsolete"}, path)

    monkeypatch.setattr(notebooks, "MODEL_PATH", path)
    monkeypatch.setattr(
        notebooks,
        "train_model",
        lambda *args, **kwargs: bundle,
    )

    assert notebooks.notebook_evaluate(
        sample, "supplied records"
    ) is bundle