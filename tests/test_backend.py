"""Regressions against the committed notebooks and their saved model."""
import ast
import hashlib
from io import StringIO
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.build_dataset import build
from src.clean_dataset import clean
from src.data_loader import load_shipments, read_raw_shipments
from src.features import extract_features
from src.model import (
    evaluate_model,
    load_model,
    predict_failure_risk,
    risk_level,
    train_model,
)
from src.preprocessing import (
    ShipmentValidationError,
    clean_raw_shipment_data,
)
from src.schema import (
    CONTEXT_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    SHIPMENT_COLUMNS,
    TARGET_COLUMN,
)
import src.notebook_tools as helpers

from scripts.train import main as verify_main

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/aura_final_model.pkl"
RAW_PATH = ROOT / "data/raw/global_supply_chain_risk_2026.csv"
PROCESSED_PATH = ROOT / "data/processed/aura_shipments.csv"


def notebook_function(number, name):
    """Read the teammate's actual function, without executing training cells."""
    path = next(
        (ROOT / "notebooks").glob(f"{number:02d}_*.ipynb")
    )
    namespace = {"np": np, "pd": pd}
    functions = []
    constants = {
        "WEATHER_SEVERITY",
        "TYPICAL_PACE",
        "HIGH_RISK_SCORE",
        "LOW_RELIABILITY",
        "SLOWDOWN",
        "INPUT_COLUMNS",
        "RAW_INPUTS",
    }

    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue

        for node in ast.parse("".join(cell["source"])).body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                key = node.targets[0].id

                if key in constants:
                    if key == "INPUT_COLUMNS":
                        namespace[key] = (
                            namespace["RAW_INPUTS"]
                            + ast.literal_eval(node.value.right)
                        )
                    else:
                        namespace[key] = ast.literal_eval(node.value)

            if (
                isinstance(node, ast.FunctionDef)
                and node.name == name
            ):
                functions.append(node)

    assert len(functions) == 1, (
        f"Expected one {name} in Notebook {number}."
    )

    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )

    return namespace[name]


@pytest.fixture(scope="module")
def records():
    return load_shipments(
        RAW_PATH,
        require_target=True,
    )


@pytest.fixture(scope="module")
def shipments():
    return pd.read_csv(PROCESSED_PATH)


@pytest.fixture(scope="module")
def model():
    return load_model(MODEL_PATH)


def test_raw_values_labels_and_context_preserved(records):
    cleaned = clean(read_raw_shipments(RAW_PATH))
    expected = records.sort_values(
        "Shipment_ID"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        cleaned[SHIPMENT_COLUMNS + [TARGET_COLUMN]],
        expected[SHIPMENT_COLUMNS + [TARGET_COLUMN]],
    )

    assert len(cleaned) == 5000
    assert cleaned[TARGET_COLUMN].sum() == 3063
    assert cleaned.Needs_Review.sum() == 975

    original = build(records, seed=42)
    flipped = build(
        records.assign(
            **{TARGET_COLUMN: 1 - records[TARGET_COLUMN]}
        ),
        seed=42,
    )

    pd.testing.assert_frame_equal(
        original[SHIPMENT_COLUMNS + CONTEXT_COLUMNS],
        flipped[SHIPMENT_COLUMNS + CONTEXT_COLUMNS],
    )
    pd.testing.assert_frame_equal(
        extract_features(original),
        extract_features(records),
    )

    assert len(MODEL_FEATURE_COLUMNS) == 11
    assert not set(
        CONTEXT_COLUMNS + [TARGET_COLUMN, "Needs_Review"]
    ).intersection(MODEL_FEATURE_COLUMNS)


@pytest.mark.parametrize("number", [3, 4, 5])
def test_features_match_each_notebooks_actual_code(
    shipments,
    number,
):
    reference = notebook_function(
        number, "add_features"
    )(shipments)

    pd.testing.assert_frame_equal(
        extract_features(shipments),
        reference[MODEL_FEATURE_COLUMNS],
        check_dtype=False,
    )


def test_predictions_match_saved_primary_for_all_5000(
    shipments,
    model,
):
    reference = notebook_function(
        5, "add_features"
    )(shipments)

    expected = model.pipeline.predict_proba(
        reference[MODEL_FEATURE_COLUMNS]
    )[:, 1]

    result = predict_failure_risk(shipments, model)

    np.testing.assert_array_equal(
        result.p_disruption,
        expected,
    )
    np.testing.assert_array_equal(
        result.p_failure,
        expected,
    )

    assert result.index.equals(shipments.index)


def test_missing_and_unknown_weather_use_notebook_fallback(
    shipments,
    model,
):
    rows = shipments.iloc[:3].copy().astype(
        {"Weather_Condition": object}
    )
    rows["Weather_Condition"] = [
        None, "Volcanic ash", "Clear",
    ]

    featured = notebook_function(
        5, "add_features"
    )(rows)

    expected_fallback = model.fallback_pipeline.predict_proba(
        featured.iloc[:2][model.fallback_features]
    )[:, 1]

    predicted = predict_failure_risk(rows, model)

    np.testing.assert_array_equal(
        predicted.p_disruption.iloc[:2],
        expected_fallback,
    )

    assert predicted.p_disruption.iloc[2] == (
        model.pipeline.predict_proba(
            featured.iloc[[2]][MODEL_FEATURE_COLUMNS]
        )[0, 1]
    )


def test_only_six_recorded_fields_needed(shipments, model):
    fields = [
        "Weather_Condition",
        "Geopolitical_Risk_Score",
        "Carrier_Reliability_Score",
        "Lead_Time_Days",
        "Distance_km",
        "Transport_Mode",
    ]

    rows = shipments.iloc[:5]

    pd.testing.assert_frame_equal(
        predict_failure_risk(rows[fields], model),
        predict_failure_risk(rows, model),
    )

    changed = rows.assign(
        Disruption_Occurred=1 - rows.Disruption_Occurred,
        Cargo_Value_USD_Simulated=1e12,
        Remaining_Buffer_Hours_Simulated=0,
        Elapsed_Transit_Hours_Simulated=1e9,
        Needs_Review=1,
    )

    pd.testing.assert_frame_equal(
        predict_failure_risk(rows, model),
        predict_failure_risk(changed, model),
    )


def test_numeric_imputation_but_insufficient_inputs_are_rejected(
    shipments,
    model,
):
    row = shipments.iloc[[0]].copy()
    row["Geopolitical_Risk_Score"] = np.nan
    row["Carrier_Reliability_Score"] = np.nan

    assert predict_failure_risk(
        row, model
    ).p_disruption.between(0, 1).all()

    row["Weather_Condition"] = None
    row["Lead_Time_Days"] = np.nan

    with pytest.raises(
        ValueError, match="[Ii]nsufficient"
    ):
        predict_failure_risk(row, model)


@pytest.mark.parametrize(
    "field,value",
    [
        ("Distance_km", 0),
        ("Distance_km", np.inf),
        ("Lead_Time_Days", -1),
        ("Carrier_Reliability_Score", 2),
        ("Geopolitical_Risk_Score", True),
    ],
)
def test_prediction_rejects_invalid_present_numbers(
    shipments,
    model,
    field,
    value,
):
    row = shipments.iloc[[0]].copy()
    row[field] = [value]

    with pytest.raises(ValueError, match=field):
        predict_failure_risk(row, model)


@pytest.mark.parametrize(
    "probability,level",
    [
        (0, "Low"),
        (.299, "Low"),
        (.30, "Medium"),
        (.60, "High"),
        (.80, "Critical"),
        (1, "Critical"),
    ],
)
def test_notebook_risk_boundaries(probability, level):
    assert risk_level(probability) == level


def test_verification_reproduces_notebook_split_without_fitting(
    shipments,
    model,
    monkeypatch,
):
    from sklearn.pipeline import Pipeline

    monkeypatch.setattr(
        Pipeline,
        "fit",
        lambda *a, **k: pytest.fail(
            "Verification must never train."
        ),
    )

    before = hashlib.sha256(
        MODEL_PATH.read_bytes()
    ).hexdigest()

    report = evaluate_model(shipments, model)
    assert report

    _, test = train_test_split(
        shipments,
        test_size=.20,
        stratify=shipments[TARGET_COLUMN],
        random_state=42,
    )

    p = predict_failure_risk(
        test, model
    ).p_disruption

    assert round(
        roc_auc_score(test[TARGET_COLUMN], p), 4
    ) == model.package["test_scores"]["ROC AUC"]

    assert report["all_reported_scores_match"]

    assert hashlib.sha256(
        MODEL_PATH.read_bytes()
    ).hexdigest() == before

    with pytest.raises(
        ValueError, match="Notebook 05"
    ):
        train_model(shipments)


def test_cli_and_helpers_do_not_replace_saved_model(
    tmp_path,
    monkeypatch,
):
    report = tmp_path / "evaluation.json"
    before = MODEL_PATH.read_bytes()

    assert verify_main(
        ["--report", str(report)]
    ) == 0
    assert json.loads(report.read_text())
    assert MODEL_PATH.read_bytes() == before

    monkeypatch.setattr(
        helpers,
        "MODEL_PATH",
        tmp_path / "missing.pkl",
    )

    with pytest.raises(FileNotFoundError):
        helpers.notebook_evaluate(
            pd.read_csv(PROCESSED_PATH)
        )

    with pytest.raises(
        ValueError, match="Notebook 05"
    ):
        helpers.notebook_train(
            pd.read_csv(PROCESSED_PATH),
            "supplied",
        )

    assert verify_main(
        ["--report", str(MODEL_PATH)]
    ) == 2

    assert MODEL_PATH.read_bytes() == before


@pytest.mark.parametrize(
    "payload",
    [
        "<<<<<<< ours\nShipment_ID\na\n",
        "Shipment_ID,Shipment_ID\na,b\n",
        "Shipment_ID,Date\na\n",
        "Origin_Port,Place_of_Origin\na,b\n",
    ],
)
def test_loader_rejects_conflicts_duplicates_or_ragged_csv(
    payload,
):
    with pytest.raises(ShipmentValidationError):
        read_raw_shipments(StringIO(payload))


def test_loader_preserves_literal_na_and_cursor():
    stream = StringIO(
        "Shipment_ID,Origin_Port,Weight_MT\n0007,NA,\n"
    )
    stream.seek(5)
    raw = read_raw_shipments(stream)

    assert stream.tell() == 5
    assert raw.iloc[0].Shipment_ID == "0007"
    assert raw.iloc[0].Origin_Port == "NA"
    assert pd.isna(raw.iloc[0].Weight_MT)


def test_predictions_need_no_network(
    shipments,
    model,
    monkeypatch,
):
    import socket

    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *a, **k: pytest.fail(
            "No external API is required."
        ),
    )

    assert predict_failure_risk(
        shipments.iloc[:10],
        model,
    ).p_disruption.between(0, 1).all()


@pytest.mark.parametrize(
    "payload",
    [b"not a pickle", None],
)
def test_obsolete_or_invalid_model_is_rejected(
    tmp_path,
    payload,
):
    path = tmp_path / "bad.pkl"

    if payload is None:
        joblib.dump(
            {"model": "obsolete cold-chain model"},
            path,
        )
    else:
        path.write_bytes(payload)

    with pytest.raises(ValueError):
        load_model(path)


def test_all_actions_match_notebook_transforms_and_probabilities(
    shipments,
    model,
):
    from src.recommendations import ACTION_CATALOG, apply_action

    reference_action = notebook_function(
        5, "apply_action"
    )

    for item in ACTION_CATALOG.values():
        expected = reference_action(
            shipments,
            **item["changes"],
        )
        actual = apply_action(
            shipments,
            **item["changes"],
        )

        pd.testing.assert_frame_equal(
            actual[expected.columns],
            expected,
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            predict_failure_risk(actual, model),
            predict_failure_risk(expected, model),
        )


def test_pharmaceutical_queue_matches_notebook_priority(
    shipments,
    model,
):
    from src.prioritize import calculate_triage_queue

    rows = shipments.join(
        predict_failure_risk(shipments, model)
    )

    expected = rows[
        rows.p_disruption.ge(.60)
        & rows.Product_Category.eq("Pharmaceuticals")
    ].copy()

    weights = expected.Hospital_Urgency_Tier_Simulated.map(
        {
            "Routine Restock": 1.0,
            "Elevated Procedure": 1.5,
            "Critical Emergency": 2.5,
        }
    )

    priority = (
        expected.p_disruption
        * expected.Cargo_Value_USD_Simulated
        * weights
        / expected.Remaining_Buffer_Hours_Simulated.clip(lower=.5)
    )

    expected["priority"] = priority
    expected = expected.sort_values(
        "priority",
        ascending=False,
        kind="mergesort",
    )

    actual = calculate_triage_queue(rows)

    assert actual.Shipment_ID.tolist() == expected.Shipment_ID.tolist()
    np.testing.assert_array_equal(
        actual.priority,
        expected.priority,
    )


def test_dry_ice_changes_only_context_and_ambient_actions_work(
    shipments,
    model,
):
    from src.recommendations import (
        get_recommended_action,
        simulate_intervention,
    )

    cold = shipments[
        shipments.Storage_Standard_Simulated.ne("Ambient")
    ].iloc[0]

    result = simulate_intervention(
        cold,
        "refrigerant_refresh",
        model,
    )

    assert result["probability_change"] == 0
    assert result["estimated_net_benefit_usd"] == -120
    assert (
        result["changed_fields"][
            "Remaining_Buffer_Hours_Simulated"
        ]["after"]
        - cold.Remaining_Buffer_Hours_Simulated
    ) == 48

    ambient = shipments[
        shipments.Storage_Standard_Simulated.eq("Ambient")
    ].iloc[0]

    p = predict_failure_risk(
        ambient.to_frame().T,
        model,
    ).p_disruption.iloc[0]

    chosen = get_recommended_action(
        "Routine monitoring",
        p,
        shipment_data=ambient,
        model=model,
    )

    assert chosen["simulation"]["estimated_net_benefit_usd"] >= 0


def test_best_action_matches_notebook_assumed_net_savings(
    shipments,
    model,
):
    from src.recommendations import (
        ACTION_CATALOG,
        get_recommended_action,
    )

    reference_action = notebook_function(
        5, "apply_action"
    )

    alerted = shipments[
        predict_failure_risk(
            shipments, model
        ).p_disruption.ge(.6)
    ].iloc[:10]

    for _, row in alerted.iterrows():
        original = row.to_frame().T
        before = predict_failure_risk(
            original, model
        ).p_disruption.iloc[0]

        net = {}

        for key, item in ACTION_CATALOG.items():
            changed = reference_action(
                original,
                **item["changes"],
            )

            after = predict_failure_risk(
                changed, model
            ).p_disruption.iloc[0]

            net[key] = (
                (before - after)
                * .10
                * row.Cargo_Value_USD_Simulated
                - item["cost_usd"]
            )

        result = get_recommended_action(
            "Combined exposure",
            before,
            shipment_data=row,
            model=model,
        )

        assert result["action_id"] == max(
            net,
            key=net.get,
        )

        assert result["simulation"][
            "estimated_net_benefit_usd"
        ] == pytest.approx(max(net.values()))


def test_weather_explanation_follows_coupled_notebook_reference(
    shipments,
    model,
):
    from src.explain import explain_prediction

    row = shipments[
        shipments.Weather_Condition.eq("Hurricane")
    ].iloc[0]

    explanation = explain_prediction(row, model)

    driver = next(
        d
        for d in explanation["drivers"]
        if d["feature"] == "Weather_Condition"
    )

    expected = notebook_function(
        5, "with_weather"
    )(row.to_frame().T, "Clear")

    probability = predict_failure_risk(
        expected, model
    ).p_disruption.iloc[0]

    assert driver["counterfactual_probability"] == probability
    assert driver["contribution"] == (
        explanation["baseline_probability"] - probability
    )
    assert set(driver["changed_fields"]) == {
        "Weather_Condition", "Lead_Time_Days",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        (TARGET_COLUMN, True),
        (TARGET_COLUMN, 2),
        (TARGET_COLUMN, np.nan),
        ("Weight_MT", -1),
        ("Fuel_Price_Index", np.inf),
        ("Date", "2025-02-30"),
    ],
)
def test_record_validation_remains_strict(
    records,
    field,
    value,
):
    row = records.iloc[[0]].copy()
    row[field] = [value]

    with pytest.raises(
        ShipmentValidationError,
        match=field,
    ):
        clean_raw_shipment_data(
            row,
            require_target=True,
        )


def test_saved_model_feature_and_dependency_compatibility(
    model,
    tmp_path,
    monkeypatch,
):
    broken = dict(
        model.package,
        features=model.package["features"] + [TARGET_COLUMN],
    )
    path = tmp_path / "wrong_features.pkl"

    joblib.dump(
        broken,
        path,
        compress=3,
    )

    with pytest.raises(
        ValueError, match="feature"
    ):
        load_model(path)

    import src.model as engine

    monkeypatch.setattr(
        engine.sklearn,
        "__version__",
        "0.0",
    )
    monkeypatch.setattr(
        engine.joblib,
        "load",
        lambda *a, **k: pytest.fail(
            "Reject mismatched sklearn before loading."
        ),
    )

    with pytest.raises(
        ValueError,
        match="matching dependencies",
    ):
        load_model(MODEL_PATH)