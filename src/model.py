"""Predict the supplied disruption label, not cold-chain or clinical failure."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import extract_features
from .preprocessing import clean_raw_shipment_data
from .schema import (
    FEATURE_SCHEMA_VERSION,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    TARGET_COLUMN,
)

ARTIFACT_FORMAT_VERSION = 3


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    comparisons: list[dict[str, Any]]


def risk_level(probability):
    try:
        probability = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Risk probability must be a number from 0 to 1."
        ) from exc

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "Risk probability must be finite and from 0 to 1."
        )

    return (
        "Low" if probability <= .3
        else "Medium" if probability <= .6
        else "High" if probability <= .8
        else "Critical"
    )


def _checked_features(features):
    if (
        not isinstance(features, pd.DataFrame)
        or features.columns.duplicated().any()
    ):
        raise ValueError(
            "Supply a feature DataFrame with unique column names."
        )

    missing = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in features
    ]
    if missing:
        raise ValueError(
            f"Missing model features: {missing}. Run extract_features first."
        )

    result = features.loc[:, MODEL_FEATURE_COLUMNS].copy()

    for column in MODEL_NUMERIC_COLUMNS:
        try:
            values = pd.to_numeric(
                result[column], errors="raise"
            )
            if np.iscomplexobj(values.to_numpy()):
                raise ValueError("Complex values are unsupported.")
            result[column] = values.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature {column!r} must contain real numeric values."
            ) from exc

        if np.isinf(result[column].to_numpy()).any():
            raise ValueError(
                f"Feature {column!r} contains infinity; check its units."
            )

    for column in MODEL_CATEGORICAL_COLUMNS:
        result[column] = result[column].astype(object).where(
            result[column].notna(), np.nan
        )

    return result


def _preprocessor():
    numeric = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])

    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric, MODEL_NUMERIC_COLUMNS),
            ("categorical", categorical, MODEL_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def _temporal_splits(clean, labels):
    if clean["Shipment_ID"].duplicated().any():
        raise ValueError(
            "Repeated Shipment_ID values are unsupported for temporal "
            "training; supply independent shipments."
        )

    dates = clean["Date"].dt.normalize()
    unique_dates = pd.DatetimeIndex(
        dates.drop_duplicates().sort_values()
    )

    if len(unique_dates) < 4:
        raise ValueError(
            "Need at least four distinct record dates for chronological "
            "holdout and training CV."
        )

    cutoff = unique_dates[int(np.floor(len(unique_dates) * .8))]
    train = np.flatnonzero(dates < cutoff)
    test = np.flatnonzero(dates >= cutoff)

    if any(
        len(np.unique(labels[positions])) != 2
        for positions in [train, test]
    ):
        raise ValueError(
            "Both outcomes must appear in chronological training and "
            "holdout. Add more dated records."
        )

    train_dates = dates.iloc[train].reset_index(drop=True)
    calendar = pd.DatetimeIndex(
        train_dates.drop_duplicates().sort_values()
    )

    folds = min(5, len(calendar) - 1)
    if folds < 2:
        raise ValueError(
            "Need at least three training dates for expanding temporal CV."
        )

    cv = []
    ranges = []

    for past, future in TimeSeriesSplit(n_splits=folds).split(calendar):
        a = np.flatnonzero(train_dates.isin(calendar[past]))
        b = np.flatnonzero(train_dates.isin(calendar[future]))

        if (
            len(np.unique(labels[train][a])) != 2
            or len(np.unique(labels[train][b])) != 2
        ):
            raise ValueError(
                "Every temporal CV training/validation block needs both "
                "outcomes; add more dated records."
            )

        cv.append((a, b))
        ranges.append({
            "train_end": calendar[past[-1]].date().isoformat(),
            "validation_start": calendar[future[0]].date().isoformat(),
            "validation_end": calendar[future[-1]].date().isoformat(),
        })

    return train, test, cv, ranges, cutoff


def dataset_fingerprint(df):
    clean = clean_raw_shipment_data(df, require_target=True)
    rows = _checked_features(extract_features(clean))

    rows["Shipment_ID"] = clean["Shipment_ID"].astype(str).to_numpy()
    rows["Date"] = clean["Date"].dt.strftime("%Y-%m-%d").to_numpy()
    rows[TARGET_COLUMN] = clean[TARGET_COLUMN].to_numpy(dtype=int)
    rows["Label_Provenance"] = (
        clean["Label_Provenance"].astype(str).to_numpy()
        if "Label_Provenance" in clean
        else "supplied label; authenticity unverified"
    )

    digest = hashlib.sha256(
        json.dumps(
            [FEATURE_SCHEMA_VERSION, TARGET_COLUMN, *rows.columns]
        ).encode()
    )
    digest.update(
        pd.util.hash_pandas_object(
            rows, index=False
        ).to_numpy().tobytes()
    )
    return digest.hexdigest()


def train_model(
    df,
    target=TARGET_COLUMN,
    data_source="user dataset",
    model_path=None,
    random_state=42,
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(
            "Training needs a nonempty dated shipment DataFrame "
            "with disruption labels."
        )

    if target != TARGET_COLUMN:
        raise ValueError(
            f"This contract predicts {TARGET_COLUMN}; "
            "do not substitute cold-chain failure labels."
        )

    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError(
            "Supply a data_source description for provenance."
        )

    clean = clean_raw_shipment_data(df, require_target=True)
    features = _checked_features(extract_features(clean))
    labels = clean[TARGET_COLUMN].to_numpy(dtype=int)

    if (
        set(labels) != {0, 1}
        or min(np.bincount(labels, minlength=2)) < 3
    ):
        raise ValueError(
            "Need both disruption outcomes and at least three "
            "independent rows per class."
        )

    train, test, cv, ranges, cutoff = _temporal_splits(clean, labels)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            random_state=random_state,
        ),
    }

    scoring = {
        "roc_auc": "roc_auc",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    comparisons = []
    pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", _preprocessor()),
            ("classifier", estimator),
        ])

        scores = cross_validate(
            pipeline,
            features.iloc[train],
            labels[train],
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        row = {
            "model": name,
            "model_name": name,
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
        }
        row.update({
            f"cv_{metric}": float(
                np.mean(scores[f"test_{metric}"])
            )
            for metric in scoring
        })

        comparisons.append(row)
        pipelines[name] = pipeline

    selected = max(
        comparisons,
        key=lambda row: (row["cv_roc_auc"], row["cv_f1"]),
    )

    pipeline = pipelines[selected["model"]].fit(
        features.iloc[train],
        labels[train],
    )

    probabilities = pipeline.predict_proba(
        features.iloc[test]
    )[:, list(pipeline.classes_).index(1)]

    predicted = (probabilities >= .5).astype(int)
    actual = labels[test]

    metrics = {
        name: float(metric(actual, predicted, zero_division=0))
        for name, metric in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]
    }
    metrics.update(
        roc_auc=float(roc_auc_score(actual, probabilities)),
        accuracy=float(accuracy_score(actual, predicted)),
        confusion_matrix=confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
        n_test=int(len(test)),
        classification_report=classification_report(
            actual,
            predicted,
            output_dict=True,
            zero_division=0,
        ),
    )

    provenance = (
        "; ".join(
            clean["Label_Provenance"].dropna().astype(str).unique()
        )
        if "Label_Provenance" in clean
        else "Disruption_Occurred supplied; authenticity unverified"
    )

    synthetic = (
        True if re.search(r"synthetic|simulat", provenance, re.I)
        else None
    )

    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_column": TARGET_COLUMN,
        "target_semantics": (
            "Recorded disruption label; not clinical or cold-chain failure"
        ),
        "model_name": selected["model"],
        "model_source": data_source,
        "data_source": data_source,
        "source_description": data_source,
        "is_synthetic": synthetic,
        "label_source": provenance,
        "label_provenance": "unverified",
        "provenance_note": (
            "Supplied labels are not independently authenticated; "
            "unmarked does not establish real-world origin."
        ),
        "features": list(MODEL_FEATURE_COLUMNS),
        "n_rows": len(features),
        "n_train": len(train),
        "n_test": len(test),
        "train_rows": len(train),
        "selection_rows": len(train),
        "holdout_rows": len(test),
        "train_positions": train.tolist(),
        "test_positions": test.tolist(),
        "train_group_count": len(train),
        "test_group_count": len(test),
        "split_method": "chronological holdout at date boundary",
        "cv_method": "TimeSeriesSplit on distinct dates",
        "cv_folds": len(cv),
        "cv_date_ranges": ranges,
        "selection_metric": "roc_auc",
        "decision_threshold": .5,
        "cv_splits": [
            {
                "train_positions": train[a].tolist(),
                "validation_positions": train[b].tolist(),
            }
            for a, b in cv
        ],
        "holdout_start": cutoff.date().isoformat(),
        "holdout_fraction": len(test) / len(features),
        "calibrated": False,
        "date_assumption": (
            "Date is assumed to be the pre-outcome shipment record date; "
            "this has not been confirmed."
        ),
        "planned_lead_time_confirmed": True,
        "excluded_pending_confirmation": [],
        "input_availability_note": (
            "User confirmed Lead_Time_Days is planned and pre-outcome "
            "for this dataset; other input timing is assumed."
        ),
        "calibration_note": (
            "Probability calibration has not been assessed; "
            "outputs are classifier scores."
        ),
        "prediction_horizon": (
            "Not established; label timing and input availability "
            "require confirmation."
        ),
        "dataset_fingerprint": dataset_fingerprint(clean),
        "sklearn_version": sklearn.__version__,
        "random_state": int(random_state),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    bundle = ModelBundle(
        pipeline, metadata, metrics, comparisons
    )

    if model_path is not None:
        save_model(bundle, model_path)

    return bundle


def _validate_bundle(bundle):
    if (
        not isinstance(bundle, ModelBundle)
        or not isinstance(bundle.pipeline, Pipeline)
        or not isinstance(bundle.metadata, dict)
        or not isinstance(bundle.metrics, dict)
        or not isinstance(bundle.comparisons, list)
    ):
        raise ValueError(
            "Expected a disruption-v2 ModelBundle; retrain old notebook "
            "dictionaries and cold-chain artifacts."
        )

    expected = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_column": TARGET_COLUMN,
        "features": MODEL_FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
    }

    if any(
        bundle.metadata.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError(
            "Target, feature contract, artifact format or sklearn "
            "version changed; retrain for disruption-v2."
        )

    if (
        not bundle.metadata.get("data_source")
        or not bundle.metadata.get("model_source")
        or not hasattr(bundle.pipeline, "classes_")
        or set(bundle.pipeline.classes_) != {0, 1}
    ):
        raise ValueError(
            "Model lacks fitted binary classes or provenance; retrain it."
        )


def predict_failure_risk(features_df, model):
    """p_disruption is primary; p_failure is an identical compatibility alias."""
    _validate_bundle(model)
    features = _checked_features(features_df)

    try:
        probabilities = (
            model.pipeline.predict_proba(features)[
                :, list(model.pipeline.classes_).index(1)
            ]
            if len(features)
            else np.array([], dtype=float)
        )
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "Cannot predict these inputs; check features or retrain the model."
        ) from exc

    if (
        not np.isfinite(probabilities).all()
        or (
            (probabilities < -1e-12)
            | (probabilities > 1 + 1e-12)
        ).any()
    ):
        raise ValueError(
            "Model returned invalid probabilities; retrain it."
        )

    probabilities = np.clip(probabilities, 0, 1)

    return pd.DataFrame(
        {
            "p_disruption": probabilities,
            "p_failure": probabilities,
            "risk_level": [risk_level(p) for p in probabilities],
            "target_column": TARGET_COLUMN,
            "model_source": model.metadata["model_source"],
        },
        index=features.index,
    )


def _local_artifact_path(path):
    if (
        "://" in str(path)
        or Path(path).suffix.lower() not in {".pkl", ".joblib"}
    ):
        raise ValueError(
            "Use a trusted local .pkl or .joblib path; "
            "never an uploaded pickle."
        )
    return Path(path).expanduser()


def save_model(bundle, path):
    _validate_bundle(bundle)
    destination = _local_artifact_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".pkl",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model(path):
    """Trusted local project artifacts only: joblib/pickle can execute code."""
    artifact = _local_artifact_path(path)

    if not artifact.is_file():
        raise FileNotFoundError(
            f"Missing model {artifact}; train a disruption-v2 artifact first."
        )

    try:
        bundle = joblib.load(artifact)
    except Exception as exc:
        raise ValueError(
            f"Cannot load {artifact}; retrain with this environment."
        ) from exc

    _validate_bundle(bundle)
    return bundle