from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from .features import extract_features
from .preprocessing import clean_raw_shipment_data
from .schema import (
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    comparisons: list[dict[str, Any]]


def risk_level(probability: float) -> str:
    try:
        probability = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Risk probability must be a number between 0 and 1."
        ) from exc

    if (
        not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError(
            "Risk probability must be a finite number "
            "between 0 and 1."
        )

    if probability <= 0.3:
        return "Low"
    if probability <= 0.6:
        return "Medium"
    if probability <= 0.8:
        return "High"
    return "Critical"


def _checked_features(
    features: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame):
        raise ValueError(
            "Model inputs must be a pandas DataFrame "
            "of engineered features."
        )

    if features.columns.duplicated().any():
        raise ValueError(
            "Model inputs contain duplicate column names; "
            "rename them first."
        )

    missing = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in features
    ]
    if missing:
        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
            + ". Run extract_features first."
        )

    result = features.loc[:, MODEL_FEATURE_COLUMNS].copy()

    for column in MODEL_NUMERIC_COLUMNS:
        try:
            result[column] = pd.to_numeric(
                result[column], errors="raise"
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Model feature {column!r} "
                "must contain numeric values."
            ) from exc

        if np.isinf(result[column].to_numpy()).any():
            raise ValueError(
                f"Model feature {column!r} contains infinity; "
                "correct the input."
            )

    for column in MODEL_CATEGORICAL_COLUMNS:
        result[column] = result[column].where(
            result[column].notna(), np.nan
        )

    return result


def _preprocessor() -> ColumnTransformer:
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
            (
                "categorical",
                categorical,
                MODEL_CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def _class_group_counts(
    labels: np.ndarray,
    groups: np.ndarray,
) -> list[int]:
    return [
        len(set(groups[labels == label]))
        for label in (0, 1)
    ]


def _cv_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    grouped: bool,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    support = (
        _class_group_counts(labels, groups)
        if grouped
        else np.bincount(labels, minlength=2)
    )
    max_folds = min(5, int(min(support)))

    for folds in range(max_folds, 1, -1):
        if grouped:
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            )
            splits = list(
                splitter.split(
                    np.zeros(len(labels)), labels, groups
                )
            )
        else:
            splitter = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=random_state,
            )
            splits = list(
                splitter.split(
                    np.zeros(len(labels)), labels
                )
            )

        if all(
            len(np.unique(labels[train])) == 2
            and len(np.unique(labels[valid])) == 2
            for train, valid in splits
        ):
            return splits

    raise ValueError(
        "Cannot form training cross-validation folds "
        "with both outcomes. Provide more independent shipments "
        "with failure and non-failure labels; "
        "do not duplicate rows."
    )


def _split_data(
    labels: np.ndarray,
    groups: np.ndarray,
    random_state: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray]],
    bool,
]:
    grouped = len(set(groups)) < len(groups)
    support = _class_group_counts(labels, groups)

    if min(support) < 3:
        raise ValueError(
            "Training requires at least 3 distinct Shipment_ID "
            "groups supporting each outcome (0 and 1): "
            "one for holdout and at least two for training CV. "
            f"Found {support[0]} non-failure "
            f"and {support[1]} failure groups."
        )

    if not grouped:
        rng = np.random.RandomState(random_state)
        train_parts, test_parts = [], []

        for label in (0, 1):
            positions = np.flatnonzero(labels == label)
            rng.shuffle(positions)

            reserved = max(
                1,
                min(
                    int(np.ceil(len(positions) * 0.2)),
                    len(positions) - 2,
                ),
            )
            test_parts.append(positions[:reserved])
            train_parts.append(positions[reserved:])

        train = rng.permutation(
            np.concatenate(train_parts)
        )
        test = rng.permutation(
            np.concatenate(test_parts)
        )

        return (
            train,
            test,
            _cv_splits(
                labels[train],
                groups[train],
                False,
                random_state,
            ),
            False,
        )

    candidates = []

    for folds in range(min(5, min(support)), 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        for train, test in splitter.split(
            np.zeros(len(labels)), labels, groups
        ):
            if (
                len(np.unique(labels[test])) != 2
                or min(
                    _class_group_counts(
                        labels[train], groups[train]
                    )
                ) < 2
            ):
                continue

            try:
                cv = _cv_splits(
                    labels[train],
                    groups[train],
                    True,
                    random_state,
                )
            except ValueError:
                continue

            rank = (
                abs(len(test) / len(labels) - 0.2),
                abs(
                    float(labels[test].mean())
                    - float(labels.mean())
                ),
            )
            candidates.append((rank, train, test, cv))

    if not candidates:
        raise ValueError(
            "Repeated Shipment_ID rows cannot be split into "
            "a grouped holdout and training CV with both outcomes. "
            "Add more independent labeled shipments or review "
            "duplicate identifiers. All rows of one shipment "
            "must stay together."
        )

    _, train, test, cv = min(
        candidates, key=lambda item: item[0]
    )
    return train, test, cv, True


def _fingerprint(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> str:
    rows = features.copy()
    rows["Shipment_ID"] = groups
    rows[TARGET_COLUMN] = labels

    digest = hashlib.sha256(
        json.dumps(list(rows.columns)).encode("utf-8")
    )
    digest.update(
        pd.util.hash_pandas_object(
            rows, index=False
        ).to_numpy().tobytes()
    )
    return digest.hexdigest()


def train_model(
    df: pd.DataFrame,
    target: str = TARGET_COLUMN,
    data_source: str = "user dataset",
    model_path: str | Path | None = None,
    random_state: int = 42,
) -> ModelBundle:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(
            "Training needs a non-empty shipment DataFrame "
            "with observed binary labels."
        )

    if (
        not isinstance(data_source, str)
        or not data_source.strip()
    ):
        raise ValueError(
            "Provide a data_source description "
            "for model provenance."
        )

    source = df.copy()

    if target != TARGET_COLUMN:
        if target not in source:
            raise ValueError(
                f"Target column {target!r} is missing. "
                "Set --target or map the label column."
            )
        source[TARGET_COLUMN] = source[target]
        source = source.drop(columns=[target])

    clean = clean_raw_shipment_data(
        source, require_target=True
    )
    features = _checked_features(extract_features(clean))

    try:
        numeric_labels = pd.to_numeric(
            clean[TARGET_COLUMN], errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Observed target labels must be binary 0/1 values."
        ) from exc

    if (
        numeric_labels.isna().any()
        or not numeric_labels.isin([0, 1]).all()
    ):
        raise ValueError(
            "Observed target labels must contain only 0 and 1, "
            "with no missing labels."
        )

    labels = numeric_labels.to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError(
            "Training requires both observed outcomes: "
            "non-failure (0) and failure (1)."
        )

    groups = clean["Shipment_ID"].astype(str).to_numpy()

    synthetic_rows = (
        clean["Shipment_ID"]
        .astype(str)
        .str.upper()
        .str.startswith("SYNTHETIC-")
        .to_numpy()
    )
    if "Data_Source" in clean.columns:
        synthetic_rows = synthetic_rows | (
            clean["Data_Source"]
            .astype(str)
            .str.contains(
                "synthetic", case=False, na=False
            )
            .to_numpy()
        )

    if synthetic_rows.any() and not synthetic_rows.all():
        raise ValueError(
            "Synthetic and unmarked shipment rows are mixed. "
            "Train demo and real datasets separately; "
            "synthetic outcomes must not enter reported "
            "real-data metrics."
        )

    synthetic = (
        bool(synthetic_rows.any())
        or "synthetic" in data_source.lower()
        or data_source.strip().lower() == "demo"
    )
    source_label = data_source
    if synthetic and "synthetic" not in data_source.lower():
        source_label = (
            f"synthetic demonstration data from {data_source}"
        )

    train, test, cv, grouped = _split_data(
        labels, groups, random_state
    )
    train_features = features.iloc[train]
    train_labels = labels[train]

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
        "precision": make_scorer(
            precision_score, zero_division=0
        ),
        "recall": make_scorer(
            recall_score, zero_division=0
        ),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    comparisons, pipelines = [], {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", _preprocessor()),
            ("classifier", estimator),
        ])

        scores = cross_validate(
            pipeline,
            train_features,
            train_labels,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        comparison = {
            "model": name,
            "model_name": name,
        }

        for metric in scoring:
            comparison[f"cv_{metric}"] = float(
                np.mean(scores[f"test_{metric}"])
            )

        comparison["cv_roc_auc_std"] = float(
            np.std(scores["test_roc_auc"])
        )
        comparisons.append(comparison)
        pipelines[name] = pipeline

    selected = max(
        comparisons,
        key=lambda row: (
            row["cv_roc_auc"], row["cv_f1"]
        ),
    )

    pipeline = pipelines[selected["model"]]
    pipeline.fit(train_features, train_labels)

    class_position = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(
        features.iloc[test]
    )[:, class_position]
    predicted = (probabilities >= 0.5).astype(int)
    heldout_labels = labels[test]

    metrics = {
        "precision": float(
            precision_score(
                heldout_labels, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                heldout_labels, predicted, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                heldout_labels, predicted, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(heldout_labels, probabilities)
        ),
        "accuracy": float(
            accuracy_score(heldout_labels, predicted)
        ),
        "confusion_matrix": confusion_matrix(
            heldout_labels, predicted, labels=[0, 1]
        ).tolist(),
        "n_test": int(len(test)),
    }

    metadata = {
        "artifact_format_version": 1,
        "model_name": selected["model"],
        "model_source": source_label,
        "data_source": source_label,
        "source_description": data_source,
        "is_synthetic": synthetic,
        "label_source": (
            "synthetic labels (demonstration only)"
            if synthetic
            else f"user-supplied observed labels: {target}"
        ),
        "target_column": target,
        "features": list(MODEL_FEATURE_COLUMNS),
        "n_rows": int(len(features)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_rows": int(len(train)),
        "selection_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "train_positions": train.tolist(),
        "test_positions": test.tolist(),
        "train_group_count": int(len(set(groups[train]))),
        "test_group_count": int(len(set(groups[test]))),
        "split_method": (
            "stratified grouped holdout"
            if grouped
            else "stratified holdout"
        ),
        "cv_method": (
            "StratifiedGroupKFold"
            if grouped
            else "StratifiedKFold"
        ),
        "cv_folds": int(len(cv)),
        "selection_metric": "roc_auc",
        "decision_threshold": 0.5,
        "holdout_fraction": float(
            len(test) / len(features)
        ),
        "calibrated": False,
        "calibration_note": (
            "Probability calibration has not been assessed; "
            "these are classifier scores."
        ),
        "prediction_horizon": (
            "Not established; requires appropriately timed "
            "real observed labels."
        ),
        "dataset_fingerprint": _fingerprint(
            features, labels, groups
        ),
        "sklearn_version": sklearn.__version__,
        "random_state": int(random_state),
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    bundle = ModelBundle(
        pipeline, metadata, metrics, comparisons
    )

    if model_path is not None:
        save_model(bundle, model_path)

    return bundle


def predict_failure_risk(
    features_df: pd.DataFrame,
    model: ModelBundle,
) -> pd.DataFrame:
    if (
        not isinstance(model, ModelBundle)
        or not hasattr(model.pipeline, "predict_proba")
        or not isinstance(model.metadata, dict)
    ):
        raise ValueError(
            "A trained Aura ModelBundle is required. "
            "Train or load a trusted local model first."
        )

    if model.metadata.get("features") != MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "Model feature schema differs from this project. "
            "Retrain the model with current code."
        )

    features = _checked_features(features_df)

    if features.empty:
        return pd.DataFrame({
            "p_failure": pd.Series(
                index=features.index, dtype=float
            ),
            "risk_level": pd.Series(
                index=features.index, dtype=str
            ),
            "model_source": pd.Series(
                index=features.index, dtype=str
            ),
        })

    try:
        class_position = list(
            model.pipeline.classes_
        ).index(1)
        probabilities = np.asarray(
            model.pipeline.predict_proba(features)[
                :, class_position
            ],
            dtype=float,
        )
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "The model cannot predict these inputs. "
            "Check feature values or retrain the artifact."
        ) from exc

    if (
        not np.isfinite(probabilities).all()
        or (
            (probabilities < -1e-12)
            | (probabilities > 1 + 1e-12)
        ).any()
    ):
        raise ValueError(
            "The model produced invalid probabilities; "
            "retrain the artifact."
        )

    probabilities = np.clip(probabilities, 0, 1)

    return pd.DataFrame(
        {
            "p_failure": probabilities,
            "risk_level": [
                risk_level(p) for p in probabilities
            ],
            "model_source": model.metadata.get(
                "model_source",
                model.metadata.get("data_source", "unknown"),
            ),
        },
        index=features.index,
    )


def _local_artifact_path(path: str | Path) -> Path:
    if "://" in str(path):
        raise ValueError(
            "Use a trusted local model path; "
            "remote artifacts are not loaded."
        )

    destination = Path(path).expanduser()
    if destination.suffix.lower() not in {".pkl", ".joblib"}:
        raise ValueError(
            "Model artifacts must have a .pkl "
            "or .joblib extension."
        )
    return destination


def save_model(
    bundle: ModelBundle,
    path: str | Path,
) -> None:
    if not isinstance(bundle, ModelBundle):
        raise ValueError(
            "save_model requires a trained ModelBundle."
        )

    destination = _local_artifact_path(path)
    destination.parent.mkdir(
        parents=True, exist_ok=True
    )
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            suffix=".pkl",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        joblib.dump(bundle, temporary_path, compress=3)
        os.replace(temporary_path, destination)
    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def load_model(path: str | Path) -> ModelBundle:
    artifact = _local_artifact_path(path)

    if not artifact.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {artifact}. "
            "Run scripts/train.py first."
        )

    try:
        bundle = joblib.load(artifact)
    except Exception as exc:
        raise ValueError(
            f"Cannot load model artifact {artifact}. "
            "Retrain it using the current environment."
        ) from exc

    if (
        not isinstance(bundle, ModelBundle)
        or not isinstance(bundle.pipeline, Pipeline)
    ):
        raise ValueError(
            "Artifact is not an Aura ModelBundle. "
            "Train a new model with scripts/train.py."
        )

    if (
        not isinstance(bundle.metadata, dict)
        or not isinstance(bundle.metrics, dict)
        or not isinstance(bundle.comparisons, list)
    ):
        raise ValueError(
            "Artifact metadata is invalid. "
            "Retrain using scripts/train.py."
        )

    if (
        bundle.metadata.get("features")
        != MODEL_FEATURE_COLUMNS
    ):
        raise ValueError(
            "Artifact uses a different feature schema. "
            "Retrain with the current project."
        )

    if (
        bundle.metadata.get("sklearn_version")
        != sklearn.__version__
    ):
        raise ValueError(
            "Artifact was trained with a different "
            "scikit-learn version. "
            "Retrain in this environment."
        )

    if (
        not bundle.metadata.get("data_source")
        or not hasattr(bundle.pipeline, "classes_")
    ):
        raise ValueError(
            "Artifact lacks fitted state or provenance. "
            "Retrain using scripts/train.py."
        )

    return bundle