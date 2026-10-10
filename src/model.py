"""Train models with separate training CV and held-out evaluation."""
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
    FEATURE_SCHEMA_VERSION,
    MODEL_CATEGORICAL_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_NUMERIC_COLUMNS,
    STORAGE_STANDARDS,
    TARGET_COLUMN,
    TRANSPORT_MODES,
)


ARTIFACT_FORMAT_VERSION = 2


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
            "Risk probability must be a number from 0 to 1."
        ) from exc

    if (
        not np.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError(
            "Risk probability must be finite and from 0 to 1."
        )

    if probability <= .3:
        return "Low"
    if probability <= .6:
        return "Medium"
    if probability <= .8:
        return "High"
    return "Critical"


def _checked_features(features: pd.DataFrame) -> pd.DataFrame:
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
            f"Missing model features: {missing}. "
            "Run extract_features first."
        )

    result = features.loc[
        :, MODEL_FEATURE_COLUMNS
    ].copy()

    for column in MODEL_NUMERIC_COLUMNS:
        try:
            result[column] = pd.to_numeric(
                result[column],
                errors="raise",
            ).astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature {column!r} must be numeric."
            ) from exc

        if np.isinf(
            result[column].to_numpy()
        ).any():
            raise ValueError(
                f"Feature {column!r} contains infinity; "
                "check input units."
            )

    for column, allowed in zip(
        MODEL_CATEGORICAL_COLUMNS,
        [TRANSPORT_MODES, STORAGE_STANDARDS],
    ):
        invalid = (
            result[column].notna()
            & ~result[column].isin(allowed)
        )

        if invalid.any():
            raise ValueError(
                f"Feature {column!r} must use "
                f"a canonical category: {allowed}."
            )

        result[column] = (
            result[column]
            .astype(object)
            .where(
                result[column].notna(),
                np.nan,
            )
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
            (
                "numeric",
                numeric,
                MODEL_NUMERIC_COLUMNS,
            ),
            (
                "categorical",
                categorical,
                MODEL_CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


def _class_group_counts(labels, groups):
    return [
        len(set(groups[labels == label]))
        for label in (0, 1)
    ]


def _cv_splits(labels, groups, grouped, random_state):
    support = (
        _class_group_counts(labels, groups)
        if grouped
        else np.bincount(labels, minlength=2)
    )

    for folds in range(
        min(5, int(min(support))),
        1,
        -1,
    ):
        splitter_class = (
            StratifiedGroupKFold
            if grouped
            else StratifiedKFold
        )

        splitter = splitter_class(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        splits = list(
            splitter.split(
                np.zeros(len(labels)),
                labels,
                groups if grouped else None,
            )
        )

        if all(
            len(np.unique(labels[train])) == 2
            and len(np.unique(labels[valid])) == 2
            for train, valid in splits
        ):
            return splits

    raise ValueError(
        "Cannot form CV with both outcomes. "
        "Add independent labeled shipments; "
        "do not duplicate rows."
    )


def _split_data(labels, groups, random_state):
    grouped = (
        len(set(groups)) < len(groups)
    )
    support = _class_group_counts(
        labels, groups
    )

    if min(support) < 3:
        raise ValueError(
            "Need at least 3 distinct shipment groups "
            "per outcome for holdout and CV; "
            f"found {support}."
        )

    if not grouped:
        rng = np.random.RandomState(
            random_state
        )
        train_parts = []
        test_parts = []

        for label in (0, 1):
            positions = rng.permutation(
                np.flatnonzero(labels == label)
            )

            reserved = max(
                1,
                min(
                    int(np.ceil(len(positions) * .2)),
                    len(positions) - 2,
                ),
            )

            test_parts.append(
                positions[:reserved]
            )
            train_parts.append(
                positions[reserved:]
            )

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

    for folds in range(
        min(5, min(support)),
        1,
        -1,
    ):
        splitter = StratifiedGroupKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )

        for train, test in splitter.split(
            np.zeros(len(labels)),
            labels,
            groups,
        ):
            if (
                len(np.unique(labels[test])) != 2
                or min(
                    _class_group_counts(
                        labels[train],
                        groups[train],
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
                abs(len(test) / len(labels) - .2),
                abs(
                    labels[test].mean()
                    - labels.mean()
                ),
            )

            candidates.append(
                (rank, train, test, cv)
            )

    if not candidates:
        raise ValueError(
            "Repeated IDs cannot form grouped holdout "
            "and CV with both outcomes. "
            "Add independent shipments."
        )

    _, train, test, cv = min(
        candidates,
        key=lambda item: item[0],
    )

    return train, test, cv, True


def _provenance(clean, data_source):
    marked = (
        clean["Shipment_ID"]
        .astype(str)
        .str.upper()
        .str.startswith("SYNTHETIC-")
        .to_numpy()
    )

    if "Data_Source" in clean:
        marked = marked | (
            clean["Data_Source"]
            .astype(str)
            .str.contains(
                r"synthetic|simulat",
                case=False,
                na=False,
            )
            .to_numpy()
        )

    if marked.any() and not marked.all():
        raise ValueError(
            "Synthetic and unmarked rows are mixed. "
            "Keep simulated and real outcome datasets separate."
        )

    synthetic = (
        bool(marked.any())
        or bool(
            re.search(
                r"\b(synthetic|simulated|simulation|demo)\b",
                data_source,
                re.I,
            )
        )
    )

    label = (
        data_source
        if (
            not synthetic
            or re.search(
                r"\bsynthetic\b",
                data_source,
                re.I,
            )
        )
        else f"synthetic data from {data_source}"
    )

    return synthetic, label


def _fingerprint(features, labels, groups):
    rows = features.assign(
        Shipment_ID=groups,
        **{TARGET_COLUMN: labels},
    )

    digest = hashlib.sha256(
        json.dumps(
            [
                FEATURE_SCHEMA_VERSION,
                *rows.columns,
            ]
        ).encode()
    )

    digest.update(
        pd.util.hash_pandas_object(
            rows,
            index=False,
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
    if (
        not isinstance(df, pd.DataFrame)
        or df.empty
    ):
        raise ValueError(
            "Training needs a nonempty shipment DataFrame "
            "with binary outcome labels."
        )

    if (
        not isinstance(data_source, str)
        or not data_source.strip()
    ):
        raise ValueError(
            "Supply a data_source description for provenance."
        )

    source = df.copy()

    if target != TARGET_COLUMN:
        if target not in source:
            raise ValueError(
                f"Target {target!r} is missing. "
                "Map the outcome column or set target."
            )

        source[TARGET_COLUMN] = source[target]
        source = source.drop(
            columns=[target]
        )

    clean = clean_raw_shipment_data(
        source,
        require_target=True,
    )

    features = _checked_features(
        extract_features(clean)
    )

    labels = clean[
        TARGET_COLUMN
    ].to_numpy(dtype=int)

    if set(labels) != {0, 1}:
        raise ValueError(
            "Training requires both outcome classes: "
            "non-failure (0) and failure (1)."
        )

    groups = clean[
        "Shipment_ID"
    ].astype(str).to_numpy()

    synthetic, source_label = _provenance(
        clean, data_source
    )

    train, test, cv, grouped = _split_data(
        labels,
        groups,
        random_state,
    )

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
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
    }

    comparisons = []
    pipelines = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            (
                "preprocessor",
                _preprocessor(),
            ),
            (
                "classifier",
                estimator,
            ),
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
            "cv_roc_auc_std": float(
                np.std(
                    scores["test_roc_auc"]
                )
            ),
        }

        row.update({
            f"cv_{metric}": float(
                np.mean(
                    scores[f"test_{metric}"]
                )
            )
            for metric in scoring
        })

        comparisons.append(row)
        pipelines[name] = pipeline

    selected = max(
        comparisons,
        key=lambda row: (
            row["cv_roc_auc"],
            row["cv_f1"],
        ),
    )

    pipeline = pipelines[
        selected["model"]
    ].fit(
        features.iloc[train],
        labels[train],
    )

    probabilities = pipeline.predict_proba(
        features.iloc[test]
    )[
        :,
        list(pipeline.classes_).index(1),
    ]

    predicted = (
        probabilities >= .5
    ).astype(int)

    actual = labels[test]

    metrics = {
        name: float(
            metric(
                actual,
                predicted,
                zero_division=0,
            )
        )
        for name, metric in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]
    }

    metrics.update(
        roc_auc=float(
            roc_auc_score(
                actual,
                probabilities,
            )
        ),
        accuracy=float(
            accuracy_score(
                actual,
                predicted,
            )
        ),
        confusion_matrix=confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).tolist(),
        n_test=int(len(test)),
        classification_report=classification_report(
            actual,
            predicted,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    )

    metadata = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_name": selected["model"],
        "model_source": source_label,
        "data_source": source_label,
        "source_description": data_source,
        "is_synthetic": synthetic,
        "label_source": (
            "synthetic labels (demonstration only)"
            if synthetic
            else "user-supplied outcome labels; provenance unverified"
        ),
        "target_column": target,
        "features": list(MODEL_FEATURE_COLUMNS),
        "n_rows": len(features),
        "n_train": len(train),
        "n_test": len(test),
        "train_rows": len(train),
        "selection_rows": len(train),
        "holdout_rows": len(test),
        "train_positions": train.tolist(),
        "test_positions": test.tolist(),
        "train_group_count": len(
            set(groups[train])
        ),
        "test_group_count": len(
            set(groups[test])
        ),
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
        "cv_folds": len(cv),
        "selection_metric": "roc_auc",
        "decision_threshold": .5,
        "holdout_fraction": (
            len(test) / len(features)
        ),
        "calibrated": False,
        "calibration_note": (
            "Probability calibration has not been assessed; "
            "outputs are classifier scores."
        ),
        "prediction_horizon": (
            "Not established; requires appropriately timed "
            "real outcome labels."
        ),
        "dataset_fingerprint": _fingerprint(
            features,
            labels,
            groups,
        ),
        "sklearn_version": sklearn.__version__,
        "random_state": int(random_state),
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    bundle = ModelBundle(
        pipeline,
        metadata,
        metrics,
        comparisons,
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
            "A valid trained Aura ModelBundle is required; "
            "train or load a trusted local model."
        )

    expected = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": MODEL_FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
    }

    if any(
        bundle.metadata.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError(
            "Model schema, formula contract, artifact format "
            "or sklearn version changed. Retrain this artifact."
        )

    if (
        not bundle.metadata.get("data_source")
        or not bundle.metadata.get("model_source")
        or not hasattr(
            bundle.pipeline, "classes_"
        )
        or set(
            bundle.pipeline.classes_
        ) != {0, 1}
    ):
        raise ValueError(
            "Model lacks fitted binary classes "
            "or provenance. Retrain it."
        )


def predict_failure_risk(features_df, model):
    _validate_bundle(model)

    features = _checked_features(
        features_df
    )

    try:
        probabilities = (
            model.pipeline.predict_proba(
                features
            )[
                :,
                list(
                    model.pipeline.classes_
                ).index(1),
            ]
            if len(features)
            else np.array(
                [], dtype=float
            )
        )
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Cannot predict these inputs. "
            "Check feature values or retrain the model."
        ) from exc

    if (
        not np.isfinite(probabilities).all()
        or (
            (probabilities < -1e-12)
            | (probabilities > 1 + 1e-12)
        ).any()
    ):
        raise ValueError(
            "Model returned invalid probabilities; "
            "retrain it."
        )

    probabilities = np.clip(
        probabilities, 0, 1
    )

    return pd.DataFrame(
        {
            "p_failure": probabilities,
            "risk_level": [
                risk_level(probability)
                for probability in probabilities
            ],
            "model_source": (
                model.metadata["model_source"]
            ),
        },
        index=features.index,
    )


def _local_artifact_path(path):
    if (
        "://" in str(path)
        or Path(path).suffix.lower()
        not in {".pkl", ".joblib"}
    ):
        raise ValueError(
            "Use a trusted local .pkl or .joblib artifact path; "
            "never an uploaded pickle."
        )

    return Path(path).expanduser()


def save_model(bundle, path):
    _validate_bundle(bundle)

    destination = _local_artifact_path(
        path
    )
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        suffix=".pkl",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

    try:
        joblib.dump(
            bundle,
            temporary,
            compress=3,
        )
        os.replace(
            temporary, destination
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def load_model(path):
    """Load only trusted local project artifacts."""
    artifact = _local_artifact_path(
        path
    )

    if not artifact.is_file():
        raise FileNotFoundError(
            f"Model artifact missing: {artifact}. "
            "Train a new model first."
        )

    try:
        bundle = joblib.load(
            artifact
        )
    except Exception as exc:
        raise ValueError(
            f"Cannot load {artifact}; "
            "retrain with this environment."
        ) from exc

    _validate_bundle(bundle)
    return bundle