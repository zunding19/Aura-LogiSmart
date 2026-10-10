"""Load and score Notebook 05's fitted models; never train or replace them."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import pickletools
import warnings
import zlib

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .features import extract_features
from .schema import MODEL_FEATURE_COLUMNS, TARGET_COLUMN

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models/aura_final_model.pkl"
)

FALLBACK_FEATURES = [
    "Geopolitical_Risk_Score",
    "Carrier_Reliability_Score",
    "Lead_Time_Days",
    "Distance_km",
    "pace_h_per_1000km",
    "pace_vs_typical",
    "Transport_Mode",
]

SCORE_NAMES = {
    "Precision": "precision",
    "Recall": "recall",
    "F1 score": "f1",
    "ROC AUC": "roc_auc",
    "Average precision": "average_precision",
}


@dataclass
class NotebookModel:
    pipeline: Pipeline
    fallback_pipeline: Pipeline
    package: dict
    metadata: dict
    metrics: dict
    comparisons: list = field(default_factory=list)

    @property
    def fallback_features(self):
        return list(self.package["fallback_features"])


def risk_level(probability):
    try:
        p = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Risk probability must be a number from 0 to 1."
        ) from exc

    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(
            "Risk probability must be finite and from 0 to 1."
        )

    # Notebook pd.cut(right=False): a lower edge belongs to the next tier.
    return (
        "Low" if p < .3
        else "Medium" if p < .6
        else "High" if p < .8
        else "Critical"
    )


def _stored_version(path):
    """Read the first sklearn version string without executing pickle objects."""
    data = path.read_bytes()

    if len(data) > 16 * 1024 * 1024:
        raise ValueError(
            "Notebook model is unexpectedly large; inspect it before loading."
        )

    if not data.startswith(b"\x80"):
        try:
            decoder = zlib.decompressobj()
            data = decoder.decompress(data, 16 * 1024 * 1024 + 1)

            if (
                not decoder.eof
                or decoder.unused_data
                or len(data) > 16 * 1024 * 1024
            ):
                raise ValueError(
                    "Invalid or oversized compressed notebook model."
                )
        except zlib.error as exc:
            raise ValueError(
                "Expected Notebook 05's uncompressed or zlib joblib file."
            ) from exc

    found = False

    try:
        for opcode, value, _ in pickletools.genops(data):
            if opcode.name in {
                "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
            }:
                if found:
                    return value
                found = value == "_sklearn_version"
    except (ValueError, UnicodeError) as exc:
        raise ValueError(
            "Cannot inspect the notebook model's sklearn version."
        ) from exc

    raise ValueError(
        "Notebook model has no inspectable sklearn version."
    )


def _validate_package(package):
    if not isinstance(package, dict):
        raise ValueError(
            "Load Notebook 05's dictionary, not an older ModelBundle artifact."
        )

    required = {
        "pipeline",
        "fallback_pipeline",
        "features",
        "numeric_features",
        "categorical_features",
        "fallback_features",
        "target",
        "weather_severity",
        "typical_pace",
        "high_risk_score",
        "low_reliability",
        "alert_line",
        "tier_edges",
        "tier_names",
        "required_any_of",
        "test_scores",
        "trained_on_shipments",
        "tested_on_shipments",
        "sklearn_version",
    }

    if required - package.keys():
        raise ValueError(
            "Incomplete notebook model package: "
            f"{sorted(required - package.keys())}."
        )

    if package["sklearn_version"] != sklearn.__version__:
        raise ValueError(
            f"Notebook model needs sklearn {package['sklearn_version']}; "
            f"installed {sklearn.__version__}."
        )

    if (
        package["target"] != TARGET_COLUMN
        or package["features"] != list(MODEL_FEATURE_COLUMNS)
        or package["fallback_features"] != FALLBACK_FEATURES
        or package["numeric_features"] != list(MODEL_FEATURE_COLUMNS[:-2])
        or package["categorical_features"]
        != ["Weather_Condition", "Transport_Mode"]
    ):
        raise ValueError(
            "Notebook target or feature order differs "
            "from this backend contract."
        )

    expected = {
        "weather_severity": {
            "Clear": 0,
            "Rain": 1,
            "Fog": 2,
            "Storm": 3,
            "Hurricane": 4,
        },
        "typical_pace": {
            "Air": 2.35,
            "Rail": 20.9,
            "Road": 17.9,
            "Sea": 41.5,
        },
        "high_risk_score": 6.,
        "low_reliability": .7,
        "alert_line": .6,
        "tier_edges": [0, .3, .6, .8, 1.0001],
        "required_any_of": [
            ["Weather_Condition"],
            ["Lead_Time_Days", "Distance_km", "Transport_Mode"],
        ],
    }

    if any(package[key] != value for key, value in expected.items()):
        raise ValueError(
            "Notebook formulas, thresholds or minimum-input rules "
            "changed; update the adapter."
        )

    if (
        len(package["tier_names"]) != 4
        or set(package["test_scores"]) != set(SCORE_NAMES)
    ):
        raise ValueError(
            "Notebook tier labels or reported scorecard are incomplete."
        )

    if any(
        not np.isfinite(float(score))
        or not 0 <= float(score) <= 1
        for score in package["test_scores"].values()
    ):
        raise ValueError(
            "Notebook reported scores must be finite values from 0 to 1."
        )

    for key in ["trained_on_shipments", "tested_on_shipments"]:
        if not isinstance(package[key], int) or package[key] <= 0:
            raise ValueError(
                "Notebook training/test row counts must be positive integers."
            )

    contracts = [
        (
            "pipeline",
            package["features"],
            package["numeric_features"],
            package["categorical_features"],
        ),
        (
            "fallback_pipeline",
            FALLBACK_FEATURES,
            FALLBACK_FEATURES[:-1],
            ["Transport_Mode"],
        ),
    ]

    for key, columns, numeric, categorical in contracts:
        pipeline = package[key]

        if (
            not isinstance(pipeline, Pipeline)
            or list(pipeline.named_steps) != ["prepare", "model"]
            or not isinstance(
                pipeline.named_steps["prepare"], ColumnTransformer
            )
            or not isinstance(
                pipeline.named_steps["model"], LogisticRegression
            )
        ):
            raise ValueError(
                f"{key} must be Notebook 05's fitted "
                "preparation/logistic-regression pipeline."
            )

        check_is_fitted(pipeline.named_steps["model"])

        if (
            list(pipeline.feature_names_in_) != columns
            or set(pipeline.classes_) != {0, 1}
            or len(pipeline.classes_) != 2
        ):
            raise ValueError(
                f"{key} has incompatible fitted feature names "
                "or binary disruption classes."
            )

        transforms = {
            name: (step, cols)
            for name, step, cols
            in pipeline.named_steps["prepare"].transformers_
        }

        if any(
            name not in transforms
            or list(transforms[name][1]) != cols
            for name, cols in [
                ("numeric", numeric),
                ("categorical", categorical),
            ]
        ):
            raise ValueError(
                f"{key} fitted preprocessing columns "
                "do not match its feature lists."
            )


def load_model(path=DEFAULT_MODEL_PATH):
    """Only use trusted, reviewed local files: joblib can execute pickle constructors."""
    artifact = Path(path).expanduser()

    if (
        "://" in str(path)
        or artifact.suffix.lower() not in {".pkl", ".joblib"}
    ):
        raise ValueError(
            "Supply a reviewed local Notebook 05 .pkl or .joblib path."
        )

    if not artifact.is_file():
        raise FileNotFoundError(
            f"Missing notebook model {artifact}; "
            "copy models/aura_final_model.pkl from main."
        )

    version = _stored_version(artifact)

    if version != sklearn.__version__:
        raise ValueError(
            f"Notebook model needs sklearn {version}; "
            f"installed {sklearn.__version__}. "
            "Install matching dependencies."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            package = joblib.load(artifact)

        _validate_package(package)
    except Exception as exc:
        raise ValueError(
            f"Cannot use notebook model {artifact}: {exc}"
        ) from exc

    source = (
        "Notebook 05's supplied fitted model; "
        "label authenticity unverified"
    )

    metadata = {
        "model_name": "Logistic Regression",
        "model_source": source,
        "data_source": source,
        "artifact_path": str(artifact.resolve()),
        "artifact_sha256": hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest(),
        "feature_schema_version": "aura-notebook-05-v1",
        "features": list(package["features"]),
        "fallback_features": list(FALLBACK_FEATURES),
        "target_column": TARGET_COLUMN,
        "target_semantics": (
            "Supplied disruption label; not spoilage or clinical failure"
        ),
        "decision_threshold": package["alert_line"],
        "alert_line": package["alert_line"],
        "tier_edges": list(package["tier_edges"]),
        "tier_names": list(package["tier_names"]),
        "weather_severity": dict(package["weather_severity"]),
        "typical_pace": dict(package["typical_pace"]),
        "high_risk_score": package["high_risk_score"],
        "low_reliability": package["low_reliability"],
        "sklearn_version": package["sklearn_version"],
        "train_rows": package["trained_on_shipments"],
        "holdout_rows": package["tested_on_shipments"],
        "label_provenance": "unverified",
        "is_synthetic": None,
        "calibrated": False,
        "calibration_note": (
            "Calibration has not been independently established."
        ),
        "split_method": (
            "Notebook stratified random 80/20 split, seed 42; "
            "not chronological"
        ),
        "evaluation_note": (
            "Reported scores require reproduction; "
            "full-data feature exploration limits independence."
        ),
        "prediction_horizon": "Not established",
        "planned_lead_time_confirmed": True,
    }

    metrics = {
        SCORE_NAMES[key]: float(value)
        for key, value in package["test_scores"].items()
    }
    metrics.update(
        reported_test_scores=dict(package["test_scores"]),
        evaluation_status="artifact_reported_not_recomputed",
    )

    return NotebookModel(
        package["pipeline"],
        package["fallback_pipeline"],
        package,
        metadata,
        metrics,
    )


def _features_for_model(df, model):
    if not isinstance(model, NotebookModel):
        raise ValueError(
            "Load Notebook 05's model with load_model first; "
            "no fallback training is available."
        )

    _validate_package(model.package)

    if (
        not isinstance(df, pd.DataFrame)
        or df.columns.duplicated().any()
    ):
        raise ValueError(
            "Supply a shipment DataFrame with unique column names."
        )

    features = extract_features(df, config=model.package)

    if list(features.columns) != model.package["features"]:
        raise ValueError(
            "extract_features must return the exact "
            "ordered Notebook 05 feature list."
        )

    if np.isinf(
        features[model.package["numeric_features"]].to_numpy(dtype=float)
    ).any():
        raise ValueError(
            "Model features contain infinity; "
            "check distance and lead-time units."
        )

    return features


def _probability(pipeline, features):
    if features.empty:
        return np.array([], dtype=float)

    p = pipeline.predict_proba(features)[
        :, list(pipeline.classes_).index(1)
    ]

    if (
        not np.isfinite(p).all()
        or ((p < 0) | (p > 1)).any()
    ):
        raise ValueError(
            "Notebook model returned invalid disruption probabilities."
        )

    return p


def predict_failure_risk(features_df, model):
    """Rebuild features; the notebook's own no-weather model handles unknown weather."""
    features = _features_for_model(features_df, model)
    known_weather = features["weather_severity"].notna().to_numpy()

    complete_trip = (
        features[["Lead_Time_Days", "Distance_km"]].notna().all(axis=1)
        & features["Transport_Mode"].isin(model.package["typical_pace"])
    ).to_numpy()

    invalid = np.flatnonzero(~known_weather & ~complete_trip)

    if len(invalid):
        indices = features.index.take(invalid).tolist()
        raise ValueError(
            "Insufficient data for reliable prediction "
            f"at rows {indices}: supply recognized weather "
            "or valid Lead_Time_Days, Distance_km and Transport_Mode."
        )

    p = np.empty(len(features), dtype=float)

    p[known_weather] = _probability(
        model.pipeline,
        features.iloc[
            np.flatnonzero(known_weather)
        ][model.package["features"]],
    )

    p[~known_weather] = _probability(
        model.fallback_pipeline,
        features.iloc[
            np.flatnonzero(~known_weather)
        ][FALLBACK_FEATURES],
    )

    tier = np.searchsorted(
        model.package["tier_edges"][1:-1],
        p,
        side="right",
    )

    return pd.DataFrame(
        {
            "p_disruption": p,
            "p_failure": p,
            "risk_level": [risk_level(value) for value in p],
            "risk_tier": [
                model.package["tier_names"][i] for i in tier
            ],
            "is_alert": p >= model.package["alert_line"],
            "target_column": TARGET_COLUMN,
            "model_source": model.metadata["model_source"],
            "model_variant": np.where(
                known_weather, "primary", "weather_missing"
            ),
        },
        index=features.index,
    )


def dataset_fingerprint(df):
    columns = [
        column
        for column in [
            "Shipment_ID",
            "Weather_Condition",
            "Geopolitical_Risk_Score",
            "Carrier_Reliability_Score",
            "Lead_Time_Days",
            "Distance_km",
            "Transport_Mode",
            TARGET_COLUMN,
        ]
        if column in df
    ]

    return hashlib.sha256(
        pd.util.hash_pandas_object(
            df[columns], index=False
        ).to_numpy().tobytes()
    ).hexdigest()


def _score(labels, probabilities, threshold):
    predicted = probabilities >= threshold

    return {
        "precision": float(
            precision_score(labels, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(labels, predicted, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "accuracy": float(
            accuracy_score(labels, predicted)
        ),
        "brier": float(
            brier_score_loss(labels, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predicted,
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_model(df, model=None):
    """Reproduce Notebook 05's split/scores on the original ordered CSV; never fit."""
    model = load_model() if model is None else model
    features = _features_for_model(df, model)

    if TARGET_COLUMN not in df:
        raise ValueError(
            f"Evaluation requires {TARGET_COLUMN} labels."
        )

    labels = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).to_numpy()

    if pd.isna(labels).any() or set(labels) != {0, 1}:
        raise ValueError(
            "Evaluation requires both binary disruption outcomes, "
            "without missing labels."
        )

    expected = (
        model.package["trained_on_shipments"]
        + model.package["tested_on_shipments"]
    )

    if len(df) != expected:
        raise ValueError(
            "Notebook score reproduction requires "
            f"the original {expected} ordered rows, got {len(df)}."
        )

    if (
        "Shipment_ID" in df
        and df["Shipment_ID"].duplicated().any()
    ):
        raise ValueError(
            "Duplicate shipment IDs prevent reproducing "
            "the notebook's independent row split."
        )

    train, test = train_test_split(
        np.arange(len(df)),
        test_size=.2,
        stratify=labels,
        random_state=42,
    )

    if (
        len(train) != model.package["trained_on_shipments"]
        or len(test) != model.package["tested_on_shipments"]
    ):
        raise ValueError(
            "Reconstructed notebook split does not "
            "match stored training/test counts."
        )

    if features.iloc[test]["weather_severity"].isna().any():
        raise ValueError(
            "Original notebook test-score reproduction requires "
            "recognized weather on every test row."
        )

    primary = _score(
        labels[test],
        _probability(
            model.pipeline,
            features.iloc[test][model.package["features"]],
        ),
        .6,
    )
    fallback = _score(
        labels[test],
        _probability(
            model.fallback_pipeline,
            features.iloc[test][FALLBACK_FEATURES],
        ),
        .6,
    )

    matches = {
        key: round(primary[SCORE_NAMES[key]], 4) == float(value)
        for key, value in model.package["test_scores"].items()
    }

    return {
        **primary,
        "primary_metrics": primary,
        "fallback_metrics": fallback,
        "reported_scores": dict(model.package["test_scores"]),
        "reported_scores_match": matches,
        "all_reported_scores_match": all(matches.values()),
        "n_train": len(train),
        "n_test": len(test),
        "train_positions": train.tolist(),
        "test_positions": test.tolist(),
        "decision_threshold": .6,
        "dataset_fingerprint": dataset_fingerprint(df),
        "target_column": TARGET_COLUMN,
        "evaluation_status": (
            "recomputed_on_notebook_split_without_refitting"
        ),
        "split_method": model.metadata["split_method"],
        "evaluation_note": (
            "Artifact has no training-data fingerprint; "
            "matching scores support reproduction, "
            "not proof of original data identity "
            "or independently selected features."
        ),
    }


def train_model(*args, **kwargs):
    raise ValueError(
        "Backend training is disabled. Use the fitted models "
        "from Notebook 05 and load_model()."
    )


def save_model(*args, **kwargs):
    raise ValueError(
        "Backend model replacement is disabled. "
        "Notebook 05 owns models/aura_final_model.pkl."
    )