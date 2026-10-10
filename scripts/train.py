"""Train using training CV and a separate holdout."""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    generate_demo_shipments,
    load_shipments,
    suggest_column_mapping,
)
from src.model import (
    save_model,
    train_model,
)
from src.schema import TARGET_COLUMN


def _column_mapping(value):
    if value is None:
        return None

    try:
        mapping = (
            json.loads(value)
            if value.lstrip().startswith("{")
            else json.loads(
                Path(value).read_text(
                    encoding="utf-8"
                )
            )
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "--column-mapping must be a JSON object "
            "or readable JSON file."
        ) from exc

    if (
        not isinstance(mapping, dict)
        or not all(
            isinstance(key, str)
            and isinstance(target, str)
            for key, target in mapping.items()
        )
    ):
        raise ValueError(
            "--column-mapping must map source column names "
            "to canonical names."
        )

    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    data = parser.add_mutually_exclusive_group(
        required=True
    )

    data.add_argument(
        "--data",
        type=Path,
        help=(
            "CSV containing binary outcome labels "
            "and data provenance."
        ),
    )

    data.add_argument(
        "--demo",
        action="store_true",
        help="Use the bundled synthetic shipment dataset.",
    )

    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
    )

    parser.add_argument(
        "--column-mapping",
        help=(
            "JSON old-to-canonical mapping "
            "or its file path."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Trusted local .pkl or .joblib output.",
    )

    parser.add_argument(
        "--random-state",
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help=(
            "Demo sample size; default is "
            "all 5,000 bundled shipments."
        ),
    )

    args = parser.parse_args(argv)

    try:
        mapping = _column_mapping(
            args.column_mapping
        )

        if args.demo:
            if (
                mapping is not None
                or args.target != TARGET_COLUMN
            ):
                raise ValueError(
                    "Custom target/mapping require --data."
                )

            frame = generate_demo_shipments(
                n=args.rows,
                seed=args.random_state,
            )

            target = TARGET_COLUMN
            source = (
                "Synthetic bundled shipment simulation"
            )

        else:
            frame = load_shipments(
                args.data,
                column_mapping=mapping,
            )

            target = (mapping or {}).get(
                args.target,
                args.target,
            )

            target = suggest_column_mapping(
                [target]
            ).get(
                target,
                target,
            )

            source = (
                "; ".join(
                    frame["Data_Source"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
                if "Data_Source" in frame
                else ""
            )

            source = source or (
                f"User CSV: {args.data}; "
                "label provenance unverified"
            )

        bundle = train_model(
            frame,
            target=target,
            data_source=source,
            random_state=args.random_state,
        )

        output = (
            args.output
            or ROOT / "models" / (
                "aura_demo_model.pkl"
                if bundle.metadata["is_synthetic"]
                else "aura_model.pkl"
            )
        )

        save_model(
            bundle, output
        )

    except (ValueError, OSError) as exc:
        print(
            f"Training stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Source: {bundle.metadata['model_source']}; "
        f"rows={len(frame)}; target={target}"
    )
    print(
        "Selected by training CV: "
        f"{bundle.metadata['model_name']}; "
        f"split={bundle.metadata['split_method']}"
    )

    for row in bundle.comparisons:
        print(
            f"CV {row['model']}: "
            f"ROC AUC={row['cv_roc_auc']:.3f}; "
            f"F1={row['cv_f1']:.3f}"
        )

    print(
        "Holdout:",
        {
            key: value
            for key, value in bundle.metrics.items()
            if key != "classification_report"
        },
    )

    for label, row in bundle.metrics[
        "classification_report"
    ].items():
        if isinstance(row, dict):
            print(
                f"  {label}: "
                f"precision={row['precision']:.3f}; "
                f"recall={row['recall']:.3f}; "
                f"F1={row['f1-score']:.3f}; "
                f"n={row['support']}"
            )

    print(
        "Scores are not calibrated; simulated outcomes "
        "do not establish real-world performance."
    )
    print(
        f"Saved: {output.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())