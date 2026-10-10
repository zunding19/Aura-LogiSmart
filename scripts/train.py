from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)

from src.data_loader import (
    generate_demo_shipments,
    load_shipments,
    suggest_column_mapping,
)
from src.model import save_model, train_model
from src.schema import TARGET_COLUMN


def _column_mapping(
    value: str | None,
) -> dict[str, str] | None:
    if value is None:
        return None

    try:
        if value.lstrip().startswith("{"):
            mapping = json.loads(value)
        else:
            with Path(value).open(
                encoding="utf-8"
            ) as handle:
                mapping = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "--column-mapping must be a JSON object "
            "or a readable JSON file."
        ) from exc

    if (
        not isinstance(mapping, dict)
        or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in mapping.items()
        )
    ):
        raise ValueError(
            "--column-mapping must map original column names "
            "to canonical names."
        )

    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train an Aura shipment-risk model."
    )

    data = parser.add_mutually_exclusive_group(
        required=True
    )
    data.add_argument(
        "--data",
        type=Path,
        help="CSV containing genuinely observed binary outcomes.",
    )
    data.add_argument(
        "--demo",
        action="store_true",
        help="Explicitly train on synthetic demonstration data.",
    )

    parser.add_argument(
        "--target",
        default=TARGET_COLUMN,
        help="Observed binary target column in your CSV.",
    )
    parser.add_argument(
        "--column-mapping",
        help="JSON old-to-canonical map or JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Trusted local output .pkl or .joblib path.",
    )
    parser.add_argument(
        "--random-state", type=int, default=42
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=400,
        help="Synthetic row count, used only with --demo.",
    )

    args = parser.parse_args(argv)

    try:
        mapping = _column_mapping(args.column_mapping)

        if args.demo:
            if mapping or args.target != TARGET_COLUMN:
                raise ValueError(
                    "--column-mapping and custom --target "
                    "are for --data CSV, not --demo."
                )

            frame = generate_demo_shipments(
                n=args.rows,
                seed=args.random_state,
            )
            source = (
                f"synthetic demo (seed={args.random_state})"
            )
            target = TARGET_COLUMN
            print(
                "DEMO: synthetic outcomes demonstrate software "
                "behavior; metrics are not real-world evidence."
            )
        else:
            frame = load_shipments(
                args.data, column_mapping=mapping
            )
            source = f"user CSV: {args.data.resolve()}"
            target = (mapping or {}).get(
                args.target, args.target
            )

            if target not in frame:
                target = suggest_column_mapping(
                    [target]
                ).get(target, target)

        print(f"Data source: {source}")
        print(
            f"Rows: {len(frame)}; observed target: {target}"
        )

        bundle = train_model(
            frame,
            target=target,
            data_source=source,
            random_state=args.random_state,
        )

        output = args.output or Path(
            "models/aura_demo_model.pkl"
            if bundle.metadata["is_synthetic"]
            else "models/aura_model.pkl"
        )
        save_model(bundle, output)

    except (
        ValueError, FileNotFoundError, OSError
    ) as exc:
        print(
            f"Training stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        "Selected using training CV: "
        + bundle.metadata["model_name"]
    )
    print(
        "Model provenance: "
        + bundle.metadata["model_source"]
    )
    print(
        f"Split: {bundle.metadata['split_method']}; "
        f"training={bundle.metadata['n_train']}, "
        f"holdout={bundle.metadata['n_test']}"
    )

    print("Training-CV comparisons:")
    print(json.dumps(bundle.comparisons, indent=2))
    print("Untouched holdout metrics:")
    print(json.dumps(bundle.metrics, indent=2))
    print(
        "Probability calibration and a real-world "
        "prediction horizon have not been established."
    )
    print(
        f"Saved trusted local artifact: {output.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())