"""Train a dated disruption model; supplied labels have unverified authenticity."""
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
from src.model import save_model, train_model
from src.schema import TARGET_COLUMN


def _column_mapping(value):
    if value is None:
        return None

    try:
        mapping = json.loads(
            value
            if value.lstrip().startswith("{")
            else Path(value).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(
            "Column mapping must be a JSON object or readable JSON file."
        ) from exc

    if (
        not isinstance(mapping, dict)
        or not all(
            isinstance(key, str) and isinstance(target, str)
            for key, target in mapping.items()
        )
    ):
        raise ValueError(
            "Column mapping must map source headers to canonical headers."
        )

    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()

    data.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/aura_shipments.csv",
    )
    data.add_argument(
        "--demo",
        action="store_true",
        help="Sample bundled supplied records; does not invent outcome labels.",
    )

    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--column-mapping")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models/aura_disruption_model.pkl",
    )
    parser.add_argument(
        "--seed",
        "--random-state",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=5000,
        help="Sample size for --demo.",
    )
    args = parser.parse_args(argv)

    try:
        mapping = _column_mapping(args.column_mapping)

        if args.demo and (
            mapping is not None or args.target != TARGET_COLUMN
        ):
            raise ValueError(
                "Custom mapping/target requires --data."
            )

        frame = (
            generate_demo_shipments(n=args.rows, seed=args.seed)
            if args.demo
            else load_shipments(
                args.data,
                column_mapping=mapping,
            )
        )

        target = suggest_column_mapping(
            [args.target]
        ).get(args.target, args.target)
        target = (mapping or {}).get(args.target, target)

        source = (
            "; ".join(
                frame["Data_Source"].dropna().astype(str).unique()
            )
            if "Data_Source" in frame
            else "Supplied supply-chain records; authenticity unverified"
        )

        bundle = train_model(
            frame,
            target=target,
            data_source=source,
            random_state=args.seed,
        )
        save_model(bundle, args.output)

    except (ValueError, OSError) as exc:
        print(
            f"Training stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Target: {TARGET_COLUMN}; "
        f"source: {bundle.metadata['model_source']}"
    )
    print(
        "Selected by temporal training CV: "
        f"{bundle.metadata['model_name']}; "
        f"holdout starts {bundle.metadata['holdout_start']}"
    )
    print("CV:", json.dumps(bundle.comparisons, indent=2))
    print("Holdout:", json.dumps(bundle.metrics, indent=2))
    print(
        "Label authenticity and Date timing are unverified; scores "
        "are not calibrated and no clinical outcome is predicted."
    )
    print(
        f"Saved trusted local artifact: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())