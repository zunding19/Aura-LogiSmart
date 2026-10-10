"""Check and evaluate Notebook 05's saved model; this command never trains."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import read_raw_shipments
from src.model import evaluate_model, load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/aura_shipments.csv",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/aura_final_model.pkl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON evaluation report.",
    )

    args = parser.parse_args(argv)

    try:
        if (
            args.report
            and args.report.resolve()
            in {
                args.model.resolve(),
                args.data.resolve(),
            }
        ):
            raise ValueError(
                "The report must not overwrite the dataset or model."
            )

        model = load_model(args.model)
        report = evaluate_model(
            read_raw_shipments(args.data),
            model,
        )
        rendered = json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )

        if args.report:
            args.report.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            args.report.write_text(
                rendered + "\n",
                encoding="utf-8",
            )

    except (ValueError, OSError) as exc:
        print(
            f"Model verification stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Using Notebook 05 artifact: {args.model.resolve()}")

    summary = {
        key: value
        for key, value in report.items()
        if key not in {
            "train_positions",
            "test_positions",
            "classification_report",
            "primary_metrics",
            "fallback_metrics",
        }
    }

    summary["fallback_metrics"] = {
        key: value
        for key, value in report["fallback_metrics"].items()
        if key != "classification_report"
    }

    print(
        json.dumps(
            summary,
            indent=2,
            allow_nan=False,
        )
    )
    print(
        "No model was trained or replaced. "
        "Retraining belongs in Notebook 05."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())