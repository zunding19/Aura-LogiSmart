"""Build the reproducible, explicitly simulated dataset."""
import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_dataset import build
from src.data_loader import read_raw_shipments
from src.schema import TARGET_COLUMN


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "data/raw/global_supply_chain_pharma.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "data/processed/aura_shipments.csv"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args(argv)

    try:
        raw = read_raw_shipments(
            args.input
        )

        prepared = build(
            raw,
            seed=args.seed,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        prepared.to_csv(
            args.output,
            index=False,
            encoding="utf-8",
        )

    except (ValueError, OSError) as exc:
        print(
            f"Dataset preparation stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    failures = int(
        prepared[TARGET_COLUMN].sum()
    )

    print(
        f"Synthetic simulation: {len(prepared)} rows; "
        f"failures={failures} "
        f"({failures / len(prepared):.1%}); "
        f"seed={args.seed}"
    )
    print(
        f"Input: {args.input.resolve()}\n"
        f"Output: {args.output.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())