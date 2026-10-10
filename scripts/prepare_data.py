"""Prepare recorded data and optional display-only context without relabelling."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_dataset import build
from src.clean_dataset import clean
from src.data_loader import read_raw_shipments
from src.schema import TARGET_COLUMN, FLAG_COLUMNS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/raw/global_supply_chain_risk_2026.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/aura_shipments.csv",
    )
    parser.add_argument(
        "--clean-output",
        type=Path,
        default=ROOT / "data/processed/supply_chain_risk_clean.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        destinations = [
            args.input.resolve(),
            args.output.resolve(),
            args.clean_output.resolve(),
        ]
        if len(set(destinations)) != 3:
            raise ValueError(
                "Input, cleaned output and context output must be "
                "three different paths."
            )

        recorded = clean(read_raw_shipments(args.input))
        prepared = build(recorded, seed=args.seed)

        for path, frame in [
            (args.clean_output, recorded),
            (args.output, prepared),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(
                path,
                index=False,
                encoding="utf-8",
                date_format="%Y-%m-%d",
            )

    except (ValueError, OSError) as exc:
        print(
            f"Dataset preparation stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Preserved {len(prepared)} records; supplied disruptions: "
        f"{int(prepared[TARGET_COLUMN].sum())} "
        f"({prepared[TARGET_COLUMN].mean():.2%})."
    )
    print({
        column: int(prepared[column].sum())
        for column in FLAG_COLUMNS + ["Needs_Review"]
    })
    print(
        "Label authenticity unverified; simulated context is excluded "
        "from model features."
    )
    print(
        f"Recorded output: {args.clean_output.resolve()}\n"
        f"Context output: {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())