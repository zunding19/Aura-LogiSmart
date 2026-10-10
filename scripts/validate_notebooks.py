from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute all five Aura notebooks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for executed copies.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Maximum seconds per notebook cell.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    notebooks = sorted(
        (root / "notebooks").glob("0[1-5]_*.ipynb")
    )

    if len(notebooks) != 5:
        raise RuntimeError(
            "Expected all five Aura workflow notebooks."
        )

    if args.output_dir is not None:
        args.output_dir.mkdir(
            parents=True, exist_ok=True
        )

    for path in notebooks:
        notebook = nbformat.read(
            path, as_version=4
        )
        nbformat.validate(notebook)

        print(
            f"Executing {path.name}", flush=True
        )

        NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={
                "metadata": {
                    "path": str(root / "notebooks")
                }
            },
        ).execute()

        code_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code"
        ]
        if not code_cells or any(
            cell.execution_count is None
            for cell in code_cells
        ):
            raise RuntimeError(
                f"{path.name}: not all code cells executed."
            )

        print(
            f"PASS: {path.name} "
            f"({len(code_cells)} code cells)",
            flush=True,
        )

        if args.output_dir is not None:
            nbformat.write(
                notebook,
                args.output_dir / path.name,
            )

    print(
        "All five notebooks executed successfully.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())