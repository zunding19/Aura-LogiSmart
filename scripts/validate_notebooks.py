"""Validate and execute all five notebooks using the active Python."""
import argparse
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import (
    CellExecutionError,
    CellTimeoutError,
)
from jupyter_client.kernelspec import (
    NoSuchKernel,
)


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Save executed copies outside "
            "the source notebook directory."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Maximum seconds per code cell.",
    )

    args = parser.parse_args(argv)

    try:
        paths = sorted(
            (ROOT / "notebooks").glob(
                "0[1-5]_*.ipynb"
            )
        )

        if (
            len(paths) != 5
            or [
                path.name[:2]
                for path in paths
            ] != [
                "01", "02", "03", "04", "05"
            ]
        ):
            raise ValueError(
                "Expected one workflow notebook "
                "for each of 01 through 05."
            )

        if args.timeout <= 0:
            raise ValueError(
                "--timeout must be positive."
            )

        if (
            args.output_dir
            and args.output_dir.resolve()
            == (ROOT / "notebooks").resolve()
        ):
            raise ValueError(
                "--output-dir must differ from "
                "the source notebook directory."
            )

        checked = []

        for path in paths:
            try:
                notebook = nbformat.read(
                    path,
                    as_version=4,
                )
                nbformat.validate(
                    notebook
                )
            except (
                ValueError,
                nbformat.ValidationError,
            ) as exc:
                raise ValueError(
                    f"{path.name}: invalid notebook "
                    f"JSON/schema: {exc}"
                ) from exc

            code = [
                cell
                for cell in notebook.cells
                if cell.cell_type == "code"
            ]

            if (
                not code
                or any(
                    not cell.source.strip()
                    for cell in code
                )
            ):
                raise ValueError(
                    f"{path.name}: add non-empty executable "
                    "code cells before validation."
                )

            checked.append(
                (path, notebook)
            )

        for path, notebook in checked:
            print(
                f"Executing {path.name}",
                flush=True,
            )

            client = NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name="python3",
                resources={
                    "metadata": {
                        "path": str(
                            ROOT / "notebooks"
                        )
                    }
                },
            )

            client.km = (
                client.create_kernel_manager()
            )
            client.km.kernel_spec.argv[0] = (
                sys.executable
            )

            client.execute()

            if any(
                cell.execution_count is None
                for cell in notebook.cells
                if cell.cell_type == "code"
            ):
                raise ValueError(
                    f"{path.name}: not all code cells executed."
                )

            if args.output_dir:
                args.output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                nbformat.write(
                    notebook,
                    args.output_dir / path.name,
                )

            print(
                f"PASS: {path.name}",
                flush=True,
            )

    except (
        OSError,
        ValueError,
        nbformat.ValidationError,
        CellExecutionError,
        CellTimeoutError,
        NoSuchKernel,
    ) as exc:
        print(
            f"Notebook validation stopped: {exc}",
            file=sys.stderr,
        )
        return 2

    print(
        "All five notebooks executed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())