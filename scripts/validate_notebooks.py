"""Validate five notebook files; optionally execute disposable copies."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Run copies; notebook training cannot "
            "replace the deployed model."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Save executed notebook copies here.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Maximum seconds per cell.",
    )

    args = parser.parse_args(argv)

    try:
        paths = sorted(
            (ROOT / "notebooks").glob("0[1-5]_*.ipynb")
        )

        if [p.name[:2] for p in paths] != [
            "01", "02", "03", "04", "05",
        ]:
            raise ValueError(
                "Expected exactly one notebook for each of 01 through 05."
            )

        if args.timeout <= 0:
            raise ValueError(
                "--timeout must be positive."
            )

        if args.output_dir:
            output = args.output_dir.resolve()
            notebook_dir = (ROOT / "notebooks").resolve()

            if (
                output == notebook_dir
                or notebook_dir in output.parents
                or output in notebook_dir.parents
            ):
                raise ValueError(
                    "--output-dir must be separate "
                    "from the source notebook directory."
                )

            if not args.execute:
                raise ValueError(
                    "--output-dir requires --execute."
                )

        checked = []

        for path in paths:
            notebook = nbformat.read(
                path,
                as_version=4,
            )
            nbformat.validate(notebook)

            code = [
                c
                for c in notebook.cells
                if c.cell_type == "code"
            ]

            if not code or any(
                not c.source.strip() for c in code
            ):
                raise ValueError(
                    f"{path.name}: expected non-empty code cells."
                )

            for index, cell in enumerate(code):
                compile(
                    cell.source,
                    f"{path.name}:code_cell_{index}",
                    "exec",
                )

            checked.append((path, notebook))
            print(
                f"VALID: {path.name}",
                flush=True,
            )

        if args.execute:
            with tempfile.TemporaryDirectory(
                prefix="aura-notebooks-"
            ) as temporary:
                sandbox = Path(temporary)

                for name in [
                    "notebooks", "data", "src", "models",
                ]:
                    shutil.copytree(
                        ROOT / name,
                        sandbox / name,
                    )

                for path, notebook in checked:
                    print(
                        f"Executing disposable copy: {path.name}",
                        flush=True,
                    )

                    client = NotebookClient(
                        notebook,
                        timeout=args.timeout,
                        kernel_name="python3",
                        resources={
                            "metadata": {
                                "path": str(sandbox / "notebooks")
                            }
                        },
                    )

                    client.km = client.create_kernel_manager()
                    client.km.kernel_spec.argv[0] = sys.executable
                    client.execute()

                    if any(
                        c.execution_count is None
                        for c in notebook.cells
                        if c.cell_type == "code"
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

        print(
            "All five notebooks executed in disposable copies."
            if args.execute
            else (
                "All five notebooks passed schema and syntax validation; "
                "execution was not requested."
            )
        )

        return 0

    except (
        OSError,
        ValueError,
        SyntaxError,
        nbformat.ValidationError,
        CellExecutionError,
        CellTimeoutError,
    ) as exc:
        print(
            f"Notebook validation stopped: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())