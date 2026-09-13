"""Remove generated outputs before committing notebooks to Git."""

import argparse
from pathlib import Path

import nbformat


def strip_outputs(path: Path) -> None:
    """Clear code-cell outputs and execution counters in place."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.notebooks:
        strip_outputs(path)
        print(f"Cleared outputs: {path}")


if __name__ == "__main__":
    main()
