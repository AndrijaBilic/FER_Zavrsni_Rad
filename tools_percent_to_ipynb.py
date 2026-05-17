import json
import sys
from pathlib import Path


def split_percent_cells(text: str):
    cells = []
    current = []
    current_type = "code"

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if current or not cells:
                cells.append((current_type, current))
            current = []
            current_type = "markdown" if "[markdown]" in stripped else "code"
            continue

        if current_type == "markdown":
            if line.startswith("# "):
                current.append(line[2:])
            elif line.startswith("#"):
                current.append(line[1:])
            else:
                current.append(line)
        else:
            current.append(line)

    cells.append((current_type, current))
    return cells


def convert(src: Path, dst: Path):
    text = src.read_text(encoding="utf-8")
    raw_cells = split_percent_cells(text)
    cells = []
    for cell_type, source in raw_cells:
        if not "".join(source).strip():
            continue
        if cell_type == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python tools_percent_to_ipynb.py source.py target.ipynb")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
