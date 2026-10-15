"""Convert a percent-format .py (# %% cells) into a .ipynb.

    python scripts/to_ipynb.py kaggle/fraud_detection_kaggle.py

Avoids a jupytext dependency; the format is simple enough to parse directly.
"""
import json
import sys
from pathlib import Path


def parse(text):
    cells, kind, buf = [], "code", []

    def flush():
        src = "\n".join(buf).strip("\n")
        if src.strip():
            cells.append((kind, src))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def to_nb(cells):
    out = []
    for kind, src in cells:
        if kind == "markdown":
            # strip the leading "# " that percent format uses for prose
            src = "\n".join(l[2:] if l.startswith("# ") else
                            ("" if l.strip() == "#" else l)
                            for l in src.splitlines()).strip("\n")
            out.append({"cell_type": "markdown", "metadata": {},
                        "source": src.splitlines(keepends=True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": src.splitlines(keepends=True)})
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    src = Path(sys.argv[1])
    dst = src.with_suffix(".ipynb")
    cells = parse(src.read_text(encoding="utf-8"))
    dst.write_text(json.dumps(to_nb(cells), indent=1), encoding="utf-8")

    n_md = sum(1 for k, _ in cells if k == "markdown")
    print(f"{dst}  ({len(cells)} cells: {n_md} markdown, {len(cells)-n_md} code)")


if __name__ == "__main__":
    main()
