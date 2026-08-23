"""Chạy các notebook chuẩn theo thứ tự và lưu lại output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    "01_EDA.ipynb",
    "02_Feature_Engineering.ipynb",
    "03_Model_Experiments.ipynb",
    "04_Model_Interpretation.ipynb",
    "05_Business_Report.ipynb",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", choices=NAMES)
    args = parser.parse_args()
    for name in args.notebooks or NAMES:
        path = ROOT / "notebooks" / name
        notebook = nbformat.read(path, as_version=4)
        print(f"Đang chạy {name}...", flush=True)
        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        nbformat.write(notebook, path)
        print(f"Đã hoàn tất {name}", flush=True)


if __name__ == "__main__":
    main()
