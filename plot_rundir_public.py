#!/usr/bin/env python3
# plot_rundir_public.py
# Aggregate a directory of TSVs and plot ONLY the public_polarization lines.

'''

Examples

Plot all TSVs in a run dir:

python plot_rundir_public.py --rundir runs/attrition


Recurse & save to a PNG:

python plot_rundir_public.py --rundir runs/attrition --recursive --save runs/attrition/summary_public.png


Custom title / bounds:

python plot_rundir_public.py --rundir runs/attrition \
  --title "Attrition sweep: public only" --ymin -1 --ymax 1

'''

import argparse
import pathlib
import re
import sys
from typing import List, Tuple

import csv

try:
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib is required to plot. Install it and re-run.", file=sys.stderr)
    raise

LABEL_RE = re.compile(r"retire(\d+(?:p\d+)?)", re.IGNORECASE)

def parse_label_from_filename(p: pathlib.Path) -> str:
    """
    Pull a retire-prob label from filenames like:
      ..._retire0p020.tsv  ->  'retire p=0.020'
    Fallback: use file stem.
    """
    m = LABEL_RE.search(p.stem)
    if m:
        raw = m.group(1)
        # Convert "0p020" -> "0.020"
        nice = raw.replace("p", ".")
        try:
            val = float(nice)
            return f"retire p={val:.3f}"
        except ValueError:
            return f"retire {nice}"
    return p.stem

def read_public_series(tsv_path: pathlib.Path) -> Tuple[List[int], List[float]]:
    years: List[int] = []
    publics: List[float] = []
    with tsv_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        # Expect columns: year, president_party, public_polarization, scotus_polarization
        if "year" not in reader.fieldnames or "public_polarization" not in reader.fieldnames:
            raise ValueError(f"{tsv_path} missing required columns")
        for row in reader:
            try:
                years.append(int(row["year"]))
                publics.append(float(row["public_polarization"]))
            except Exception:
                # Skip malformed rows
                continue
    return years, publics

def collect_tsvs(rundir: pathlib.Path, recursive: bool) -> List[pathlib.Path]:
    if recursive:
        return sorted(p for p in rundir.rglob("*.tsv"))
    return sorted(p for p in rundir.glob("*.tsv"))

def main():
    ap = argparse.ArgumentParser(description="Plot public_polarization from all TSVs in a run directory.")
    ap.add_argument("--rundir", required=True, help="Directory containing TSV outputs")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--title", default="Public Polarization (all runs)", help="Plot title")
    ap.add_argument("--ymin", type=float, default=-1.0, help="Y-axis min (default -1)")
    ap.add_argument("--ymax", type=float, default=1.0, help="Y-axis max (default +1)")
    ap.add_argument("--save", default="", help="If set, save figure to this path instead of showing")
    args = ap.parse_args()

    rundir = pathlib.Path(args.rundir)
    if not rundir.exists() or not rundir.is_dir():
        print(f"--rundir path not found or not a directory: {rundir}", file=sys.stderr)
        sys.exit(1)

    tsvs = collect_tsvs(rundir, args.recursive)
    if not tsvs:
        print(f"No .tsv files found in {rundir} (recursive={args.recursive})", file=sys.stderr)
        sys.exit(1)

    plt.figure()
    any_plotted = False

    for tsv in tsvs:
        try:
            years, publics = read_public_series(tsv)
            if not years or not publics:
                continue
            label = parse_label_from_filename(tsv)
            plt.plot(years, publics, label=label)
            any_plotted = True
        except Exception as e:
            print(f"Skipping {tsv}: {e}", file=sys.stderr)

    if not any_plotted:
        print("Found TSVs but none had valid public_polarization data.", file=sys.stderr)
        sys.exit(1)

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Public polarization (−1 … +1)")
    plt.title(args.title)
    plt.ylim(args.ymin, args.ymax)
    plt.legend(loc="best")
    plt.tight_layout()

    if args.save:
        out_path = pathlib.Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
