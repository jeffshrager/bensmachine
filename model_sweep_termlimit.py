#!/usr/bin/env python3
# polarization_term_sweep_standalone.py
# Standalone simulator for coupled Public ↔ SCOTUS polarization
# Focus: sweep SCOTUS term limits (--sweep-term-limit), with feedback knobs and optional attrition.

'''

examples

single run (alternating presidents, 18-year terms, big plot):

python polarization_term_sweep_standalone.py \
  --start 1866 --end 2025 \
  --term-limit 18 --alpha 0.15 --gain 1.5 --noise 0.03 \
  --plot


sweep term limits and save TSVs under runs/term_sweep:

python polarization_term_sweep_standalone.py \
  --sweep-term-limit "12,18,22,none" \
  --alpha 0.15 --gain 1.5 --noise 0.03 \
  --outdir runs/term_sweep --plot


use presidents.csv (recommended):

python polarization_term_sweep_standalone.py \
  --presidents-csv presidents.csv \
  --sweep-term-limit "12,18,22,none" \
  --outdir runs/term_sweep

'''

import argparse
import pathlib
import sys
from typing import List, Optional, Tuple

from polarization_core import (
    SimParams, SimResult, simulate, write_tsv, plot_series,
    parse_term_limit_list, make_base_params_from_args, make_presidents_mapping_from_args
)

# =============================================================================
# Term limit specific sweeps
# =============================================================================

def sweep_term_limits(
    term_limits: List[Optional[int]],
    base_params: SimParams,
    presidents_by_year: dict,
) -> List[Tuple[Optional[int], SimResult]]:
    """Sweep through different term limit configurations."""
    out: List[Tuple[Optional[int], SimResult]] = []
    for tl in term_limits:
        p = SimParams(**{**base_params.__dict__, "term_limit_years": tl})
        sim = simulate(p, presidents_by_year)
        out.append((tl, sim))
    return out

# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Standalone SCOTUS–Public polarization simulator with term-limit sweep"
    )
    # timeline
    ap.add_argument("--start", type=int, default=1866)
    ap.add_argument("--end", type=int, default=2025)

    # feedback knobs
    ap.add_argument("--alpha", type=float, default=0.15, help="Public response rate λ") # TODO alpha/lambda?
    ap.add_argument("--gain", type=float, default=1.50, help="Public gain γ in tanh(γ S)")
    ap.add_argument("--noise", type=float, default=0.03, help="Public noise σ")

    # court config
    ap.add_argument("--court-size", type=int, default=9, help="Number of justices")
    ap.add_argument("--term-limit", type=str, default="18",
                    help='Default single-run term limit in years, or "none" for lifetime')
    ap.add_argument("--retire-prob", type=float, default=0.00,
                    help="Annual random retirement probability per justice-year")

    # seeds & init
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--initial-public", type=float, default=0.0)

    # presidents mapping (exogenous)
    ap.add_argument("--presidents-csv", type=str, default="",
                    help='Optional CSV: start_year,end_year,party with party in {D,R}')
    ap.add_argument("--alt-first", type=str, default="R",
                    help="If no CSV provided, alternating timeline starts with this party (D or R)")
    ap.add_argument("--alt-term", type=int, default=4,
                    help="Fallback alternating timeline term length (years)")

    # sweep + outputs
    ap.add_argument("--sweep-term-limit", type=str, default="",
                    help='Comma list of term limits to sweep, e.g. "12,18,22,none". '
                         'If omitted, runs a single simulation using --term-limit.')
    ap.add_argument("--out", type=str, default="",
                    help="If set (single run): write TSV to this path. "
                         "If sweeping: used as prefix to write multiple TSVs.")
    ap.add_argument("--outdir", type=str, default="runs/term_sweep",
                    help="If sweeping and --out is empty, write TSVs to this directory.")
    ap.add_argument("--plot", action="store_true", help="Show a plot for each run")

    args = ap.parse_args()

    # Presidents mapping (exogenous)
    mapping = make_presidents_mapping_from_args(args)

    # Base params
    base = make_base_params_from_args(args)

    # Sweep or single run
    if args.sweep_term_limit:
        term_limits = parse_term_limit_list(args.sweep_term_limit)
        results = sweep_term_limits(term_limits, base, mapping)

        # Decide output naming
        prefix_path = pathlib.Path(args.out) if args.out else None
        outdir = None if prefix_path else pathlib.Path(args.outdir)
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)

        for tl, sim in results:
            label = "lifetime" if tl is None else f"{tl}y"
            if prefix_path:
                # If user gave --out like "runs/term_sweep/pol", produce pol_18y.tsv etc.
                p = prefix_path
                out_path = p if p.suffix else p.with_suffix("")
                out_file = pathlib.Path(f"{out_path}_{label}.tsv")
            else:
                out_file = outdir / f"polarization_{label}.tsv"
            write_tsv(out_file, sim)
            print(f"Wrote {out_file}")
            if args.plot:
                plot_series(sim, title=f"Term limit: {label}")
    else:
        # Single run with --term-limit
        sim = simulate(base, mapping)
        if args.out:
            write_tsv(pathlib.Path(args.out), sim)
            print(f"Wrote {args.out}")
        if args.plot:
            ttl = "lifetime" if base.term_limit_years is None else f"{base.term_limit_years}y"
            plot_series(sim, title=f"Term limit: {ttl}")

if __name__ == "__main__":
    main()
