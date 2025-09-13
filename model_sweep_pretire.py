#!/usr/bin/env python3
# polarization_min_attrition.py
# Minimal simulator for coupled Public ↔ SCOTUS polarization
# Focused on: feedback knobs (--alpha, --gain, --noise) and attrition (--retire-prob)

'''

Quick runs

Single run, alternating presidents, 18-year terms, modest noise:

python polarization_min_attrition.py --start 1866 --end 2025 \
  --term-limit 18 --alpha 0.15 --gain 1.5 --noise 0.03 \
  --retire-prob 0.02 --plot


Use a presidents CSV (recommended for realism):

python polarization_min_attrition.py --presidents-csv presidents.csv \
  --term-limit 18 --alpha 0.15 --gain 1.5 --noise 0.03 \
  --retire-prob 0.02 --out runs/attrition_single.tsv --plot


Sweep attrition (writes multiple TSVs + plots each):

python polarization_min_attrition.py --presidents-csv presidents.csv \
  --term-limit 18 --alpha 0.15 --gain 1.5 --noise 0.03 \
  --sweep-retire "0.00,0.02,0.05,0.08" --out runs/attrition

'''

import argparse
import pathlib
from typing import List, Tuple

from polarization_core import (
    SimParams, SimResult, simulate, write_tsv, plot_series,
    parse_float_list, make_base_params_from_args, make_presidents_mapping_from_args
)

# =============================================================================
# Retirement probability specific sweeps
# =============================================================================

def sweep_retire_prob(
    retire_probs: List[float],
    base_params: SimParams,
    presidents_by_year: dict,
) -> List[Tuple[float, SimResult]]:
    """Sweep through different retirement probability configurations."""
    results = []
    for rp in retire_probs:
        p = SimParams(**{**base_params.__dict__, "retire_prob": rp})
        sim = simulate(p, presidents_by_year)
        results.append((rp, sim))
    return results

# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Minimal SCOTUS–Public polarization simulator (attrition & feedback)")
    ap.add_argument("--start", type=int, default=1866)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--term-limit", type=str, default="18", help='Years or "none" for lifetime')
    ap.add_argument("--alpha", type=float, default=0.15, help="Public response rate λ")
    ap.add_argument("--gain", type=float, default=1.50, help="Public gain γ inside tanh(γ S)")
    ap.add_argument("--noise", type=float, default=0.03, help="Public noise σ")
    ap.add_argument("--retire-prob", type=float, default=0.00, help="Annual retirement probability per justice-year")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--initial-public", type=float, default=0.0)
    ap.add_argument("--court-size", type=int, default=9)

    ap.add_argument("--presidents-csv", type=str, default="",
                    help='Optional CSV: start_year,end_year,party with party in {D,R}')
    ap.add_argument("--alt-first", type=str, default="R",
                    help="If no CSV, alternating timeline starts with this party (D or R)")
    ap.add_argument("--alt-term", type=int, default=4, help="Alt timeline term length (years)")

    ap.add_argument("--out", type=str, default="", help="Write single-run TSV to this path")
    ap.add_argument("--plot", action="store_true", help="Show a matplotlib plot")

    ap.add_argument("--sweep-retire", type=str, default="",
                    help='Comma list of retirement probs to sweep, e.g. "0.00,0.02,0.05,0.08". '
                         'If set, writes multiple TSVs when --out is provided; plots each if --plot.')

    args = ap.parse_args()

    # Presidents mapping (exogenous only)
    mapping = make_presidents_mapping_from_args(args)
    
    # Base parameters
    base = make_base_params_from_args(args)

    if args.sweep_retire:
        rps = parse_float_list(args.sweep_retire)
        results = sweep_retire_prob(rps, base, mapping)
        for rp, sim in results:
            label = f"retire{rp:.3f}".replace(".", "p")
            if args.out:
                p = pathlib.Path(args.out)
                out_path = p if p.suffix else p.with_suffix("")
                out_file = pathlib.Path(f"{out_path}_{label}.tsv")
                write_tsv(out_file, sim)
                print(f"Wrote {out_file}")
            if args.plot:
                plot_series(sim, title=f"Attrition p={rp:.3f}")
    else:
        sim = simulate(base, mapping)
        if args.out:
            write_tsv(pathlib.Path(args.out), sim)
            print(f"Wrote {args.out}")
        if args.plot:
            ttl = "lifetime" if base.term_limit_years is None else f"{base.term_limit_years}y"
            plot_series(sim, title=f"Term limit: {ttl} | retire p={base.retire_prob:.3f}")

if __name__ == "__main__":
    main()