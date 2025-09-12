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
import csv
import math
import random
import statistics
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional

# =============================================================================
# Conventions and utilities
# =============================================================================

# Parties encoded as: D = -1.0, R = +1.0
# Public polarization P_t ∈ [-1, 1]; SCOTUS polarization S_t ∈ [-1, 1]
# Time step = 1 year

PARTY_TO_LEAN = {"D": -1.0, "R": +1.0}

def clamp(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

# =============================================================================
# Core data structures
# =============================================================================

@dataclass
class Justice:
    appointed_year: int
    term_limit_years: Optional[int]  # None = lifetime
    lean: float  # -1 (D) to +1 (R)

    def expires_in_year(self, year: int) -> bool:
        if self.term_limit_years is None:
            return False
        return (year - self.appointed_year) >= self.term_limit_years

@dataclass
class Court:
    size: int = 9
    term_limit_years: Optional[int] = None
    justices: List[Justice] = field(default_factory=list)

    def polarization(self) -> float:
        if not self.justices:
            return 0.0
        return statistics.fmean(j.lean for j in self.justices)

    def step_year(
        self,
        year: int,
        president_party: str,
        vacancies_this_year: int = 0,
        retire_prob: float = 0.0,
    ) -> None:
        """
        Advance one year: natural expirations + stochastic retirements, then fill vacancies.
        """
        # Natural expirations (term limits)
        remaining = []
        for j in self.justices:
            if j.expires_in_year(year):
                vacancies_this_year += 1
            else:
                remaining.append(j)

        # Stochastic attrition (health, scandals, strategic exits, etc.)
        really_remaining = []
        for j in remaining:
            if retire_prob > 0.0 and random.random() < retire_prob:
                vacancies_this_year += 1
            else:
                really_remaining.append(j)

        self.justices = really_remaining

        # Fill to size with current president’s party lean
        while len(self.justices) < self.size:
            self.justices.append(
                Justice(
                    appointed_year=year,
                    term_limit_years=self.term_limit_years,
                    lean=PARTY_TO_LEAN[president_party],
                )
            )

# =============================================================================
# Presidents timeline helpers (exogenous presidents only)
# =============================================================================

def load_presidents_from_csv(csv_path: pathlib.Path) -> Dict[int, str]:
    """
    Load presidents timeline from CSV with columns:
      start_year,end_year,party
    party ∈ {"D","R"} (case-insensitive).
    Returns dict: year -> party for all years [start_year, end_year] inclusive.
    """
    mapping: Dict[int, str] = {}
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sy, ey = int(row["start_year"]), int(row["end_year"])
            party = row["party"].strip().upper()
            assert party in ("D", "R"), f"Invalid party {party}"
            for y in range(sy, ey + 1):
                mapping[y] = party
    return mapping

def make_alternating_presidents(
    start_year: int,
    end_year: int,
    first_party: str = "R",
    term_years: int = 4,
) -> Dict[int, str]:
    """
    Simple fallback timeline: alternate parties every 'term_years' starting at start_year.
    """
    mapping: Dict[int, str] = {}
    p0 = first_party.upper()
    assert p0 in ("D", "R")
    for y in range(start_year, end_year + 1):
        block = (y - start_year) // term_years
        if p0 == "R":
            party = "R" if (block % 2 == 0) else "D"
        else:
            party = "D" if (block % 2 == 0) else "R"
        mapping[y] = party
    return mapping

# =============================================================================
# Simulation
# =============================================================================

@dataclass
class SimParams:
    start_year: int = 1866
    end_year: int = 2025
    public_alpha: float = 0.15   # λ: how quickly public responds
    public_gain: float = 1.50    # γ: how strongly court S shapes public; inside tanh(γ S)
    noise_sigma: float = 0.03    # σ: public noise
    term_limit_years: Optional[int] = 18  # None = lifetime
    retire_prob: float = 0.00    # annual random retirement prob per justice-year
    initial_public: float = 0.0  # start “non-polarized”
    court_size: int = 9
    seed: int = 42

@dataclass
class SimResult:
    years: List[int]
    public: List[float]
    scotus: List[float]
    presidents: List[str]

def simulate(params: SimParams, presidents_by_year: Dict[int, str]) -> SimResult:
    random.seed(params.seed)

    court = Court(size=params.court_size, term_limit_years=params.term_limit_years)

    # Bootstrap: fill Court in start_year with the then-president’s party
    pres0 = presidents_by_year.get(params.start_year, "R")
    court.step_year(
        params.start_year,
        pres0,
        vacancies_this_year=params.court_size,
        retire_prob=params.retire_prob,
    )

    P = params.initial_public
    years: List[int] = []
    publics: List[float] = []
    scoti: List[float] = []
    pres: List[str] = []

    for year in range(params.start_year, params.end_year + 1):
        president_party = presidents_by_year.get(year, pres0)

        # Court evolves first (appointments/expirations at 'year')
        court.step_year(
            year,
            president_party,
            vacancies_this_year=0,
            retire_prob=params.retire_prob,
        )
        S = court.polarization()  # mean lean in [-1, 1]

        # Public responds to court (feedback)
        # P_{t+1} = (1-λ) P_t + λ * tanh(γ S_t) + noise
        influence = math.tanh(params.public_gain * S)
        noise = random.gauss(0.0, params.noise_sigma)
        P = (1.0 - params.public_alpha) * P + params.public_alpha * influence + noise
        P = clamp(P)

        years.append(year)
        publics.append(P)
        scoti.append(S)
        pres.append(president_party)

    return SimResult(years=years, public=publics, scotus=scoti, presidents=pres)

# =============================================================================
# I/O helpers (TSV + plotting)
# =============================================================================

def write_tsv(path: pathlib.Path, sim: SimResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("year\tpresident_party\tpublic_polarization\tscotus_polarization\n")
        for y, pr, P, S in zip(sim.years, sim.presidents, sim.public, sim.scotus):
            f.write(f"{y}\t{pr}\t{P:.6f}\t{S:.6f}\n")

def plot_series(sim: SimResult, title: str = "Polarization over time", figsize=(12, 6), fontsize=14):
    if plt is None:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return
    plt.figure(figsize=figsize)
    plt.plot(sim.years, sim.public, label="Public polarization (P_t)", linewidth=2.0)
    plt.plot(sim.years, sim.scotus, label="SCOTUS polarization (S_t)", linewidth=2.0)
    plt.axhline(0.0, linestyle="--", color="gray")
    plt.xlabel("Year", fontsize=fontsize)
    plt.ylabel("Polarization (−1 … +1)", fontsize=fontsize)
    plt.title(title, fontsize=fontsize + 2)
    plt.xticks(fontsize=fontsize - 2)
    plt.yticks(fontsize=fontsize - 2)
    plt.legend(fontsize=fontsize - 2)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Sweeps
# =============================================================================

def parse_term_limit_list(s: str) -> List[Optional[int]]:
    """
    Parse a comma-separated list like "12,18,22,none" into [12, 18, 22, None].
    """
    vals: List[Optional[int]] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in ("none", "lifetime", "inf"):
            vals.append(None)
        else:
            vals.append(int(tok))
    return vals

def sweep_term_limits(
    term_limits: List[Optional[int]],
    base_params: SimParams,
    presidents_by_year: Dict[int, str],
) -> List[Tuple[Optional[int], SimResult]]:
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
    ap.add_argument("--alpha", type=float, default=0.15, help="Public response rate λ")
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
    if args.presidents_csv:
        mapping = load_presidents_from_csv(pathlib.Path(args.presidents_csv))
    else:
        mapping = make_alternating_presidents(
            args.start, args.end, first_party=args.alt_first, term_years=args.alt_term
        )

    # Base params
    tl_default = None if args.term_limit.lower() in ("none", "lifetime", "inf") else int(args.term_limit)
    base = SimParams(
        start_year=args.start, end_year=args.end,
        public_alpha=args.alpha, public_gain=args.gain,
        noise_sigma=args.noise, term_limit_years=tl_default,
        retire_prob=args.retire_prob,
        initial_public=args.initial_public, court_size=args.court_size,
        seed=args.seed,
    )

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
            ttl = "lifetime" if tl_default is None else f"{tl_default}y"
            plot_series(sim, title=f"Term limit: {ttl}")

if __name__ == "__main__":
    main()
