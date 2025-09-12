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

import argparse, csv, math, random, statistics, sys, pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting is optional

# ----------------------------
# Conventions
# ----------------------------
# Parties: D = -1.0, R = +1.0
# Public polarization P_t in [-1, 1], SCOTUS polarization S_t in [-1, 1]
# Time step = 1 year

PARTY_TO_LEAN = {"D": -1.0, "R": +1.0}

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
        """Advance one year: natural expirations + stochastic retirements, then fill vacancies."""
        # Natural expirations from term limits
        remaining = []
        for j in self.justices:
            if j.expires_in_year(year):
                vacancies_this_year += 1
            else:
                remaining.append(j)

        # Stochastic attrition (health, scandal, strategic retirement, etc.)
        really_remaining = []
        for j in remaining:
            if retire_prob > 0.0 and random.random() < retire_prob:
                vacancies_this_year += 1
            else:
                really_remaining.append(j)

        self.justices = really_remaining

        # Fill to full size with current president’s party
        while len(self.justices) < self.size:
            self.justices.append(
                Justice(
                    appointed_year=year,
                    term_limit_years=self.term_limit_years,
                    lean=PARTY_TO_LEAN[president_party],
                )
            )

def load_presidents_from_csv(csv_path: pathlib.Path) -> Dict[int, str]:
    """
    CSV columns: start_year,end_year,party  (party ∈ {D,R})
    Returns dict year->party for years [start_year, end_year] inclusive (per row).
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
    """Fallback: alternate party every 'term_years' starting at start_year."""
    mapping: Dict[int, str] = {}
    p0 = first_party.upper()
    assert p0 in ("D", "R")
    for y in range(start_year, end_year + 1):
        block = (y - start_year) // term_years
        party = "R" if (p0 == "R" and block % 2 == 0) or (p0 == "D" and block % 2 == 1) else "D"
        mapping[y] = party
    return mapping

@dataclass
class SimParams:
    start_year: int = 1866
    end_year: int = 2025
    public_alpha: float = 0.15   # λ: public response rate to court
    public_gain: float = 1.50    # γ: strength of court → public inside tanh
    noise_sigma: float = 0.03    # σ: public noise
    term_limit_years: Optional[int] = 18
    retire_prob: float = 0.00    # annual random retirement prob per justice-year
    initial_public: float = 0.0
    court_size: int = 9
    seed: int = 42

@dataclass
class SimResult:
    years: List[int]
    public: List[float]
    scotus: List[float]
    presidents: List[str]

def clamp(x: float, lo=-1.0, hi=1.0) -> float:
    return max(lo, min(hi, x))

def simulate(params: SimParams, presidents_by_year: Dict[int, str]) -> SimResult:
    random.seed(params.seed)

    court = Court(size=params.court_size, term_limit_years=params.term_limit_years)

    # Bootstrap: fill the Court at start with that year's president
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

        # Court evolves (expirations + attrition + fills)
        court.step_year(
            year,
            president_party,
            vacancies_this_year=0,
            retire_prob=params.retire_prob,
        )
        S = court.polarization()

        # Public responds: P_{t+1} = (1-λ)P_t + λ*tanh(γ S_t) + noise
        influence = math.tanh(params.public_gain * S)
        noise = random.gauss(0.0, params.noise_sigma)
        P = (1.0 - params.public_alpha) * P + params.public_alpha * influence + noise
        P = clamp(P)

        years.append(year)
        publics.append(P)
        scoti.append(S)
        pres.append(president_party)

    return SimResult(years=years, public=publics, scotus=scoti, presidents=pres)

def write_tsv(path: pathlib.Path, sim: SimResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("year\tpresident_party\tpublic_polarization\tscotus_polarization\n")
        for y, pr, P, S in zip(sim.years, sim.presidents, sim.public, sim.scotus):
            f.write(f"{y}\t{pr}\t{P:.6f}\t{S:.6f}\n")

def plot_series(sim: SimResult, title: str = "Polarization over time"):
    if plt is None:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return
    plt.figure()
    plt.plot(sim.years, sim.public, label="Public polarization (P_t)")
    plt.plot(sim.years, sim.scotus, label="SCOTUS polarization (S_t)")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Polarization (−1=D … +1=R)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def parse_float_list(s: str) -> List[float]:
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals

def sweep_retire_prob(
    retire_probs: List[float],
    base_params: SimParams,
    presidents_by_year: Dict[int, str],
) -> List[Tuple[float, SimResult]]:
    results = []
    for rp in retire_probs:
        p = SimParams(**{**base_params.__dict__, "retire_prob": rp})
        sim = simulate(p, presidents_by_year)
        results.append((rp, sim))
    return results

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
    if args.presidents_csv:
        mapping = load_presidents_from_csv(pathlib.Path(args.presidents_csv))
    else:
        mapping = make_alternating_presidents(args.start, args.end,
                                              first_party=args.alt_first,
                                              term_years=args.alt_term)

    tl = None if args.term_limit.lower() in ("none", "lifetime", "inf") else int(args.term_limit)
    base = SimParams(
        start_year=args.start, end_year=args.end,
        public_alpha=args.alpha, public_gain=args.gain,
        noise_sigma=args.noise, term_limit_years=tl,
        retire_prob=args.retire_prob, initial_public=args.initial_public,
        court_size=args.court_size, seed=args.seed
    )

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
            ttl = "lifetime" if tl is None else f"{tl}y"
            plot_series(sim, title=f"Term limit: {ttl} | retire p={base.retire_prob:.3f}")

if __name__ == "__main__":
    main()
