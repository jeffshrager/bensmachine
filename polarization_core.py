#!/usr/bin/env python3
"""
polarization_core.py
Core simulation components for coupled Public ↔ SCOTUS polarization modeling.
Shared by multiple simulation scripts.
"""

import csv
import math
import pathlib
import random
import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # plotting optional

# =============================================================================
# Constants and utilities
# =============================================================================

# Parties encoded as: D = -1.0, R = +1.0
# Public polarization P_t ∈ [-1, 1]; SCOTUS polarization S_t ∈ [-1, 1]
# Time step = 1 year

PARTY_TO_LEAN = {"D": -1.0, "R": +1.0}


def clamp(x: float, lo=-1.0, hi=1.0) -> float:
    """Clamp value to range [lo, hi]."""
    return max(lo, min(hi, x))


# =============================================================================
# Core data structures
# =============================================================================

@dataclass
class Justice:
    """Represents a Supreme Court justice with appointment details and political lean."""
    appointed_year: int
    term_limit_years: Optional[int]  # None = lifetime tenure
    lean: float  # -1 (D) to +1 (R)

    def expires_in_year(self, year: int) -> bool:
        """Check if this justice's term expires in the given year."""
        if self.term_limit_years is None:
            return False
        return (year - self.appointed_year) >= self.term_limit_years


@dataclass
class Court:
    """Manages the Supreme Court with justices and their lifecycle."""
    size: int = 9
    term_limit_years: Optional[int] = None
    justices: List[Justice] = field(default_factory=list)

    def polarization(self) -> float:
        """Calculate current court polarization as mean justice lean."""
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

        # Fill to size with current president's party lean
        while len(self.justices) < self.size:
            self.justices.append(
                Justice(
                    appointed_year=year,
                    term_limit_years=self.term_limit_years,
                    lean=PARTY_TO_LEAN[president_party],
                )
            )


@dataclass
class SimParams:
    """Configuration parameters for polarization simulation."""
    start_year: int = 1866
    end_year: int = 2025
    public_alpha: float = 0.15   # λ: how quickly public responds
    public_gain: float = 1.50    # γ: how strongly court S shapes public; inside tanh(γ S)
    noise_sigma: float = 0.03    # σ: public noise
    term_limit_years: Optional[int] = 18  # None = lifetime
    retire_prob: float = 0.00    # annual random retirement prob per justice-year
    initial_public: float = 0.0  # start "non-polarized"
    court_size: int = 9
    seed: int = 42


@dataclass
class SimResult:
    """Results from a polarization simulation run."""
    years: List[int]
    public: List[float]
    scotus: List[float]
    presidents: List[str]


# =============================================================================
# Presidents timeline helpers
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
# Core simulation
# =============================================================================

def simulate(params: SimParams, presidents_by_year: Dict[int, str]) -> SimResult:
    """Run the core polarization simulation with given parameters and president timeline."""
    random.seed(params.seed)

    court = Court(size=params.court_size, term_limit_years=params.term_limit_years)

    # Bootstrap: fill Court in start_year with the then-president's party
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
# I/O helpers
# =============================================================================

def write_tsv(path: pathlib.Path, sim: SimResult) -> None:
    """Write simulation results to TSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("year\tpresident_party\tpublic_polarization\tscotus_polarization\n")
        for y, pr, P, S in zip(sim.years, sim.presidents, sim.public, sim.scotus):
            f.write(f"{y}\t{pr}\t{P:.6f}\t{S:.6f}\n")


def plot_series(sim: SimResult, title: str = "Polarization over time", figsize=(12, 6), fontsize=14):
    """Plot simulation results with matplotlib."""
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
# Common sweep utilities
# =============================================================================

def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated list of floats."""
    vals = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals


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


def make_base_params_from_args(args) -> SimParams:
    """Create SimParams from command line arguments (common pattern)."""
    tl_default = None if args.term_limit.lower() in ("none", "lifetime", "inf") else int(args.term_limit)
    return SimParams(
        start_year=args.start, 
        end_year=args.end,
        public_alpha=args.alpha, 
        public_gain=args.gain,
        noise_sigma=args.noise, 
        term_limit_years=tl_default,
        retire_prob=args.retire_prob,
        initial_public=args.initial_public, 
        court_size=args.court_size,
        seed=args.seed,
    )


def make_presidents_mapping_from_args(args) -> Dict[int, str]:
    """Create presidents mapping from command line arguments (common pattern)."""
    if args.presidents_csv:
        return load_presidents_from_csv(pathlib.Path(args.presidents_csv))
    else:
        return make_alternating_presidents(
            args.start, args.end, first_party=args.alt_first, term_years=args.alt_term
        )