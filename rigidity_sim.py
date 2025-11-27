#!/usr/bin/env python3
"""
rigidity_sim.py

Toy simulation of institutional "rigidity" under lifetime vs fixed-term
appointments as life expectancy increases.

Idea:
- Environment (public preference) drifts as a random walk.
- Each seat has a fixed "position" set at appointment time.
- New appointees are drawn from a distribution centered on the current
  environment, plus some noise.
- Rigidity is measured as the average misalignment between the mean seat
  position and the environment over time.

We compare:
- Lifetime appointments with tenure = life_expectancy - appointment_age
- Fixed terms with a constant term_length (e.g., 18 years)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class SimConfig:
    n_seats: int = 9
    years: int = 300           # total simulated years
    burn_in: int = 100         # ignore early years for measuring rigidity
    env_sigma: float = 0.20    # year-to-year environment drift (random walk step std)
    appoint_sigma: float = 0.50  # noise of appointee around environment at appointment
    n_runs: int = 50           # independent runs for averaging
    seed: int | None = 123     # base RNG seed (set None for fully random)


def simulate_rigidity(
    tenure_years: int,
    cfg: SimConfig,
) -> float:
    """
    Simulate average misalignment ("rigidity") for a given tenure length.

    Parameters
    ----------
    tenure_years : int
        How long each appointment lasts, in years. After this many years,
        a seat is reappointed to a new member whose position is drawn
        around the current environment.
    cfg : SimConfig
        Global simulation settings.

    Returns
    -------
    float
        Average |composition - environment| over time and runs,
        after burn-in.
    """
    assert tenure_years > 0, "tenure_years must be positive"
    rigidities: List[float] = []

    for run in range(cfg.n_runs):
        # Run-specific RNG
        rng = np.random.default_rng(
            None if cfg.seed is None else cfg.seed + run
        )

        # Environment starts at 0 and drifts
        env = 0.0

        # Initial appointments at env=0
        judges = rng.normal(loc=env, scale=cfg.appoint_sigma, size=cfg.n_seats)
        time_since_appointment = np.zeros(cfg.n_seats, dtype=int)

        misalignments: List[float] = []

        for t in range(cfg.years):
            # Environment random walk step
            env += rng.normal(loc=0.0, scale=cfg.env_sigma)

            # Mean position of the institution
            composition = float(judges.mean())

            # Misalignment measure (absolute difference)
            misalign = abs(composition - env)

            if t >= cfg.burn_in:
                misalignments.append(misalign)

            # Advance clock and reappoint where tenure has expired
            time_since_appointment += 1
            expired = time_since_appointment >= tenure_years
            if np.any(expired):
                n_expired = int(expired.sum())
                # New appointees drawn around *current* environment
                judges[expired] = rng.normal(
                    loc=env, scale=cfg.appoint_sigma, size=n_expired
                )
                time_since_appointment[expired] = 0

        if misalignments:
            rigidities.append(float(np.mean(misalignments)))

    return float(np.mean(rigidities)) if rigidities else float("nan")


def explore_life_expectancy(
    appointment_age: int = 50,
    life_expectancies: List[int] | None = None,
    term_length: int = 18,
    cfg: SimConfig | None = None,
) -> List[Dict[str, float]]:
    """
    Sweep over life expectancies and compare rigidity for:

    - lifetime tenure = life_expectancy - appointment_age
    - fixed term = term_length (constant)

    Returns a list of dicts with results for each life expectancy.
    """
    if cfg is None:
        cfg = SimConfig()

    if life_expectancies is None:
        life_expectancies = [70, 75, 80, 85, 90]

    results: List[Dict[str, float]] = []

    for E in life_expectancies:
        tenure_lifetime = E - appointment_age
        if tenure_lifetime <= 0:
            raise ValueError(
                f"life_expectancy={E} <= appointment_age={appointment_age}"
            )

        R_lifetime = simulate_rigidity(tenure_years=tenure_lifetime, cfg=cfg)
        R_fixed = simulate_rigidity(tenure_years=term_length, cfg=cfg)

        results.append(
            {
                "life_expectancy": float(E),
                "appointment_age": float(appointment_age),
                "tenure_lifetime": float(tenure_lifetime),
                "tenure_fixed": float(term_length),
                "rigidity_lifetime": R_lifetime,
                "rigidity_fixed": R_fixed,
            }
        )

    return results


def main():
    # Example usage: compare lifetime vs 18-year terms
    cfg = SimConfig(
        n_seats=9,
        years=300,
        burn_in=100,
        env_sigma=0.20,
        appoint_sigma=0.50,
        n_runs=50,
        seed=123,
    )

    appointment_age = 50
    life_expectancies = [70, 75, 80, 85, 90]
    term_length = 18

    results = explore_life_expectancy(
        appointment_age=appointment_age,
        life_expectancies=life_expectancies,
        term_length=term_length,
        cfg=cfg,
    )

    print("Simulation results (rigidity = avg |composition - environment|):")
    print(
        f"  n_seats={cfg.n_seats}, env_sigma={cfg.env_sigma}, "
        f"appoint_sigma={cfg.appoint_sigma}, n_runs={cfg.n_runs}\n"
    )
    print(
        f"{'E':>5} {'tenure_life':>11} {'tenure_fixed':>13} "
        f"{'R_life':>10} {'R_fixed':>10}"
    )
    print("-" * 55)

    for row in results:
        E = int(row["life_expectancy"])
        tenure_life = row["tenure_lifetime"]
        tenure_fixed = row["tenure_fixed"]
        R_life = row["rigidity_lifetime"]
        R_fixed = row["rigidity_fixed"]
        print(
            f"{E:5d} {tenure_life:11.1f} {tenure_fixed:13.1f} "
            f"{R_life:10.3f} {R_fixed:10.3f}"
        )


if __name__ == "__main__":
    main()
