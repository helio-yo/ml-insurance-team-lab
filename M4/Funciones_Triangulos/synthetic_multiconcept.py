"""Synthetic multiconcept monthly triangle database generator.

Generates a long-format database with four related concepts:
  Exposure, Earned Premium, Claims Reported, Loss Incurred.

Dependency structure:
  Final Exposure
      ├── Final Earned Premium     (correlated via premium rate)
      └── Ultimate Claims Reported (correlated via frequency * exposure)
               └── Ultimate Loss Incurred (correlated via severity * claims)

Simulation flow:
  1. Simulate final Exposure per accident period.
  2. Simulate final Earned Premium from Exposure × premium rate.
  3. Simulate ultimate Claims Reported from Exposure × claim frequency.
  4. Simulate ultimate Loss Incurred from Claims × average severity.
  5. Apply concept-specific development curves to produce triangle cells.
  6. Flatten all cells to the project long-format database schema.

Final accident-period quantities are simulated first; triangle cells are derived
from those quantities via development curves, not simulated independently.

Exposure and Earned Premium use a cancellation/audit-adjustment interpretation:
the value at dev_0 is an initial booked estimate; successive development periods
reflect downward revisions from cancellations, endorsements, and audit adjustments.
Development factors for these two concepts are therefore below one.

Claims Reported and Loss Incurred use a conventional non-decreasing cumulative
pattern. Claims develop materially faster than Loss Incurred.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class MultiConceptSimulationConfig:
    """Configuration for the multiconcept synthetic database simulation.

    Trend parameters are annual rates (e.g. 0.03 = 3% per year, applied as
    exp(rate * months / 12)).  Seasonal amplitudes are fractions of the monthly
    mean (e.g. 0.10 = ±10% amplitude on a sine wave).

    The seed controls the NumPy random number generator; the same seed always
    produces the same database, enabling reproducible classroom demonstrations.
    """

    # ---- Period configuration ----
    n_accident_months: int = 120
    n_dev_periods: int = 120
    start_accident_period: str = "2016-01"

    # ---- Reproducibility ----
    seed: int = 42  # Same seed → same database on every run

    # ---- Level column values ----
    level_1: str = "Synthetic"
    level_2: str = "Portfolio"
    level_3: str = "Auto"

    # ---- Exposure (insured vehicles) ----
    exposure_base: float = 10_000.0           # Starting exposure count
    exposure_annual_trend: float = 0.03       # 3% annual growth rate
    exposure_seasonal_amp: float = 0.10       # ±10% seasonal oscillation
    exposure_noise_sigma: float = 0.04        # Log-normal noise (per-AP variation)
    # Initial booked exposure = final_exposure * overbook_factor at dev_0
    exposure_initial_overbook: float = 1.15
    # Mean cancellation adjustment speed (exponential decay rate per month)
    exposure_cancellation_speed: float = 0.08
    # Cohort-to-cohort variation in cancellation speed; nonzero so accident periods
    # develop at different rates (prevents all APs from having identical factor patterns)
    exposure_cohort_speed_sigma: float = 0.015

    # ---- Earned Premium (currency units per month) ----
    premium_rate_base: float = 75.0           # Base annual premium rate per exposure unit
    premium_rate_annual_trend: float = 0.025  # 2.5% annual rate increase
    premium_rate_seasonal_amp: float = 0.03
    premium_rate_noise_sigma: float = 0.02    # Independent rate noise (not from Exposure)
    premium_initial_overbook: float = 1.12
    premium_cancellation_speed: float = 0.06
    # Cohort-to-cohort variation in premium cancellation speed
    premium_cohort_speed_sigma: float = 0.012

    # ---- Claims Reported (integer count) ----
    freq_base: float = 0.006                  # Base claim frequency (claims per exposure unit)
    freq_annual_trend: float = 0.015          # 1.5% annual frequency increase
    freq_seasonal_amp: float = 0.12           # ±12% seasonal variation
    # Negative Binomial dispersion (higher = less overdispersed; lower = more volatile)
    freq_nb_dispersion: float = 8.0

    # Claims logistic development curve (fast — high alpha, low midpoint)
    claims_dev_alpha: float = 0.35            # Steepness of logistic curve (reporting speed)
    claims_dev_n0: float = 4.0                # Dev-period midpoint (50% reported by dev_4)
    # Cohort-to-cohort variation in reporting speed midpoint
    claims_dev_cohort_n0_sigma: float = 0.5

    # ---- Loss Incurred (currency units) ----
    sev_base: float = 5_000.0                 # Base average severity
    sev_annual_trend: float = 0.045           # 4.5% annual severity increase
    sev_seasonal_amp: float = 0.05
    sev_noise_sigma: float = 0.15             # Per-AP severity shock (clipped to ±2σ)

    # Loss logistic development curve (slow — low alpha, high midpoint)
    loss_dev_alpha: float = 0.15              # Steepness of logistic curve (settlement speed)
    loss_dev_n0: float = 12.0                 # Dev-period midpoint (50% incurred by dev_12)
    # Cohort-to-cohort variation in settlement speed midpoint
    loss_dev_cohort_n0_sigma: float = 1.0


def simulate_multiconcept_database(
    config: MultiConceptSimulationConfig = MultiConceptSimulationConfig(),
) -> pd.DataFrame:
    """Generate a synthetic long-format multiconcept monthly triangle database.

    Model construction:
        Final quantities (ultimate exposure, premium, claims, loss) are simulated
        once per accident period, incorporating trend, seasonality, and random
        variation.  Triangle cells are derived from those quantities by applying
        logistic development curves (Claims, Loss) or exponential cancellation
        curves (Exposure, Premium).

    Concept interpretation:
        Exposure and Earned Premium: initial booked estimates that decline toward
        the final audited value as cancellations and adjustments are processed.
        Development factors are below 1.0.

        Claims Reported and Loss Incurred: cumulative counts and amounts that grow
        toward ultimate values.  Development factors are above 1.0.

    Development conventions:
        dev_0 is the first observable value: the initial booked amount for Exposure
        and Earned Premium, and the first recognised claims or losses for Claims
        Reported and Loss Incurred.  Development proceeds in monthly steps.
        Only cells on or below the latest observable diagonal are included.

    Relationship structure:
        Exposure → Earned Premium  (typically r ≈ 0.96–0.98): both share the
            Exposure base; independent premium rate noise reduces correlation from 1.0.
        Exposure → Claims Reported (typically r ≈ 0.15–0.25): indirect causal
            link; NegBinomial frequency dispersion adds substantial variation.
        Claims Reported → Loss Incurred (typically r ≈ 0.75–0.85): direct causal
            link; per-AP severity shock adds moderate variation.

    Return schema:
        Standard 9-column long-format DataFrame:
        concept, basis, amount_type, level_1, level_2, level_3,
        accident_period, development_period, amount.
        All amounts are positive. Row count = 4 × (n × (n+1) / 2) where
        n = config.n_accident_months.

    Safe classroom parameter changes:
        Adjusting base levels, trend rates, seasonal amplitudes, or noise sigmas
        changes the appearance of the data without affecting model structure.
        Change the seed to explore variability across different random draws.
        Avoid changing n_accident_months, n_dev_periods, or start_accident_period
        during a first demonstration — these alter triangle dimensions and row count.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_accident_months
    m = config.n_dev_periods

    start = pd.Period(config.start_accident_period, freq="M")
    accident_periods = [str(start + i) for i in range(n)]

    # Time index: months elapsed since start of accident-period window
    t = np.arange(n, dtype=float)

    # ---- Final Exposure ----
    exp_trend = np.exp(config.exposure_annual_trend * t / 12.0)
    exp_seasonal = 1.0 + config.exposure_seasonal_amp * np.sin(2.0 * np.pi * t / 12.0)
    exp_noise = np.exp(rng.normal(0.0, config.exposure_noise_sigma, n))
    final_exposure = config.exposure_base * exp_trend * exp_seasonal * exp_noise

    # ---- Final Earned Premium ----
    rate_trend = np.exp(config.premium_rate_annual_trend * t / 12.0)
    rate_seasonal = 1.0 + config.premium_rate_seasonal_amp * np.sin(
        2.0 * np.pi * t / 12.0 + 0.5
    )
    rate_noise = np.exp(rng.normal(0.0, config.premium_rate_noise_sigma, n))
    premium_rate = config.premium_rate_base * rate_trend * rate_seasonal * rate_noise
    final_premium = final_exposure * premium_rate

    # ---- Ultimate Claims Reported ----
    freq_trend = np.exp(config.freq_annual_trend * t / 12.0)
    freq_seasonal = 1.0 + config.freq_seasonal_amp * np.sin(
        2.0 * np.pi * t / 12.0 - 0.5
    )
    mu_claims = final_exposure * config.freq_base * freq_trend * freq_seasonal
    phi = config.freq_nb_dispersion
    p_nb = phi / (phi + mu_claims)
    # numpy Generator.negative_binomial accepts float n (numpy >= 1.17)
    ult_claims = rng.negative_binomial(phi, p_nb).astype(float)
    ult_claims = np.maximum(ult_claims, 1.0)

    # ---- Ultimate Loss Incurred ----
    sev_trend = np.exp(config.sev_annual_trend * t / 12.0)
    sev_seasonal = 1.0 + config.sev_seasonal_amp * np.sin(
        2.0 * np.pi * t / 12.0 + 1.0
    )
    sev_shocks = rng.normal(0.0, config.sev_noise_sigma, n)
    sev_shocks = np.clip(
        sev_shocks, -2.0 * config.sev_noise_sigma, 2.0 * config.sev_noise_sigma
    )
    avg_severity = config.sev_base * sev_trend * sev_seasonal * (1.0 + sev_shocks)
    ult_loss = ult_claims * avg_severity

    # ---- Per-cohort development curve parameters ----
    claims_n0 = config.claims_dev_n0 + rng.normal(
        0.0, config.claims_dev_cohort_n0_sigma, n
    )
    loss_n0 = config.loss_dev_n0 + rng.normal(
        0.0, config.loss_dev_cohort_n0_sigma, n
    )
    exp_speed = np.maximum(
        config.exposure_cancellation_speed
        + rng.normal(0.0, config.exposure_cohort_speed_sigma, n),
        0.005,
    )
    prem_speed = np.maximum(
        config.premium_cancellation_speed
        + rng.normal(0.0, config.premium_cohort_speed_sigma, n),
        0.005,
    )

    # ---- Assemble triangle rows ----
    rows = []
    r_exp = 1.0 / config.exposure_initial_overbook
    r_prem = 1.0 / config.premium_initial_overbook
    alpha_c = config.claims_dev_alpha
    alpha_l = config.loss_dev_alpha
    lv1, lv2, lv3 = config.level_1, config.level_2, config.level_3

    for i in range(n):
        # Upper-triangle constraint: AP[i] + dev <= AP[n-1] + 0
        # i.e. dev <= (n - 1) - i, capped at m - 1
        max_dev = min(m - 1, n - 1 - i)
        ap = accident_periods[i]
        devs = np.arange(max_dev + 1, dtype=float)

        # Claims Reported: logistic proportion-developed curve (fast)
        cum_claims = ult_claims[i] * _logistic(alpha_c * (devs - claims_n0[i]))

        # Loss Incurred: logistic proportion-developed curve (slower)
        cum_loss = ult_loss[i] * _logistic(alpha_l * (devs - loss_n0[i]))

        # Exposure: initial overbooked estimate decaying to final (decreasing)
        # amount(dev) = initial_booked * (r_exp + (1-r_exp)*exp(-speed*dev))
        # At dev=0: amount = initial_booked; as dev->inf: amount -> final_exposure
        adj_exp = r_exp + (1.0 - r_exp) * np.exp(-exp_speed[i] * devs)
        cum_exp = final_exposure[i] * config.exposure_initial_overbook * adj_exp

        # Earned Premium: same decreasing pattern via premium cancellation curve
        adj_prem = r_prem + (1.0 - r_prem) * np.exp(-prem_speed[i] * devs)
        cum_prem = final_premium[i] * config.premium_initial_overbook * adj_prem

        n_devs = len(devs)
        for d in range(n_devs):
            dev_int = int(devs[d])
            base = ["month", "cumulative", lv1, lv2, lv3, ap, dev_int]
            rows.append(["Claims Reported"] + base + [float(cum_claims[d])])
            rows.append(["Loss Incurred"] + base + [float(cum_loss[d])])
            rows.append(["Exposure"] + base + [float(cum_exp[d])])
            rows.append(["Earned Premium"] + base + [float(cum_prem[d])])

    df = pd.DataFrame(
        rows,
        columns=[
            "concept",
            "basis",
            "amount_type",
            "level_1",
            "level_2",
            "level_3",
            "accident_period",
            "development_period",
            "amount",
        ],
    )
    return df


def _logistic(x: np.ndarray) -> np.ndarray:
    """Element-wise logistic (sigmoid) function."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
