"""
faults/fault_models.py

Responsible for:
    - fault event definitions (what, when, how severe)
    - severity schedules over time (transient trapezoid vs. persistent)
    - the mapping from severity -> physical parameter MODIFIERS

Does NOT apply modifiers to the physics itself -- that happens inside the
relevant models/*.py file (e.g. models/pump_model.py applies
kappa_dryrun() and blockage_K_block() to the pump curve; motor_model.py
applies bearing_wear_Bm_multiplier() and blockage_Bm_secondary_addend()
to B_m). This keeps "what does a fault mean numerically" (here) separate
from "how does that number enter the ODE" (models/).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from solar_pump_digital_twin.simulation.schemas import FAULT_LABELS

FAULT_TYPES = tuple(f for f in FAULT_LABELS if f != "normal")


@dataclass
class FaultEvent:
    """A single scheduled fault occurrence.

    fault_type     : one of FAULT_TYPES
    start_time     : pd.Timestamp, fault onset (ramp begins here)
    end_time       : pd.Timestamp, fault offset for a 'transient' event
                      (ramp-down completes at/near here); IGNORED for a
                      'persistent' event except as the point after which
                      severity is held constant (no ramp-down).
    max_severity   : float in (0, 1]
    severity_form  : 'transient' (ramp-in, hold, ramp-out) or
                      'persistent' (ramp-in, then HOLD -- never ramps
                      down within the scenario). See Decision 4.
    ramp_s         : seconds for the ramp-in (and, if transient, ramp-out).
    """
    fault_type: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    max_severity: float
    severity_form: str = "transient"
    ramp_s: float = 600.0

    def __post_init__(self):
        assert self.fault_type in FAULT_TYPES, f"unknown fault_type {self.fault_type!r}"
        assert self.severity_form in ("transient", "persistent")
        assert 0.0 < self.max_severity <= 1.0


def _seconds_since(times: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    return (times - t0).total_seconds().values


def transient_severity_schedule(times: pd.DatetimeIndex, event: FaultEvent) -> np.ndarray:
    """Trapezoidal ramp-in / hold / ramp-out envelope (generalizes V1's
    get_ramp_severity(), minutes -> configurable seconds)."""
    sev = np.zeros(len(times))
    active = (times >= event.start_time) & (times <= event.end_time)
    if not active.any():
        return sev

    elapsed = _seconds_since(times, event.start_time)
    remaining = (event.end_time - times).total_seconds().values

    ramp = max(event.ramp_s, 1e-9)
    frac = np.ones(len(times))
    frac = np.where(elapsed < ramp, elapsed / ramp, frac)
    frac = np.where(remaining < ramp, np.minimum(frac, remaining / ramp), frac)
    frac = np.clip(frac, 0.0, 1.0)

    sev = np.where(active, event.max_severity * frac, 0.0)
    return sev


def persistent_severity_schedule(times: pd.DatetimeIndex, event: FaultEvent) -> np.ndarray:
    """Ramp-in over ramp_s, then HOLD at max_severity for the remainder of
    the scenario (from start_time onward). Never ramps down -- Decision 4:
    PV degradation must not automatically recover within a scenario."""
    sev = np.zeros(len(times))
    active = times >= event.start_time
    if not active.any():
        return sev
    elapsed = _seconds_since(times, event.start_time)
    ramp = max(event.ramp_s, 1e-9)
    frac = np.clip(elapsed / ramp, 0.0, 1.0)
    sev = np.where(active, event.max_severity * frac, 0.0)
    return sev


def severity_timeseries(times: pd.DatetimeIndex, event: FaultEvent) -> np.ndarray:
    if event.severity_form == "transient":
        return transient_severity_schedule(times, event)
    return persistent_severity_schedule(times, event)


def build_fault_severity_df(times: pd.DatetimeIndex, events: Sequence[FaultEvent]) -> pd.DataFrame:
    """Return a DataFrame with one severity column per fault type in
    FAULT_TYPES (0.0 where inactive). If multiple events of the SAME
    fault_type overlap, severities are summed then clipped to
    max_severity across events (kept simple; V2's scenario library does
    not currently script overlapping same-type events)."""
    cols = {ft: np.zeros(len(times)) for ft in FAULT_TYPES}
    for ev in events:
        cols[ev.fault_type] = np.clip(cols[ev.fault_type] + severity_timeseries(times, ev), 0.0, 1.0)
    return pd.DataFrame(cols, index=times)


def derive_fault_label(severity_df: pd.DataFrame):
    """Ground-truth label derivation: whichever fault type has the highest
    severity at a given row wins; 'normal' if all are zero. V2's
    scenario library is designed so at most one equipment fault is
    active at a time (Section 7's taxonomy), so this reduces to "the
    active fault, or normal" -- the argmax logic is a general-purpose
    safeguard, not a claim that faults are guaranteed disjoint."""
    cols = list(severity_df.columns)
    values = severity_df[cols].to_numpy()
    max_idx = values.argmax(axis=1)
    max_val = values.max(axis=1)
    labels = np.where(max_val > 0.0, np.array(cols)[max_idx], "normal")
    return labels, max_val


# ---------------------------------------------------------------------
# Severity -> physical-modifier mapping functions.
# Each is simple, bounded, monotonic, and deterministic (per Decision 2).
# ---------------------------------------------------------------------

def bearing_wear_Bm_multiplier(severity: np.ndarray, gain: float) -> np.ndarray:
    """B_m multiplier contribution from bearing wear: 1 + gain*severity."""
    return 1.0 + gain * severity


def blockage_Bm_secondary_addend(severity: np.ndarray, gain: float) -> np.ndarray:
    """B_m multiplier ADDEND from blockage's secondary mechanical drag
    (kept separate from bearing_wear's multiplier so the two compose
    additively rather than compounding multiplicatively)."""
    return gain * severity


def blockage_K_block(severity: np.ndarray, K_block_max: float) -> np.ndarray:
    """Primary blockage effect: internal pump-loss coefficient added to K3.
    Linear, bounded [0, K_block_max], monotonic in severity."""
    return K_block_max * severity


def kappa_dryrun(severity: np.ndarray, kappa_min: float) -> np.ndarray:
    """Hydraulic coupling/priming factor for dry running (PART C.9).
    kappa=1 at severity=0 (fully primed), kappa=kappa_min at severity=1.
    Linear, bounded [kappa_min, 1], monotonic, deterministic -- the
    simplest defensible mapping; flagged for future calibration against
    real dry-run data (see PARAMETER_PROVENANCE)."""
    kappa = 1.0 - (1.0 - kappa_min) * severity
    return np.clip(kappa, kappa_min, 1.0)


# ---------------------------------------------------------------------
# Secondary utility: demoted from V1's primary fault generator.
# Kept for exploratory / robustness-stress-test use only -- the 11 core
# scenarios in scenarios/scenario_library.py do NOT use this.
# ---------------------------------------------------------------------

def sample_random_faults(
    times: pd.DatetimeIndex,
    seed: int,
    fault_types: Optional[Sequence[str]] = None,
    daylight_start_hour: float = 9.0,
    daylight_end_hour: float = 15.0,
    n_events: Optional[int] = None,
) -> List[FaultEvent]:
    """Ported from V1's generate_faults(). Produces an uncontrolled random
    mixture of transient fault events within a daylight window. NOT used
    for the core 11 scenarios (which are fully declarative and
    reproducible) -- retained only as an optional tool for exploratory /
    robustness experiments outside the core dataset."""
    rng = np.random.default_rng(seed)
    fault_types = list(fault_types) if fault_types else [ft for ft in FAULT_TYPES if ft != "partial_shading"]
    hours = times.hour + times.minute / 60.0
    daylight_times = times[(hours >= daylight_start_hour) & (hours <= daylight_end_hour)]
    if len(daylight_times) == 0:
        return []

    total_days = max(1, (times[-1] - times[0]).days)
    n_events = n_events or max(2, total_days)

    events = []
    chosen = rng.choice(fault_types, size=n_events, replace=True)
    for i in range(n_events):
        ft = str(chosen[i])
        start = pd.Timestamp(rng.choice(daylight_times.values))
        duration_s = rng.integers(30, 120) * 60
        end = start + pd.Timedelta(seconds=int(duration_s))
        sev = float(rng.uniform(0.4, 0.9))
        events.append(FaultEvent(ft, start, end, sev, severity_form="transient", ramp_s=600.0))
    return events
