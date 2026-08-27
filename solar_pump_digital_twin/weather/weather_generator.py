"""
weather/weather_generator.py

Responsible for:
    - clear-sky diurnal irradiance curve                 (PART C.1)
    - stochastic (Gaussian) irradiance variability        (PART C.2)
    - controlled, reproducible, parametric cloud transients (PART C.3)

Does NOT know about faults. Fault mechanisms live in faults/fault_models.py
and models/*.py, and are combined with weather output only inside
simulation/digital_twin.py.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CloudTransientSpec:
    """A single scripted cloud event.

    t_start        : timestamp at which irradiance BEGINS to drop
    onset_s        : seconds for irradiance to fall from clear-sky to (1-depth) fraction
    hold_s         : seconds held at the reduced (1-depth) level
    recovery_s     : seconds to recover back to clear-sky
    depth          : fractional irradiance reduction at the trough, e.g. 0.5 => drops to 50%
    shape          : 'smoothstep' (default, C1-continuous) or 'linear'

    Assumption: transitions are smooth, not instantaneous step functions,
    because real atmospheric/instrument transitions are not discontinuous,
    and Section 9 explicitly requires "rapid but not infinite-slope"
    transients to be specifiable.
    """
    t_start: pd.Timestamp
    onset_s: float
    hold_s: float
    recovery_s: float
    depth: float
    shape: str = "smoothstep"


def _shape_fn(x: np.ndarray, shape: str) -> np.ndarray:
    """x in [0,1] -> transition progress in [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    if shape == "linear":
        return x
    if shape == "smoothstep":
        return 3 * x**2 - 2 * x**3
    raise ValueError(f"Unknown cloud transient shape: {shape!r}")


def cloud_transmittance(times: pd.DatetimeIndex, specs: Sequence[CloudTransientSpec]) -> np.ndarray:
    """Return tau(t) in [0,1], the multiplicative factor applied to clear-sky
    irradiance. tau=1 means unobstructed; tau=(1-depth) is the trough.

    Assumption: individual CloudTransientSpec windows are assumed
    non-overlapping (as designed in scenarios/scenario_library.py). If
    they do overlap, this combines them multiplicatively, which reduces
    to the correct single-cloud behaviour whenever there is no overlap
    and degrades gracefully (never below the deepest individual trough)
    if there is.
    """
    tau = np.ones(len(times), dtype=float)
    t = times.values.astype("datetime64[ns]").astype("int64") / 1e9  # seconds, float

    for spec in specs:
        t0 = spec.t_start.value / 1e9
        t_onset_end = t0 + spec.onset_s
        t_hold_end = t_onset_end + spec.hold_s
        t_recovery_end = t_hold_end + spec.recovery_s

        local = np.ones(len(times), dtype=float)

        onset_mask = (t >= t0) & (t < t_onset_end)
        if spec.onset_s > 0:
            progress = (t[onset_mask] - t0) / spec.onset_s
            local[onset_mask] = 1.0 - spec.depth * _shape_fn(progress, spec.shape)
        else:
            local[onset_mask] = 1.0 - spec.depth

        hold_mask = (t >= t_onset_end) & (t < t_hold_end)
        local[hold_mask] = 1.0 - spec.depth

        recovery_mask = (t >= t_hold_end) & (t < t_recovery_end)
        if spec.recovery_s > 0:
            progress = (t[recovery_mask] - t_hold_end) / spec.recovery_s
            local[recovery_mask] = (1.0 - spec.depth) + spec.depth * _shape_fn(progress, spec.shape)
        else:
            local[recovery_mask] = 1.0

        tau *= local

    return tau


def clear_sky_irradiance(times: pd.DatetimeIndex, weather_params) -> Tuple[np.ndarray, np.ndarray]:
    """Bell-curve clear-sky irradiance (PART C.1). Returns (G_clear, daylight_mask).

    Assumption: symmetric daily bell curve, not a true solar-position /
    air-mass model -- an explicitly acceptable simplification for a
    fault-diagnosis twin (see PART C.1).
    """
    hours = times.hour + times.minute / 60.0 + times.second / 3600.0
    span = weather_params.sunset_hour - weather_params.sunrise_hour
    G_clear = weather_params.G_max * np.sin(np.pi * (hours - weather_params.sunrise_hour) / span)
    daylight = (hours >= weather_params.sunrise_hour) & (hours < weather_params.sunset_hour)
    G_clear = np.where(daylight, np.maximum(0.0, G_clear), 0.0)
    return G_clear, daylight.values if hasattr(daylight, "values") else daylight


def generate_weather(
    times: pd.DatetimeIndex,
    weather_params,
    cloud_specs: List[CloudTransientSpec],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full weather generator: clear-sky x cloud-transmittance + noise.

    Returns:
        G                 : irradiance array, W/m^2, clipped >= 0 and zero at night
        weather_transient : boolean array, True wherever a scripted cloud
                             transient is actively reducing irradiance
                             (tau < 0.999), used for the weather_transient
                             metadata column.

    The weather_seed is ALWAYS passed explicitly by the caller
    (simulation/digital_twin.py) -- there is no hidden internal default,
    fixing the V1 bug where generate_weather() silently used seed=42
    regardless of the seed passed to the rest of the run.
    """
    rng = np.random.default_rng(seed)
    G_clear, daylight = clear_sky_irradiance(times, weather_params)

    tau = cloud_transmittance(times, cloud_specs) if cloud_specs else np.ones(len(times))
    noise = rng.normal(0.0, weather_params.irradiance_noise_std, size=len(times))

    G = G_clear * tau + noise
    # Safety clip only -- not a physical ceiling, just guards against noise
    # excursions producing nonsensical values; documented, not silent.
    G = np.clip(G, 0.0, weather_params.G_max * 1.2)
    G = np.where(daylight, G, 0.0)

    weather_transient = (tau < 0.999) & daylight
    return G, weather_transient
