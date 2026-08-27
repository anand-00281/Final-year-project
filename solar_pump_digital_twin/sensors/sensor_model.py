"""
sensors/sensor_model.py

Responsible ONLY for converting ideal simulated states into measured
telemetry. It must NEVER change the underlying physics -- it reads the
ideal-state columns and produces a separate measured-telemetry
DataFrame; it never mutates its input and is always the LAST step in
the simulation pipeline (weather -> PV -> motor -> pump -> hydraulics ->
faults -> sensors -> measured telemetry).
"""

from typing import Dict

import numpy as np
import pandas as pd


def _apply_channel(values: np.ndarray, cfg, rng: np.random.Generator) -> np.ndarray:
    """Apply bias -> drift -> noise -> quantization -> missingness, in that
    order, to one channel. Every effect is a no-op at its default
    (magnitude/probability 0)."""
    x = np.asarray(values, dtype=float).copy()
    n = len(x)

    # constant bias (fractional then absolute)
    x = x * (1.0 + cfg.bias_frac) + cfg.bias_abs

    # slow drift: a random walk normalized so its max |excursion| over the
    # run equals drift_max (keeps drift bounded and configurable in
    # physical units rather than as an unbounded random-walk variance).
    if cfg.drift_max > 0:
        steps = rng.normal(0.0, 1.0, size=n)
        walk = np.cumsum(steps)
        peak = np.max(np.abs(walk))
        if peak > 0:
            walk = walk / peak * cfg.drift_max
        x = x + walk

    # measurement noise
    if cfg.noise_std > 0:
        x = x + rng.normal(0.0, cfg.noise_std, size=n)

    # quantization
    if cfg.quantization_step:
        x = np.round(x / cfg.quantization_step) * cfg.quantization_step

    # missingness (applied last, so it doesn't get "fixed" by later steps)
    if cfg.missing_prob > 0:
        mask = rng.uniform(size=n) < cfg.missing_prob
        x[mask] = np.nan

    return x


def apply_sensor_model(ideal_df: pd.DataFrame, sensor_params, hydraulic_params, seed: int) -> pd.DataFrame:
    """Ideal physical state -> measured telemetry (PART C.18).

    Reads the *_ideal columns; NEVER writes back into ideal_df. Power and
    efficiency are recomputed FROM the already-noised base channels
    (voltage, current, flow, head) rather than independently noised, so
    the measured dataset stays internally consistent (P = V*I really
    holds for the measured columns, not just the ideal ones).
    """
    rng = np.random.default_rng(seed)
    measured: Dict[str, np.ndarray] = {}

    measured["Irradiance_W_m2"] = _apply_channel(ideal_df["Irradiance_W_m2_ideal"].to_numpy(), sensor_params.irradiance, rng)
    measured["DC_Voltage_V"] = _apply_channel(ideal_df["DC_Voltage_V_ideal"].to_numpy(), sensor_params.dc_voltage, rng)
    measured["DC_Current_A"] = _apply_channel(ideal_df["DC_Current_A_ideal"].to_numpy(), sensor_params.dc_current, rng)
    measured["Motor_RPM"] = _apply_channel(ideal_df["Motor_RPM_ideal"].to_numpy(), sensor_params.rpm, rng)
    measured["Flow_Rate_LPM"] = _apply_channel(ideal_df["Flow_Rate_LPM_ideal"].to_numpy(), sensor_params.flow, rng)
    measured["Pressure_Head_m"] = _apply_channel(ideal_df["Pressure_Head_m_ideal"].to_numpy(), sensor_params.head, rng)

    measured["DC_Power_W"] = measured["DC_Voltage_V"] * measured["DC_Current_A"]
    Q_m3s = measured["Flow_Rate_LPM"] / 60000.0
    measured["Hydraulic_Power_W"] = hydraulic_params.rho_water * hydraulic_params.g * Q_m3s * measured["Pressure_Head_m"]
    with np.errstate(divide="ignore", invalid="ignore"):
        # Guard against dividing two independently-noised near-zero quantities
        # (e.g. at night: tiny noise-driven DC power alongside a small
        # noise-driven "phantom" hydraulic power computed against the
        # legitimate static-head reading) -- see efficiency_min_power_W
        # docstring in config/parameters.py for the empirical motivation.
        eff = np.where(
            measured["DC_Power_W"] > sensor_params.efficiency_min_power_W,
            measured["Hydraulic_Power_W"] / measured["DC_Power_W"],
            0.0,
        )
    measured["Efficiency_Proxy"] = eff

    return pd.DataFrame(measured, index=ideal_df.index)
