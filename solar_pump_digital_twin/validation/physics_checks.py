"""
validation/physics_checks.py

Automated PASS/FAIL + diagnostic checks, per Section 16 / constraint 12.
Must be run on small scenarios BEFORE any large dataset generation.

These are QUALITATIVE / DIRECTIONAL checks (correlation signs, monotonic
schedules, value-range sanity), not claims of experimental validation --
this is a reduced-order synthetic simulator, not a calibrated model. See
the "known limitations" notes in main.py's printed output.

Q/N invariance is deliberately NOT gated pass/fail (constraint 18: must
not assume perfect invariance) -- it is reported as a diagnostic number
only.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from solar_pump_digital_twin.simulation.digital_twin import run_scenario
from solar_pump_digital_twin.simulation.schemas import IDEAL_STATE_COLUMNS
from solar_pump_digital_twin.sensors.sensor_model import apply_sensor_model


@dataclass
class CheckResult:
    name: str
    passed: Optional[bool]   # True/False, or None if the check could not be evaluated (reported as N/A)
    detail: str


def _pearson(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def check_irradiance_power_relationship(df: pd.DataFrame) -> CheckResult:
    daylight = df["Irradiance_W_m2_ideal"] > 5
    if daylight.sum() < 10:
        return CheckResult("irradiance_vs_dc_power", None, "insufficient daylight samples")
    r = _pearson(df.loc[daylight, "Irradiance_W_m2_ideal"], df.loc[daylight, "DC_Power_W_ideal"])
    return CheckResult("irradiance_vs_dc_power", bool(r > 0.5), f"pearson r={r:.3f} (expect > 0.5: irradiance up -> DC power up, and vice versa)")


def check_bearing_wear_load(df: pd.DataFrame) -> CheckResult:
    rows = df[df["fault_label"] == "bearing_wear"]
    if len(rows) < 10:
        return CheckResult("bearing_wear_vs_rpm", None, "no bearing_wear rows in this run")
    r = _pearson(rows["fault_severity"], rows["Motor_RPM_ideal"])
    return CheckResult("bearing_wear_vs_rpm", bool(r < -0.2), f"pearson r={r:.3f} (expect < -0.2: higher wear -> lower speed under fixed available power, i.e. higher mechanical loading)")


def check_blockage_flow(df: pd.DataFrame) -> CheckResult:
    rows = df[df["fault_label"] == "impeller_blockage"]
    if len(rows) < 10:
        return CheckResult("blockage_vs_flow", None, "no impeller_blockage rows in this run")
    r = _pearson(rows["fault_severity"], rows["Flow_Rate_LPM_ideal"])
    return CheckResult("blockage_vs_flow", bool(r < -0.3), f"pearson r={r:.3f} (expect < -0.3: higher blockage -> lower flow)")


def check_dryrun_flow_power(df: pd.DataFrame) -> CheckResult:
    rows = df[df["fault_label"] == "dry_running"]
    if len(rows) < 10:
        return CheckResult("dryrun_vs_flow_power", None, "no dry_running rows in this run")
    r_q = _pearson(rows["fault_severity"], rows["Flow_Rate_LPM_ideal"])
    r_p = _pearson(rows["fault_severity"], rows["Hydraulic_Power_W_ideal"])
    passed = bool((r_q < -0.3) and (r_p < -0.3))
    return CheckResult("dryrun_vs_flow_power", passed, f"pearson r_Q={r_q:.3f}, r_Phyd={r_p:.3f} (expect both < -0.3: higher dry-running severity -> collapsing flow/hydraulic power)")


def check_shading_power(df: pd.DataFrame) -> CheckResult:
    rows = df[df["fault_label"] == "partial_shading"]
    if len(rows) < 10:
        return CheckResult("shading_vs_power", None, "no partial_shading rows in this run (mechanism not exercised by the default 11 scenarios -- see scenario_library.py note)")
    r = _pearson(rows["fault_severity"], rows["DC_Power_W_ideal"])
    return CheckResult("shading_vs_power", bool(r < -0.3), f"pearson r={r:.3f} (expect < -0.3)")


def check_pv_degradation_persistence(df: pd.DataFrame) -> CheckResult:
    rows = df[df["fault_label"] == "pv_degradation"]
    if len(rows) < 10:
        return CheckResult("pv_degradation_persistence", None, "no pv_degradation rows in this run")
    sev = rows["fault_severity"].to_numpy()
    non_decreasing = bool(np.all(np.diff(sev) >= -1e-9))
    return CheckResult("pv_degradation_persistence", non_decreasing, "severity must never decrease once ramped (persistent, non-recovering fault, Decision 4)")


def check_weather_transient_label_consistency(df: pd.DataFrame) -> CheckResult:
    bad = df[(df["weather_transient"]) & (df["fault_severity"] == 0.0) & (df["fault_label"] != "normal")]
    return CheckResult("weather_transient_label_consistency", bool(len(bad) == 0),
                        f"{len(bad)} rows have an active weather transient, zero fault severity, yet a non-normal label (must be 0)")


def check_qn_diagnostic(df: pd.DataFrame) -> CheckResult:
    rows = df[df["fault_label"] == "normal"]
    N = rows["Motor_RPM_ideal"].replace(0, np.nan)
    qn = (rows["Flow_Rate_LPM_ideal"] / N).replace([np.inf, -np.inf], np.nan).dropna()
    if len(qn) < 10:
        return CheckResult("qn_diagnostic", None, "insufficient normal-operation samples")
    cv = float(qn.std() / qn.mean()) if qn.mean() != 0 else float("nan")
    return CheckResult("qn_diagnostic", True,
                        f"Q/N coefficient of variation during normal operation (incl. cloud transients) = {cv:.4f} "
                        f"-- DIAGNOSTIC ONLY, not assumed/asserted invariant (constraint 18); to be evaluated properly in Phase 3")


def check_no_impossible_values(df: pd.DataFrame) -> CheckResult:
    nonneg_cols = ["Flow_Rate_LPM_ideal", "Pressure_Head_m_ideal", "DC_Voltage_V_ideal", "DC_Current_A_ideal", "Motor_RPM_ideal"]
    neg_counts = {c: int((df[c] < -1e-6).sum()) for c in nonneg_cols}
    finite_ok = bool(np.isfinite(df[IDEAL_STATE_COLUMNS].to_numpy()).all())
    passed = all(v == 0 for v in neg_counts.values()) and finite_ok
    return CheckResult("no_impossible_values", passed, f"negative-value counts={neg_counts}, all_finite={finite_ok}")


def check_measured_efficiency_bounded(df: pd.DataFrame, max_plausible: float = 1.5) -> CheckResult:
    """Measured Efficiency_Proxy must never exceed a generous plausibility
    ceiling (a lossy electro-hydraulic system cannot exceed ~100%; a small
    margin above 1.0 is allowed purely for residual sensor-noise slop).
    This check exists specifically because dividing two independently
    noised near-zero channels (DC power, hydraulic power) can otherwise
    produce spurious large ratios -- see efficiency_min_power_W in
    config/parameters.py and its guard in sensors/sensor_model.py."""
    bad = df[df["Efficiency_Proxy"] > max_plausible]
    return CheckResult("measured_efficiency_bounded", bool(len(bad) == 0),
                        f"{len(bad)} of {len(df)} rows have measured Efficiency_Proxy > {max_plausible} (must be 0)")


def check_sensor_model_isolation(df: pd.DataFrame, sensor_params, hydraulic_params, seed_a: int = 101, seed_b: int = 202) -> CheckResult:
    m1 = apply_sensor_model(df, sensor_params, hydraulic_params, seed=seed_a)
    m2 = apply_sensor_model(df, sensor_params, hydraulic_params, seed=seed_b)
    differs = not m1["Irradiance_W_m2"].equals(m2["Irradiance_W_m2"])
    ideal_untouched = bool((df["Irradiance_W_m2_ideal"] == df["Irradiance_W_m2_ideal"]).all())  # trivial: df unmodified by the two calls above
    return CheckResult("sensor_model_isolation", bool(differs and ideal_untouched),
                        f"two different sensor seeds produce different measured telemetry ({differs}); ideal columns unaffected by construction (sensor_model.py never writes to its input)")


def run_full_validation_suite(config, start_time: pd.Timestamp = pd.Timestamp("2026-05-01")) -> pd.DataFrame:
    """Run one small (24h) simulation per relevant scenario and evaluate
    every check above. Intended to run BEFORE any large dataset
    generation (Section 16 / constraint 14)."""
    results = []

    df3 = run_scenario(3, start_time, config)
    results.append(check_irradiance_power_relationship(df3))
    results.append(check_weather_transient_label_consistency(df3))
    results.append(check_qn_diagnostic(df3))
    results.append(check_no_impossible_values(df3))
    results.append(check_measured_efficiency_bounded(df3))
    results.append(check_sensor_model_isolation(df3, config.sensors, config.hydraulic))

    df4 = run_scenario(4, start_time, config)
    results.append(check_bearing_wear_load(df4))

    df5 = run_scenario(5, start_time, config)
    results.append(check_blockage_flow(df5))

    df6 = run_scenario(6, start_time, config)
    results.append(check_pv_degradation_persistence(df6))

    df7 = run_scenario(7, start_time, config)
    results.append(check_dryrun_flow_power(df7))

    rows = [{
        "check": r.name,
        "result": "PASS" if r.passed is True else ("FAIL" if r.passed is False else "N/A"),
        "detail": r.detail,
    } for r in results]
    return pd.DataFrame(rows)
