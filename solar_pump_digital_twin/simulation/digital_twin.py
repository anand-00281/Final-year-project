"""
simulation/digital_twin.py

Orchestration layer ONLY -- contains no physical equations itself.
Wires together, in strict causal order (per the approved architecture):

    weather -> PV available power -> DC operating point -> electromagnetic
    torque -> mechanical dynamics -> pump operating point -> hydraulic
    load -> physical state -> sensor imperfections -> measured telemetry

Never branches on scenario_id -- it only calls
scenarios.scenario_library.build_scenario() and runs whatever comes out,
so adding scenario #12 later requires zero changes here.
"""

from typing import Optional

import numpy as np
import pandas as pd

from solar_pump_digital_twin.weather.weather_generator import generate_weather
from solar_pump_digital_twin.faults.fault_models import build_fault_severity_df, derive_fault_label, FAULT_TYPES
from solar_pump_digital_twin.models.pv_model import pv_available_power, solve_dc_operating_point
from solar_pump_digital_twin.models.pump_model import pump_head_coefficients
from solar_pump_digital_twin.models.hydraulic_model import solve_operating_point, hydraulic_power, hydraulic_torque
from solar_pump_digital_twin.models.motor_model import (
    effective_Bm, friction_torque, churn_torque, electromagnetic_torque, integrate_omega,
)
from solar_pump_digital_twin.sensors.sensor_model import apply_sensor_model
from solar_pump_digital_twin.scenarios.scenario_library import build_scenario
from solar_pump_digital_twin.simulation.schemas import IDEAL_STATE_COLUMNS, validate_schema


def run_scenario(
    scenario_id: int,
    start_time: pd.Timestamp,
    config,
    weather_seed: Optional[int] = None,
    fault_seed: Optional[int] = None,  # reserved: only used if a scenario is ever built from
                                        # sample_random_faults() instead of the declarative
                                        # library; the 11 core scenarios are fully deterministic
                                        # given (scenario_id, start_time), so fault_seed has no
                                        # effect on them. Kept for API symmetry/reproducibility.
    sensor_seed: Optional[int] = None,
    omega_init: float = 0.0,
) -> pd.DataFrame:
    """Run one scenario end-to-end and return the full telemetry DataFrame
    (metadata + labels + ideal state + measured telemetry), validated
    against simulation/schemas.py.
    """
    weather_seed = config.simulation.weather_seed if weather_seed is None else weather_seed
    sensor_seed = config.simulation.sensor_seed if sensor_seed is None else sensor_seed

    times, cloud_specs, fault_events, weather_condition, scenario_name = build_scenario(scenario_id, start_time, config)

    G, weather_transient = generate_weather(times, config.weather, cloud_specs, seed=weather_seed)
    severity_df = build_fault_severity_df(times, fault_events)

    n = len(times)
    internal_dt = config.simulation.internal_dt_s
    sub_steps = int(round(config.simulation.output_dt_s / internal_dt))
    assert sub_steps >= 1, "output_dt_s must be >= internal_dt_s"

    out = {col: np.empty(n, dtype=float) for col in IDEAL_STATE_COLUMNS}
    omega = omega_init

    sev_wear_arr = severity_df["bearing_wear"].to_numpy()
    sev_block_arr = severity_df["impeller_blockage"].to_numpy()
    sev_pvdeg_arr = severity_df["pv_degradation"].to_numpy()
    sev_dry_arr = severity_df["dry_running"].to_numpy()
    sev_shade_arr = severity_df["partial_shading"].to_numpy()

    for i in range(n):
        Gi = float(G[i])
        sev_wear, sev_block = float(sev_wear_arr[i]), float(sev_block_arr[i])
        sev_pvdeg, sev_dry, sev_shade = float(sev_pvdeg_arr[i]), float(sev_dry_arr[i]), float(sev_shade_arr[i])

        for _ in range(sub_steps):
            P_pv_avail = pv_available_power(Gi, config.pv, shading_severity=sev_shade, degradation_severity=sev_pvdeg)
            V_dc, I_dc, P_dc = solve_dc_operating_point(P_pv_avail, omega, config.pv, config.motor)

            K1_eff, K2, K3_eff = pump_head_coefficients(config.pump, config.faults, blockage_severity=sev_block, dryrun_severity=sev_dry)
            Q, H = solve_operating_point(omega, K1_eff, K2, K3_eff, config.hydraulic)
            T_hyd = hydraulic_torque(Q, H, omega, config.hydraulic, config.pump)

            B_m = effective_Bm(config.motor, config.faults, wear_severity=sev_wear, blockage_severity=sev_block)
            T_load = friction_torque(omega, B_m) + churn_torque(omega, config.motor.C_churn) + T_hyd
            T_elec = electromagnetic_torque(I_dc, config.motor)

            omega = integrate_omega(omega, T_elec, T_load, config.motor, internal_dt)

        P_hyd = hydraulic_power(Q, H, config.hydraulic)

        out["Irradiance_W_m2_ideal"][i] = Gi
        out["DC_Voltage_V_ideal"][i] = V_dc
        out["DC_Current_A_ideal"][i] = I_dc
        out["DC_Power_W_ideal"][i] = P_dc
        out["Motor_RPM_ideal"][i] = omega * 30.0 / np.pi
        out["Flow_Rate_LPM_ideal"][i] = Q * 60000.0
        out["Pressure_Head_m_ideal"][i] = H
        out["Hydraulic_Power_W_ideal"][i] = P_hyd
        out["Efficiency_Proxy_ideal"][i] = (P_hyd / P_dc) if P_dc > 0 else 0.0

    ideal_df = pd.DataFrame(out, index=times)
    measured_df = apply_sensor_model(ideal_df, config.sensors, config.hydraulic, seed=sensor_seed)

    fault_label, fault_severity = derive_fault_label(severity_df)

    meta_df = pd.DataFrame({
        "timestamp": times,
        "scenario_id": scenario_name,
        "weather_condition": weather_condition,
        "weather_transient": weather_transient,
        "fault_label": fault_label,
        "fault_severity": fault_severity,
    }, index=times)

    full_df = pd.concat([meta_df, ideal_df, measured_df], axis=1)
    validate_schema(full_df)
    return full_df
