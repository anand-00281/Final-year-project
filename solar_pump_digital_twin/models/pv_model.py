"""
models/pv_model.py

Responsible for:
    - PV available electrical power                (PART C.5)
    - PV-side fault modifiers: partial_shading vs. pv_degradation (PART C.4)
    - the self-consistent DC bus solve that replaces V1's fixed V_bus
      model and produces genuinely time-varying V_dc, I_dc, P_dc (PART C.6)

Does NOT know about fault SCHEDULES (start/end times) -- it only ever
receives an instantaneous severity value in [0,1] for each PV-side fault.
"""

import numpy as np


def effective_pv_area_fraction(shading_severity: float, degradation_severity: float) -> float:
    """A_pv_eff / A_pv = (1 - shading) * (1 - degradation).

    Mechanistically distinct (Decision 4 / 5):
        shading      -- transient, resolves within the scenario (a physical
                         obstruction of the array, e.g. debris, temporary cover)
        degradation  -- persistent/slowly varying, does not resolve within
                         the scenario (panel aging/damage)
    Both act on effective collector area/output in the same functional
    form; what differs is only their TIME SCHEDULE (owned by
    faults/fault_models.py), not their instantaneous physical effect.
    """
    return (1.0 - shading_severity) * (1.0 - degradation_severity)


def pv_available_power(G: float, pv_params, shading_severity: float = 0.0, degradation_severity: float = 0.0) -> float:
    """P_pv_avail = G * A_pv * area_fraction * eta_pv   (PART C.5). Units: W.

    Assumption: linear irradiance-to-power scaling with a fixed
    conversion efficiency (implicit MPPT tracking the array's own I-V
    curve); no explicit PV I-V curve or temperature derating (deferred,
    Decision 5)."""
    area_frac = effective_pv_area_fraction(shading_severity, degradation_severity)
    return G * pv_params.A_pv * area_frac * pv_params.eta_pv


def solve_dc_operating_point(P_pv_avail: float, omega: float, pv_params, motor_params):
    """Self-consistent DC bus solve (PART C.6) -- REPLACES V1's fixed
    V_bus model.

    Physical setup:
        P_dc = eta_conv * P_pv_avail       (converter delivers this power to the bus)
        V_dc = E_back + I_dc * R_m          (motor terminal equation, E_back = K_e*omega)
        P_dc = V_dc * I_dc

    Substituting gives a quadratic in I_dc:
        R_m * I_dc^2 + E_back * I_dc - P_dc = 0
        I_dc = [-E_back + sqrt(E_back^2 + 4*R_m*P_dc)] / (2*R_m)

    Assumptions (flagged, need future calibration):
        (1) the converter/MPPT stage transfers array power to the bus at
            a fixed efficiency eta_conv (default 0.95, configurable,
            NOT a primary experimental variable -- Decision 1);
        (2) electrical dynamics are fast relative to the 1s integration
            step, so this is solved quasi-statically each step (same
            time-scale-separation assumption already present in V1);
        (3) no explicit PV I-V curve is invented -- deliberately avoided
            per the instruction not to hallucinate unnecessary physics.

    Returns: (V_dc, I_dc, P_dc) in (V, A, W).
    """
    if P_pv_avail < pv_params.P_cutin_W:
        return 0.0, 0.0, 0.0

    P_dc = pv_params.eta_conv * P_pv_avail
    E_back = motor_params.K_e * omega
    R_m = motor_params.R_m

    disc = E_back**2 + 4.0 * R_m * P_dc
    I_dc = (-E_back + np.sqrt(max(disc, 0.0))) / (2.0 * R_m)
    I_dc = max(I_dc, 0.0)
    V_dc = E_back + I_dc * R_m
    return V_dc, I_dc, P_dc
