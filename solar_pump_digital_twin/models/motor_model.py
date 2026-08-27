"""
models/motor_model.py

Responsible for:
    - friction and churn torque, and the bearing_wear / impeller_blockage
      (secondary) modifiers to B_m                        (PART C.8, C.9-ish)
    - electromagnetic torque                               (PART C.7)
    - the single mechanical ODE integration step            (PART C.14)

Does NOT know about hydraulics directly -- it receives T_hyd as an input
(computed by hydraulic_model.py) and simply adds it into the load-torque
balance.
"""

from solar_pump_digital_twin.faults.fault_models import (
    bearing_wear_Bm_multiplier,
    blockage_Bm_secondary_addend,
)


def friction_torque(omega: float, B_m: float) -> float:
    """T_friction = B_m * omega. Units: N*m."""
    return B_m * omega


def churn_torque(omega: float, C_churn: float) -> float:
    """T_churn = C_churn * omega^2 -- idle hydraulic churn/windage loss.
    Coefficient is a named, configurable parameter (C_churn), decoupled
    from fluid density (which is now always constant -- Decision 2)."""
    return C_churn * omega**2


def effective_Bm(motor_params, fault_params, wear_severity: float = 0.0, blockage_severity: float = 0.0) -> float:
    """B_m = B_m_base * [ (1 + bearing_wear_gain*wear_severity)
                            + blockage_secondary_gain*blockage_severity ]

    bearing_wear is the PRIMARY driver of this term; impeller_blockage's
    contribution here is deliberately SECONDARY (smaller gain) per
    Decision 3 -- its primary effect is hydraulic, applied in
    pump_model.py's pump_head_coefficients()."""
    mult = bearing_wear_Bm_multiplier(wear_severity, fault_params.bearing_wear_Bm_gain)
    addend = blockage_Bm_secondary_addend(blockage_severity, fault_params.blockage_Bm_secondary_gain)
    return motor_params.B_m_base * (mult + addend)


def electromagnetic_torque(I_dc: float, motor_params) -> float:
    """T_elec = K_t * I_dc, with K_t = K_e (PART C.7, standard PMDC
    simplification in consistent SI units)."""
    return motor_params.K_e * I_dc


def integrate_omega(omega: float, T_elec: float, T_load: float, motor_params, dt_s: float) -> float:
    """J * domega/dt = T_elec - T_load, explicit Euler at dt_s (PART C.14).

    Numerical-stability note: at dt_s = 1s, accuracy depends on the
    mechanical time constant tau_mech ~= J / B_m,eff. This should be
    checked numerically (validation/physics_checks.py) rather than
    assumed; SimulationParameters.internal_dt_s can be reduced below
    output_dt_s as a fallback if a given parameter set proves unstable
    at 1s steps."""
    domega_dt = (T_elec - T_load) / motor_params.J
    omega_new = omega + domega_dt * dt_s
    return max(0.0, omega_new)
