"""
models/hydraulic_model.py

Responsible for:
    - the system (pipe) curve                          (PART C.12)
    - the pump/system operating-point intersection      (PART C.13)
    - hydraulic power                                   (PART C.15)
    - hydraulic torque (shaft power required)           (PART C.8)

Does NOT know about faults directly -- it consumes pump-curve coefficients
that models/pump_model.py has already modified for blockage/dry-running,
and its own K_system parameter is untouched by any fault in V2 (Decision 3).
"""

import numpy as np


def system_head(Q: float, hydraulic_params) -> float:
    """H_system = H_static + K_system * Q^2  (PART C.12). Q in m^3/s, H in m.
    Assumption: turbulent-flow pipe/fitting losses (~Q^2), lumped into one
    coefficient -- a standard reduced-order simplification."""
    return hydraulic_params.H_static + hydraulic_params.K_system * Q**2


def solve_operating_point(omega: float, K1_eff: float, K2: float, K3_eff: float, hydraulic_params):
    """Solve H_pump(omega,Q) = H_system(Q) for the physically valid,
    non-negative flow Q (m^3/s). Algebra unchanged from V1 (PART A.1 /
    C.13): rearranging to a quadratic in Q,

        -(K3_eff + K_system)*Q^2 - K2*omega*Q + (K1_eff*omega^2 - H_static) = 0

    Returns (Q, H). If no positive-head/positive-discriminant solution
    exists (e.g. omega too low, or K1_eff collapsed toward 0 under dry
    running), Q resolves to 0 and H is reported as H_static -- the
    physically correct "no-flow / shutoff" condition."""
    a = -(K3_eff + hydraulic_params.K_system)
    b = -K2 * omega
    c = K1_eff * omega**2 - hydraulic_params.H_static

    Q = 0.0
    if omega > 0.0 and c > 0.0:
        disc = b**2 - 4.0 * a * c
        if disc >= 0.0:
            Q_root = (-b - np.sqrt(disc)) / (2.0 * a)
            Q = max(0.0, Q_root)

    H = system_head(Q, hydraulic_params) if Q > 0.0 else hydraulic_params.H_static
    return Q, H


def hydraulic_power(Q: float, H: float, hydraulic_params) -> float:
    """P_hyd = rho_water * g * Q * H  (PART C.15). Q in m^3/s, H in m. Returns W.
    rho_water is the constant physical value -- NEVER fault-modified (Decision 2)."""
    return hydraulic_params.rho_water * hydraulic_params.g * Q * H


def hydraulic_torque(Q: float, H: float, omega: float, hydraulic_params, pump_params) -> float:
    """T_hyd = rho*g*H*Q / (eta_pump*omega) for omega>0, else 0 (PART C.8) --
    the shaft torque required to deliver hydraulic power P_hyd = T*omega
    through the pump at efficiency eta_pump. Unchanged from V1."""
    if omega <= 0.0:
        return 0.0
    return (hydraulic_params.rho_water * hydraulic_params.g * H * Q) / (pump_params.eta_pump * omega)
