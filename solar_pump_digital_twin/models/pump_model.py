"""
models/pump_model.py

Responsible for:
    - the pump H-Q-omega characteristic                        (PART C.11)
    - where impeller_blockage's PRIMARY effect is applied       (PART C.10)
    - where dry_running's coupling-factor effect is applied     (PART C.9)

Does NOT know about the system curve (that's hydraulic_model.py) or about
fault SCHEDULES -- only instantaneous severities.
"""

from solar_pump_digital_twin.faults.fault_models import kappa_dryrun, blockage_K_block


def pump_head_coefficients(pump_params, fault_params, blockage_severity: float = 0.0, dryrun_severity: float = 0.0):
    """Return effective (K1_eff, K2, K3_eff) pump-curve coefficients after
    applying:

        dry_running (PART C.9)  -- scales K1 (shutoff-head coefficient) by
            kappa(severity) in [kappa_min, 1]. As kappa -> kappa_min, the
            pump loses its ability to generate ANY head, which forces the
            pump/system operating point (see hydraulic_model.py) toward
            Q=0 -- i.e. flow and hydraulic torque collapse. This is a
            direct physical consequence of losing hydraulic coupling, not
            an imposed telemetry value. rho_water is NEVER modified
            (Decision 2).

        impeller_blockage (PART C.10), PRIMARY effect -- adds a
            configurable internal-loss coefficient to K3, shrinking the
            pump's flow-delivery envelope at a given head. This is a
            pump-INTERNAL restriction, not a system(piping)-side one
            (K_system in hydraulic_model.py is untouched by this fault
            in V2 -- Decision 3).
    """
    kappa = kappa_dryrun(dryrun_severity, fault_params.dryrun_kappa_min)
    K1_eff = pump_params.K1 * kappa
    K3_eff = pump_params.K3 + blockage_K_block(blockage_severity, fault_params.blockage_K_block_max)
    return K1_eff, pump_params.K2, K3_eff


def pump_head(omega: float, Q: float, K1_eff: float, K2: float, K3_eff: float) -> float:
    """H_pump = K1_eff*omega^2 - K2*omega*Q - K3_eff*Q^2.
    Units: omega rad/s, Q m^3/s, H m. (PART C.11, unchanged algebra vs V1.)
    """
    return K1_eff * omega**2 - K2 * omega * Q - K3_eff * Q**2
