"""
config/parameters.py

Central configuration for the Solar Pump Digital Twin (V2).

Design principles:
    - One dataclass per physical subsystem (PV, motor, pump, hydraulic),
      per the agreed modular architecture.
    - Simulation-level and cross-cutting concerns (timing, seeds,
      weather, sensors, fault-mechanism gains) get their own dataclasses.
    - Parameter provenance is tracked explicitly in PARAMETER_PROVENANCE
      so that no numeric value is ever presented as more trustworthy
      than it is. As of V2, EVERY parameter is either 'assumed' or
      'synthetically_calibrated' -- NONE are 'literature_based' or
      'datasheet_based' in the sense of being traceable to a specific
      real pump/motor/PV datasheet. (rho_water and g are 'literature_based'
      as physical constants, which is a different thing from a
      component-specific datasheet value.)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

PROVENANCE_TAGS = (
    "literature_based",
    "datasheet_based",
    "assumed",
    "synthetically_calibrated",
)


@dataclass
class PVParameters:
    A_pv: float = 10.0          # m^2, effective PV collector area
    eta_pv: float = 0.18        # dimensionless, panel -> DC-bus conversion efficiency (implicit MPPT)
    eta_conv: float = 0.95      # dimensionless, converter/MPPT-stage efficiency (array power -> bus power)
    P_cutin_W: float = 10.0     # W, minimum available PV power below which the motor draws no current


@dataclass
class MotorParameters:
    J: float = 0.05             # kg*m^2, rotor + coupled load inertia
    K_e: float = 1.2            # V*s/rad, back-EMF constant. K_t = K_e assumed (standard PMDC
                                 # simplification in consistent SI units) -- see PART C.7.
    R_m: float = 0.5            # Ohm, motor winding resistance
    B_m_base: float = 0.01      # N*m*s/rad, baseline (healthy-bearing) viscous friction coefficient
    C_churn: float = 0.0001     # N*m*s^2/rad^2, idle hydraulic churn/windage loss coefficient


@dataclass
class PumpParameters:
    # H_pump = K1*omega^2 - K2*omega*Q - K3*Q^2  (SI units: omega rad/s, Q m^3/s, H m)
    K1: float = 0.0015          # shutoff-head coefficient
    K2: float = 0.01            # droop coefficient
    K3: float = 100.0           # internal flow-loss coefficient
    eta_pump: float = 0.60      # dimensionless, pump mechanical -> hydraulic efficiency


@dataclass
class HydraulicParameters:
    rho_water: float = 1000.0   # kg/m^3. CONSTANT. Never modified by any fault mechanism in V2
                                 # (see PART C.9 -- dry running is modeled as loss of hydraulic
                                 # coupling, NOT as a change in fluid density).
    g: float = 9.81             # m/s^2
    H_static: float = 20.0      # m, static lift
    K_system: float = 500000.0  # system-curve (pipe/fitting) resistance coefficient.
                                 # NOT modified by impeller_blockage in V2 (Decision 3) --
                                 # reserved for a possible future pipe-fouling fault type.


@dataclass
class WeatherParameters:
    G_max: float = 1000.0               # W/m^2, clear-sky peak irradiance
    sunrise_hour: float = 6.0           # decimal local hour
    sunset_hour: float = 18.0           # decimal local hour
    irradiance_noise_std: float = 15.0  # W/m^2, short-timescale Gaussian variability


@dataclass
class SensorChannelConfig:
    """Configuration for one measured channel's sensor imperfections.
    Every effect is a no-op when its magnitude/probability is 0, so a
    channel can be made "perfect" by leaving all fields at default 0.
    """
    noise_std: float = 0.0                       # Gaussian measurement noise, absolute channel units
    bias_abs: float = 0.0                        # constant additive offset, absolute units
    bias_frac: float = 0.0                       # constant multiplicative offset (fraction of ideal value)
    drift_max: float = 0.0                       # max |slow random-walk drift| over the run, absolute units
    quantization_step: Optional[float] = None     # round to nearest multiple of this, if set
    missing_prob: float = 0.0                     # probability a given sample is dropped (-> NaN)


@dataclass
class SensorParameters:
    irradiance: SensorChannelConfig = field(default_factory=lambda: SensorChannelConfig(
        noise_std=5.0, quantization_step=1.0, missing_prob=0.001))
    dc_voltage: SensorChannelConfig = field(default_factory=lambda: SensorChannelConfig(
        noise_std=0.3, quantization_step=0.1, missing_prob=0.001))
    dc_current: SensorChannelConfig = field(default_factory=lambda: SensorChannelConfig(
        noise_std=0.05, quantization_step=0.01, missing_prob=0.001))
    rpm: SensorChannelConfig = field(default_factory=lambda: SensorChannelConfig(
        noise_std=2.0, quantization_step=1.0, missing_prob=0.001))
    flow: SensorChannelConfig = field(default_factory=lambda: SensorChannelConfig(
        noise_std=0.5, quantization_step=0.1, missing_prob=0.002))
    head: SensorChannelConfig = field(default_factory=lambda: SensorChannelConfig(
        noise_std=0.1, quantization_step=0.05, missing_prob=0.002))
    # Minimum MEASURED DC power (W) below which Efficiency_Proxy is reported as 0
    # rather than computed as a ratio. Needed because at true-zero operation
    # (e.g. night), independent sensor noise on voltage/current and on
    # flow/head can each be individually tiny but nonzero, and dividing two
    # such near-zero noisy quantities can produce an unbounded/nonsensical
    # ratio (empirically observed: a single noisy 0.3 LPM flow reading
    # against the legitimate H_static reading at night produced a spurious
    # Efficiency_Proxy of ~99). This threshold is set well above the
    # sensor noise floor (dc_voltage.noise_std * dc_current.noise_std is of
    # order 0.01-0.1 W) and well below real daytime operating power
    # (hundreds of W), mirroring how a real instrument/SCADA system would
    # simply not report an efficiency figure below a minimum operating power.
    efficiency_min_power_W: float = 5.0


@dataclass
class FaultMechanismParameters:
    """
    Gains mapping fault severity (0-1) to physical-parameter modifiers.
    ALL values here are 'synthetically_calibrated' or 'assumed' (see
    PARAMETER_PROVENANCE below) -- chosen only to produce a visible,
    bounded, physically-directionally-correct effect. They are NOT
    derived from any real pump/motor datasheet or literature source,
    and must never be presented as experimentally validated.
    """
    # bearing_wear: increases viscous friction. B_m = B_m_base * (1 + gain * severity)
    # NOTE: reduced from the V1 prototype's value of 10.0 to 2.0. Empirical testing during
    # V2 implementation showed the inherited V1 pump-curve defaults put the nominal
    # operating point close to the pump's shutoff head (H_static); a gain of 10.0 pushed
    # flow to EXACTLY zero even at moderate severity, which is a numerically-driven cliff,
    # not a graded fault signature. gain=2.0 was chosen empirically to give a smooth,
    # monotonic RPM/flow/current response across severity 0-1 without collapsing flow to
    # zero (see PART E validation notes). Still purely synthetically_calibrated.
    bearing_wear_Bm_gain: float = 2.0

    # impeller_blockage: PRIMARY effect = added pump-internal loss coefficient (added to K3)
    # NOTE: an initial guess of 1500 (comparable to K3=100) was empirically found to be
    # numerically invisible, because K_system=500,000 already dwarfs K3 by ~5000x, so a
    # +1500 addition to K3 could never meaningfully change the (K3_eff + K_system) term
    # that actually governs the operating point. Values were swept from 1.5e3 to 2e7; 3e6
    # was chosen because it produces a smooth, monotonic, evenly-graded flow reduction
    # across the full severity range (RPM 1210->1434, Flow 172->119 LPM at severity=1.0)
    # without saturating almost immediately (as e.g. 1.5e7 does, where most of the effect
    # appears by severity~0.2). Still purely synthetically_calibrated -- NOT derived from
    # a real pump/blockage measurement.
    blockage_K_block_max: float = 3.0e6
    # impeller_blockage: SECONDARY effect = small extra mechanical drag.
    # B_m += B_m_base * gain * severity. Deliberately smaller than bearing_wear's own gain
    # (Decision 3: blockage's mechanical-drag contribution must remain secondary).
    blockage_Bm_secondary_gain: float = 1.0

    # dry_running: coupling factor kappa(severity) = 1 - (1 - kappa_min) * severity, clipped [kappa_min, 1].
    # kappa_min = 0 => fully decoupled (no shutoff-head capability) at severity = 1.
    dryrun_kappa_min: float = 0.0

    # pv faults (partial_shading, pv_degradation) use severity directly as the fractional
    # PV-area/output reduction -- no extra gain layer, kept deliberately simple/transparent
    # (see PART C.4).


@dataclass
class SimulationParameters:
    output_dt_s: float = 1.0     # seconds per output sample (genuine 1 Hz => 1.0)
    internal_dt_s: float = 0.1   # seconds per internal ODE integration step.
                                  # Must evenly divide output_dt_s (output_dt_s / internal_dt_s
                                  # must be a whole number of sub-steps).
                                  #
                                  # EMPIRICAL FINDING (not a default guess): explicit-Euler
                                  # integration of the mechanical ODE at dt=1.0s DIVERGES for
                                  # the default MotorParameters (J=0.05 kg*m^2) during the
                                  # near-zero-speed startup transient -- the near-unbounded
                                  # startup current (no inductance term limits current rise in
                                  # this quasi-static electrical model) produces a torque spike
                                  # that overshoots to ~1330 rad/s in a single 1s step before
                                  # collapsing back to the omega>=0 clip floor. Verified: 10
                                  # sub-steps/s (dt=0.1s) converges to the same trajectory as
                                  # 100 and 1000 sub-steps/s (126.743 rad/s after 60s from cold
                                  # start at G=900 W/m^2), so dt=0.1s is used as the default.
                                  # This does NOT reduce OUTPUT resolution -- one row is still
                                  # produced per output_dt_s=1.0s; only the internal integration
                                  # is sub-stepped, and no intermediate rows are discarded from
                                  # the exported dataset (contrast with V1, which discarded 59
                                  # of every 60 computed states).
    global_seed: int = 0
    weather_seed: int = 0
    fault_seed: int = 0
    sensor_seed: int = 0


@dataclass
class DigitalTwinConfig:
    pv: PVParameters = field(default_factory=PVParameters)
    motor: MotorParameters = field(default_factory=MotorParameters)
    pump: PumpParameters = field(default_factory=PumpParameters)
    hydraulic: HydraulicParameters = field(default_factory=HydraulicParameters)
    weather: WeatherParameters = field(default_factory=WeatherParameters)
    sensors: SensorParameters = field(default_factory=SensorParameters)
    faults: FaultMechanismParameters = field(default_factory=FaultMechanismParameters)
    simulation: SimulationParameters = field(default_factory=SimulationParameters)


PARAMETER_PROVENANCE: Dict[str, Dict[str, str]] = {
    "PVParameters.A_pv": {"tag": "assumed", "note": "Representative small agricultural PV array size; not a specific product."},
    "PVParameters.eta_pv": {"tag": "assumed", "note": "Typical crystalline-silicon panel efficiency order of magnitude."},
    "PVParameters.eta_conv": {"tag": "assumed", "note": "Representative MPPT/converter efficiency; configurable for sensitivity analysis, not a primary experimental variable (Decision 1)."},
    "PVParameters.P_cutin_W": {"tag": "assumed", "note": "Placeholder controller cut-in threshold; no datasheet source."},
    "MotorParameters.J": {"tag": "assumed", "note": "Representative small PMDC pump-motor rotor inertia."},
    "MotorParameters.K_e": {"tag": "assumed", "note": "Representative torque/back-EMF constant; K_t=K_e assumed (standard PMDC simplification, SI units)."},
    "MotorParameters.R_m": {"tag": "assumed", "note": "Representative winding resistance."},
    "MotorParameters.B_m_base": {"tag": "assumed", "note": "Representative healthy-bearing viscous friction coefficient."},
    "MotorParameters.C_churn": {"tag": "synthetically_calibrated", "note": "Chosen only to give a small, bounded idle churn loss at typical operating speeds; not measured. (Retained from V1's ad hoc term, now a named/configurable parameter and decoupled from density.)"},
    "PumpParameters.K1": {"tag": "assumed", "note": "Reduced-order pump-curve shutoff-head coefficient; example value, not from a pump curve datasheet."},
    "PumpParameters.K2": {"tag": "assumed", "note": "Reduced-order pump-curve droop coefficient; example value."},
    "PumpParameters.K3": {"tag": "assumed", "note": "Reduced-order pump-curve internal-loss coefficient; example value."},
    "PumpParameters.eta_pump": {"tag": "assumed", "note": "Representative small centrifugal pump efficiency."},
    "HydraulicParameters.rho_water": {"tag": "literature_based", "note": "Standard water density at ~room temperature; a physical constant, not a modeling choice."},
    "HydraulicParameters.g": {"tag": "literature_based", "note": "Standard gravitational acceleration."},
    "HydraulicParameters.H_static": {"tag": "assumed", "note": "Representative agricultural borewell/field static lift."},
    "HydraulicParameters.K_system": {"tag": "assumed", "note": "Representative lumped pipe/fitting resistance; example value."},
    "WeatherParameters.G_max": {"tag": "literature_based", "note": "Approximate peak terrestrial solar irradiance under clear sky (~1000 W/m^2, a widely used reference such as STC)."},
    "WeatherParameters.irradiance_noise_std": {"tag": "assumed", "note": "Illustrative short-timescale variability magnitude; not derived from measured irradiance data."},
    "FaultMechanismParameters.bearing_wear_Bm_gain": {"tag": "synthetically_calibrated", "note": "Empirically reduced from the V1 prototype value (10.0) to 2.0 during V2 testing: 10.0 collapsed flow to exactly zero at moderate severity given the inherited near-shutoff pump operating point. 2.0 gives a smooth, monotonic, non-degenerate response across severity 0-1."},
    "FaultMechanismParameters.blockage_K_block_max": {"tag": "synthetically_calibrated", "note": "Empirically calibrated: an initial guess of 1500 was numerically invisible next to K_system=500,000. Swept 1.5e3-2e7; 3e6 gives a smoothly graded (not early-saturating) flow reduction across the full severity range. Requires future calibration against real blockage data."},
    "FaultMechanismParameters.blockage_Bm_secondary_gain": {"tag": "synthetically_calibrated", "note": "Deliberately smaller than the primary hydraulic effect and than bearing_wear_Bm_gain, per Decision 3 (secondary effect only)."},
    "FaultMechanismParameters.dryrun_kappa_min": {"tag": "assumed", "note": "Default 0 = fully decoupled at severity=1; simplest bounded/monotonic assumption, flagged for future calibration against real dry-run behaviour."},
    "SensorParameters.*": {"tag": "assumed", "note": "All sensor noise/bias/drift/quantization/missingness magnitudes are illustrative defaults, not derived from any specific real sensor's datasheet."},
}
