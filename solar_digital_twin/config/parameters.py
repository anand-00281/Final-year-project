from dataclasses import dataclass
import numpy as np

@dataclass
@dataclass
class SimulationParameters:
    output_dt_s: float = 1.0
    internal_dt_s: float = 0.05
    internal_dt: float = 0.05     # Alias to prevent any attribute errors
    seed: int = 42

@dataclass
class WeatherParameters:
    peak_irradiance: float = 1000.0
    noise_std: float = 15.0
    base_temp: float = 25.0
    temp_amplitude: float = 12.0         # Celsius

@dataclass
class PVParameters:
    p_max: float = 3300.0       # Watts
    v_mp: float = 372.0         # Volts at max power
    i_mp: float = 8.88          # Amps at max power
    temp_coeff_p: float = -0.004# Power loss per degree C

@dataclass
class MotorParameters:
    r_m: float = 1.5        # Armature resistance (ohms)
    l_m: float = 0.01       # Armature inductance (H)
    k_t: float = 0.8        # Torque constant (Nm/A)
    k_e: float = 0.8        # Back-EMF constant (V/(rad/s))
    j: float = 0.05         # Rotor inertia (kg*m^2)
    b_m: float = 0.001      # Motor viscous friction coefficient

@dataclass
class PumpParameters:
    k1: float = 0.008     # Increased head coefficient for demo scaling
    k2: float = 0.00050
    k3: float = 0.00052
    eta_pump_base: float = 0.60

@dataclass
class HydraulicParameters:
    h_static: float = 2.0       # Set to 2 meters for easy startup flow
    k_system: float = 0.003
    density: float = 1000.0
    gravity: float = 9.81

@dataclass
class FaultParameters:
    # Modifiers that alter the physics (0.0 = healthy)
    pv_degradation_severity: float = 0.0
    bearing_wear_severity: float = 0.0
    impeller_blockage_severity: float = 0.0
    is_dry_running: bool = False

@dataclass
class SensorParameters:
    # Default moderate Gaussian noise levels
    voltage_noise_std: float = 0.5
    current_noise_std: float = 0.1
    rpm_noise_std: float = 5.0
    flow_noise_std: float = 2.0
    head_noise_std: float = 0.5

# Global Random Number Generator for strict reproducibility
sim_params = SimulationParameters()
rng = np.random.default_rng(sim_params.seed)

# --- Quick Test Block ---
if __name__ == "__main__":
    sim = SimulationParameters()
    motor = MotorParameters()
    print(f"Simulation Seed: {sim.seed}")
    print(f"Motor Resistance: {motor.r_m} Ohms")
    print(f"Controlled Random Number: {rng.random():.4f}")