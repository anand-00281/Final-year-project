from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationParameters:
    internal_dt: float = 1.0  # 1-second internal physics resolution
    output_dt: float = 1.0    # 1-second telemetry output (1-Hz)
    seed: int = 42            # Strict reproducibility

@dataclass
class WeatherParameters:
    peak_irradiance: float = 1000.0  # W/m2
    base_temp: float = 15.0          # Celsius
    peak_temp: float = 35.0          # Celsius

@dataclass
class PVParameters:
    p_max: float = 3300.0       # Watts
    v_mp: float = 372.0         # Volts at max power
    i_mp: float = 8.88          # Amps at max power
    temp_coeff_p: float = -0.004# Power loss per degree C

@dataclass
class MotorParameters:
    r_m: float = 1.5            # Stator resistance (Ohms)
    k_e: float = 0.8            # Back-EMF constant (V/(rad/s))
    k_t: float = 0.8            # Torque constant (Nm/A)
    j_rotor: float = 0.05       # Rotor inertia (kg*m2)
    b_m_base: float = 0.01      # Base mechanical friction

@dataclass
class PumpParameters:
    # Reduced-order coefficients for H_pump = K1*w^2 - K2*w*Q - K3*Q^2
    k1: float = 0.015
    k2: float = 0.002
    k3: float = 0.001
    
@dataclass
class HydraulicParameters:
    h_static: float = 30.0      # Static head (meters)
    k_system: float = 0.005     # Pipe friction/system resistance
    density: float = 1000.0     # kg/m3
    gravity: float = 9.81       # m/s2

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