import sys
import os

# Ensure python can find our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import MotorParameters, FaultParameters, SimulationParameters

class MotorModel:
    def __init__(self, motor_params: MotorParameters, fault_params: FaultParameters, sim_params: SimulationParameters):
        self.params = motor_params
        self.faults = fault_params
        self.dt = sim_params.internal_dt
        
        # The critical dynamic state variable (Angular Velocity in rad/s)
        self.omega_rad_s = 0.0

    def calculate_state(self, v_in: float, t_load: float = 0.0):
        """
        Calculates the dynamic physical state of the motor.
        v_in: Input voltage (from PV/Inverter)
        t_load: Mechanical load torque exerted by the water in the pump
        """
        # 1. INJECT FAULT: Bearing Wear
        # Increases the base mechanical friction based on severity (0.0 to 1.0)
        # We use a multiplier of 10 to simulate severe bearing failure at max severity
        b_effective = self.params.b_m_base * (1.0 + 10.0 * self.faults.bearing_wear_severity)

        # 2. Electrical Equations
        # Calculate Back-EMF (E = Ke * omega)
        e_back = self.params.k_e * self.omega_rad_s
        
        # Calculate Motor Current (I = (V - E) / R)
        i_motor = max(0.0, (v_in - e_back) / self.params.r_m)

        # 3. Mechanical Equations
        # Calculate Electromagnetic Torque (Te = Kt * I)
        t_e = self.params.k_t * i_motor
        
        # Calculate Friction Torque
        t_friction = b_effective * self.omega_rad_s
        
        # 4. Differential Equation for Rotor Dynamics
        # J * (d_omega / dt) = T_e - T_load - T_friction
        net_torque = t_e - t_load - t_friction
        
        # Euler integration to update the dynamic state
        d_omega = (net_torque / self.params.j_rotor) * self.dt
        self.omega_rad_s = max(0.0, self.omega_rad_s + d_omega)

        # 5. Convert to RPM for telemetry
        rpm = self.omega_rad_s * (60.0 / (2 * 3.14159))

        return {
            'motor_current_true': round(i_motor, 3),
            'motor_torque_true': round(t_e, 3),
            'omega_true': round(self.omega_rad_s, 3),
            'rpm_true': round(rpm, 1)
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    mot_p = MotorParameters()
    sim_p = SimulationParameters()
    
    print("Testing Motor Spool-Up Dynamics (Constant 300V Input, No Pump Load)")
    
    # Test Healthy Motor
    fault_healthy = FaultParameters(bearing_wear_severity=0.0)
    motor_healthy = MotorModel(mot_p, fault_healthy, sim_p)
    
    for second in range(1, 6):
        state = motor_healthy.calculate_state(v_in=300.0, t_load=0.0)
        print(f"Sec {second}: {state['rpm_true']} RPM | Current: {state['motor_current_true']} A")
        
    print("\nTesting Degraded Motor (50% Bearing Wear)")
    fault_degraded = FaultParameters(bearing_wear_severity=0.5)
    motor_degraded = MotorModel(mot_p, fault_degraded, sim_p)
    
    for second in range(1, 6):
        state = motor_degraded.calculate_state(v_in=300.0, t_load=0.0)
        print(f"Sec {second}: {state['rpm_true']} RPM | Current: {state['motor_current_true']} A")