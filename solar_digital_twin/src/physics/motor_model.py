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

    def calculate_state(self, v_in, p_pv_available, t_load, dt):
        """
        v_in: DC voltage from PV
        p_pv_available: Maximum power available from PV array
        t_load: Load torque from pump
        dt: internal timestep (0.05s)
        """
        # 1. Back EMF
        e_back = self.params.k_e * self.omega_rad_s
        
        # 2. Electrical coupling constraint (The Guide's Critical Issue 3 fix)
        # Motor wants this current:
        i_desired = (v_in - e_back) / self.params.r_m
        
        # But PV can only supply this maximum current:
        i_max_pv = (p_pv_available / v_in) if v_in > 0 else 0.0
        
        # The actual current is bounded by the PV capability and cannot be negative
        i_actual = max(0.0, min(i_desired, i_max_pv))
        
        # 3. Torque & Integration
        t_e = self.params.k_t * i_actual
        
        # Apply bearing wear if active
        b_effective = self.params.b_m * (1 + 10 * self.faults.bearing_wear_severity)
        t_friction = b_effective * self.omega_rad_s
        
        # ODE: d(omega)/dt = (Te - T_load - T_friction) / J
        net_torque = t_e - t_load - t_friction
        d_omega = net_torque / self.params.j
        
        # Euler integration using the micro-timestep
        self.omega_rad_s += d_omega * dt
        self.omega_rad_s = max(0.0, self.omega_rad_s)  # Cannot spin backwards
        
        return {
            'i_dc_true': i_actual,
            't_e_true': t_e,
            'omega_true': self.omega_rad_s,
            'rpm_true': self.omega_rad_s * (30 / 3.14159)
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