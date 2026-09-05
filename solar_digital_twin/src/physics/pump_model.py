import sys
import os
import numpy as np

# Ensure python can find our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import PumpParameters, HydraulicParameters, FaultParameters

class PumpModel:
    def __init__(self, pump_params: PumpParameters, hyd_params: HydraulicParameters, fault_params: FaultParameters):
        self.p_params = pump_params
        self.h_params = hyd_params
        self.faults = fault_params

    def calculate_operating_point(self, omega_rad_s: float):
        """
        Calculates the hydraulic operating point (Flow and Head) 
        and the resulting mechanical load torque on the motor.
        """
        # If the motor is barely spinning, it can't overcome static head
        if omega_rad_s < 10.0:
            return {
                'flow_true': 0.0,
                'head_true': self.h_params.h_static,
                'hydraulic_power_true': 0.0,
                'load_torque_true': 0.0
            }

        # 1. INJECT FAULT: Impeller Blockage / Pipe Clogging
        # This increases the system resistance coefficient (K_system)
        # A severe blockage (1.0) could increase resistance by 5x
        k_sys_effective = self.h_params.k_system * (1.0 + 4.0 * self.faults.impeller_blockage_severity)

        # 2. INJECT FAULT: Dry Running
        # If dry running, fluid density drops drastically (air instead of water)
        density_effective = self.h_params.density
        if self.faults.is_dry_running:
            density_effective = self.h_params.density * 0.01 # Pumping air

        # 3. Solve for Flow (Q) using the intersection of the two curves:
        # H_pump = K1*w^2 - K2*w*Q - K3*Q^2
        # H_sys = H_static + K_sys_eff*Q^2
        # Therefore: (K3 + K_sys_eff)*Q^2 + (K2*w)*Q + (H_static - K1*w^2) = 0
        
        A = self.p_params.k3 + k_sys_effective
        B = self.p_params.k2 * omega_rad_s
        C = self.h_params.h_static - (self.p_params.k1 * (omega_rad_s**2))

        # Solve quadratic equation for Q
        discriminant = (B**2) - (4 * A * C)
        
        if discriminant < 0 or C > 0:
            # The pump isn't spinning fast enough to overcome static head
            flow_m3_s = 0.0
            head_m = self.h_params.h_static
        else:
            # Positive root gives the physical flow rate
            flow_m3_s = (-B + np.sqrt(discriminant)) / (2 * A)
            # Calculate corresponding head
            head_m = self.h_params.h_static + (k_sys_effective * (flow_m3_s**2))

        # 4. Calculate Hydraulic Power and resulting Mechanical Load Torque
        # P = rho * g * Q * H
        p_hyd = density_effective * self.h_params.gravity * flow_m3_s * head_m
        
        # T_load = P_hyd / omega (Assuming pump efficiency is built into the K coefficients)
        t_load = p_hyd / omega_rad_s if omega_rad_s > 0 else 0.0

        # Convert flow to Liters Per Minute (LPM) for telemetry
        flow_lpm = flow_m3_s * 1000.0 * 60.0

        return {
            'flow_true': round(flow_lpm, 2),
            'head_true': round(head_m, 2),
            'hydraulic_power_true': round(p_hyd, 2),
            'load_torque_true': round(t_load, 3)
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    p_p = PumpParameters()
    h_p = HydraulicParameters()
    
    # Assume the motor has spooled up to 300 rad/s (~2800 RPM)
    test_omega = 300.0
    
    print(f"Testing Pump at {test_omega} rad/s")
    
    # Test 1: Healthy System
    fault_healthy = FaultParameters(impeller_blockage_severity=0.0)
    pump_h = PumpModel(p_p, h_p, fault_healthy)
    state_h = pump_h.calculate_operating_point(test_omega)
    print(f"Healthy:  Flow = {state_h['flow_true']} LPM | Head = {state_h['head_true']} m | Load Torque = {state_h['load_torque_true']} Nm")
    
    # Test 2: 50% Impeller Blockage
    fault_blocked = FaultParameters(impeller_blockage_severity=0.5)
    pump_b = PumpModel(p_p, h_p, fault_blocked)
    state_b = pump_b.calculate_operating_point(test_omega)
    print(f"Blocked:  Flow = {state_b['flow_true']} LPM | Head = {state_b['head_true']} m | Load Torque = {state_b['load_torque_true']} Nm")
    
    # Test 3: Dry Running
    fault_dry = FaultParameters(is_dry_running=True)
    pump_d = PumpModel(p_p, h_p, fault_dry)
    state_d = pump_d.calculate_operating_point(test_omega)
    print(f"Dry Run:  Flow = {state_d['flow_true']} LPM | Head = {state_d['head_true']} m | Load Torque = {state_d['load_torque_true']} Nm (Notice near-zero torque)")