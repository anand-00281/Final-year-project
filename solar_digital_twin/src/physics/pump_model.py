import sys
import os
import numpy as np

# Ensure python can find our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import PumpParameters, HydraulicParameters, FaultParameters

import math

class PumpModel:
    def __init__(self, p_params, h_params, f_params):
        self.p_params = p_params
        self.h_params = h_params
        self.f_params = f_params

    def calculate_operating_point(self, omega):
        """
        Calculates Q and H by finding the intersection of:
        Hp = K1*w^2 - K2*w*Q - K3*Q^2
        Hs = H_static + K_sys*Q^2
        """
        # If motor isn't spinning fast enough, no flow
        if omega < 10.0:
            return {'flow_true': 0.0, 'head_true': self.h_params.h_static, 'p_hyd_true': 0.0, 'load_torque_true': 0.0}

        # 1. Apply Impeller Blockage Fault directly to Pump Coefficients
        # Blockage reduces generation capability (K1) and increases internal friction (K2, K3)
        blockage = self.f_params.impeller_blockage_severity
        k1_eff = self.p_params.k1 * (1.0 - 0.4 * blockage) 
        k2_eff = self.p_params.k2 * (1.0 + 2.0 * blockage)
        k3_eff = self.p_params.k3 * (1.0 + 2.0 * blockage)
        
        k_sys = self.h_params.k_system
        h_stat = self.h_params.h_static
        
        # 2. Solve Quadratic Intersection: A*Q^2 + B*Q + C = 0
        A = k3_eff + k_sys
        B = k2_eff * omega
        C = h_stat - (k1_eff * (omega ** 2))
        
        # If C >= 0, the pump hasn't reached enough speed to overcome static head
        if C >= 0:
            Q_lpm = 0.0
            H_m = h_stat
        else:
            # Quadratic formula for Q
            discriminant = (B ** 2) - (4 * A * C)
            if discriminant > 0:
                Q_lpm = (-B + math.sqrt(discriminant)) / (2 * A)
                Q_lpm = max(0.0, Q_lpm)
            else:
                Q_lpm = 0.0
            
            # Calculate Head at this Flow
            H_m = h_stat + (k_sys * (Q_lpm ** 2))

        # 3. Handle Dry Running (Loss of Prime / Hydraulic Coupling)
        # Instead of changing water density, dry running destroys the pump's ability to grip water
        kappa_hyd = 0.05 if self.f_params.is_dry_running else 1.0
        
        Q_actual = Q_lpm * kappa_hyd
        
        # 4. Power and Torque Calculations
        # P_hyd = rho * g * Q(m^3/s) * H
        q_m3s = Q_actual / 60000.0  # Convert LPM to m^3/s
        p_hyd = self.h_params.density * self.h_params.gravity * q_m3s * H_m
        
        # Calculate Mechanical Load Torque
        if omega > 0:
            # Pump efficiency degrades with blockage
            eta_eff = self.p_params.eta_pump_base * (1.0 - 0.5 * blockage)
            # Ensure we don't divide by zero
            eta_eff = max(0.1, eta_eff)
            
            p_mech = p_hyd / eta_eff
            
            # Even if Q=0 (e.g. shutoff or dry running), churning water creates base torque
            churning_torque = 0.001 * (omega ** 2)
            
            t_load = (p_mech / omega) + churning_torque
        else:
            t_load = 0.0

        return {
            'flow_true': Q_actual,
            'head_true': H_m,
            'p_hyd_true': p_hyd,
            'load_torque_true': t_load
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