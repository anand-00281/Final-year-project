import sys
import os

# Ensure python can find our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import PVParameters, FaultParameters

class PVModel:
    def __init__(self, pv_params: PVParameters, fault_params: FaultParameters):
        self.params = pv_params
        self.faults = fault_params

    def calculate_state(self, irradiance_w_m2: float, ambient_temp_c: float):
        """
        Calculates the true physical state of the PV array.
        Returns V_dc, I_dc, and P_dc.
        """
        # If irradiance is too low to turn on the inverter
        if irradiance_w_m2 < 10.0:
            return {
                'v_dc_true': 0.0,
                'i_dc_true': 0.0,
                'p_dc_true': 0.0
            }

        # 1. Estimate actual cell temperature (NOCT model)
        cell_temp_c = ambient_temp_c + (irradiance_w_m2 / 800.0) * (45.0 - 20.0)

        # 2. Calculate thermodynamic efficiency loss
        temp_derating_factor = 1.0 + (self.params.temp_coeff_p * (cell_temp_c - 25.0))

        # 3. Calculate Ideal Power 
        p_ideal = self.params.p_max * (irradiance_w_m2 / 1000.0) * temp_derating_factor

        # 4. INJECT FAULT: PV Degradation
        # This represents physical panel aging, micro-cracks, or severe permanent soiling.
        # Equation: P_available = P_healthy * (1 - S_pv)
        degradation_factor = 1.0 - self.faults.pv_degradation_severity
        p_actual = max(0.0, p_ideal * degradation_factor)

        # 5. Derive Voltage and Current (Reduced-order MPPT behavior)
        # A healthy MPPT controller will maintain voltage near V_mp. 
        v_dc = self.params.v_mp
        
        # Derive current directly to satisfy P = V * I
        i_dc = p_actual / v_dc if v_dc > 0 else 0.0

        return {
            'v_dc_true': round(v_dc, 2),
            'i_dc_true': round(i_dc, 3),
            'p_dc_true': round(p_actual, 2)
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    pv_p = PVParameters()
    
    # Test 1: Healthy Panels
    fault_healthy = FaultParameters(pv_degradation_severity=0.0)
    pv_healthy = PVModel(pv_p, fault_healthy)
    state_h = pv_healthy.calculate_state(1000, 25)
    print(f"Healthy Array:  {state_h['p_dc_true']} W (V={state_h['v_dc_true']}V, I={state_h['i_dc_true']}A)")
    
    # Test 2: 20% Degraded Panels
    fault_degraded = FaultParameters(pv_degradation_severity=0.20)
    pv_degraded = PVModel(pv_p, fault_degraded)
    state_d = pv_degraded.calculate_state(1000, 25)
    print(f"Degraded Array: {state_d['p_dc_true']} W (V={state_d['v_dc_true']}V, I={state_d['i_dc_true']}A)")