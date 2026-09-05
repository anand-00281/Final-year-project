"""
Motor & Pump System Physics Model module.
Calculates hydraulic performance (flow rate, head pressure, hydraulic power) based on electrical input.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.baseline_config import MOTOR_EFFICIENCY, PUMP_RATED_HEAD_M, WATER_DENSITY, GRAVITY

class PumpSystem:
    def __init__(self):
        self.motor_efficiency = MOTOR_EFFICIENCY
        self.head_m = PUMP_RATED_HEAD_M
        self.density = WATER_DENSITY
        self.gravity = GRAVITY
        # Standard centrifugal pump hydraulic efficiency (~45%)
        self.pump_efficiency = 0.45 

    def calculate_flow_rate(self, ac_power_w):
        """
        Calculates water flow rate in Liters Per Minute (LPM) based on AC power.
        """
        if ac_power_w <= 0:
            return 0.0

        # 1. Electrical to Mechanical (Motor)
        mech_power_w = ac_power_w * self.motor_efficiency
        
        # 2. Mechanical to Hydraulic (Pump)
        hydraulic_power_w = mech_power_w * self.pump_efficiency
        
        # 3. Physics Equation: Power = Density * Gravity * Head * Flow(m3/s)
        # Therefore: Flow(m3/s) = Power / (Density * Gravity * Head)
        flow_m3_per_sec = hydraulic_power_w / (self.density * self.gravity * self.head_m)
        
        # 4. Convert m3/s to Liters per minute (LPM)
        flow_lpm = flow_m3_per_sec * 1000.0 * 60.0
        
        return flow_lpm

# --- Quick Test Block ---
if __name__ == "__main__":
    pump = PumpSystem()
    print(f"1500W AC input -> {pump.calculate_flow_rate(1500):.2f} LPM flow rate")