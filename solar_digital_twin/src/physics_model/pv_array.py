"""
PV Array Physics Model module.
Calculates electrical outputs (voltage, current, power) under varying solar irradiance and temperature.
"""
import sys
import os

# Ensure python can find your config file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.baseline_config import PV_P_MAX, STC_TEMP, STC_IRRAD, PV_TEMP_COEFF_P

class PVArray:
    def __init__(self):
        self.p_max_stc = PV_P_MAX
        self.temp_coeff = PV_TEMP_COEFF_P

    def calculate_dc_power(self, irradiance_w_m2, ambient_temp_c):
        """
        Calculates the actual DC power output of the solar array.
        """
        # If there is no sun, there is no power
        if irradiance_w_m2 <= 0:
            return 0.0

        # 1. Estimate the actual solar cell temperature 
        # (Cells get hotter than the ambient air when the sun shines on them)
        # Using the standard Nominal Operating Cell Temperature (NOCT) approximation
        cell_temp_c = ambient_temp_c + (irradiance_w_m2 / 800.0) * (45.0 - 20.0)

        # 2. Calculate how much efficiency is lost due to heat
        temp_diff = cell_temp_c - STC_TEMP
        temp_derating_factor = 1 + (self.temp_coeff * temp_diff)

        # 3. Calculate final power (Base Power * Sun Ratio * Heat Loss)
        dc_power = self.p_max_stc * (irradiance_w_m2 / STC_IRRAD) * temp_derating_factor

        # Power cannot drop below 0
        return max(0.0, dc_power)

# --- Quick Test Block ---
if __name__ == "__main__":
    pv = PVArray()
    # Test 1: Perfect conditions (1000 W/m2, 25C)
    power_perfect = pv.calculate_dc_power(1000, 25)
    print(f"Power at 1000 W/m2, 25C (Ambient): {power_perfect:.2f} W")
    
    # Test 2: Hot afternoon (800 W/m2, 40C)
    power_hot = pv.calculate_dc_power(800, 40)
    print(f"Power at 800 W/m2, 40C (Ambient): {power_hot:.2f} W")   