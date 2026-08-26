"""
Inverter / VFD Physics Model module.
Simulates power conversion efficiency, frequency modulation, and AC output for the pump motor.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.baseline_config import INV_RATED_POWER, INV_EFFICIENCY

class Inverter:
    def __init__(self):
        self.rated_power = INV_RATED_POWER
        self.efficiency = INV_EFFICIENCY

    def convert_dc_to_ac(self, dc_power_w):
        """
        Converts DC power to AC power, capped at the inverter's maximum rating.
        """
        # The inverter cannot output more than its rated capacity
        max_input_dc = self.rated_power / self.efficiency
        usable_dc = min(dc_power_w, max_input_dc)
        
        # Inverters require a minimum amount of power just to turn on (e.g., 50W)
        if usable_dc < 50.0:
            return 0.0
            
        ac_power = usable_dc * self.efficiency
        return ac_power

# --- Quick Test Block ---
if __name__ == "__main__":
    inv = Inverter()
    print(f"2000W DC input -> {inv.convert_dc_to_ac(2000):.2f}W AC output")
    print(f"4000W DC input (Overload) -> {inv.convert_dc_to_ac(4000):.2f}W AC output (Capped)")