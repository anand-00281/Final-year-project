"""
Digital Twin Integrator module.
Connects PV Array, Inverter, and Pump System into a unified end-to-end simulation system.
"""
import sys
import os

# Ensure python can find our files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.physics_model.pv_array import PVArray
from src.physics_model.inverter import Inverter
from src.physics_model.pump_system import PumpSystem

class SolarPumpDigitalTwin:
    def __init__(self):
        # Instantiate the three physical subsystems
        self.pv = PVArray()
        self.inverter = Inverter()
        self.pump = PumpSystem()

    # IMPORTANT: Notice how this def is indented the exact same amount as def __init__
    def simulate_timestep(self, irradiance_w_m2, ambient_temp_c):
        """
        Runs a single timestep of the simulation.
        Takes weather conditions and returns full system telemetry.
        """
        # 1. Weather hits the panels -> generates DC Power
        dc_power = self.pv.calculate_dc_power(irradiance_w_m2, ambient_temp_c)
        
        # 2. DC Power enters Inverter -> generates AC Power
        ac_power = self.inverter.convert_dc_to_ac(dc_power)
        
        # 3. AC Power drives the Motor/Pump -> generates Water Flow
        flow_lpm = self.pump.calculate_flow_rate(ac_power)
        
        # Return all simulated "sensor readings" as a dictionary
        return {
            "irradiance_w_m2": irradiance_w_m2,
            "ambient_temp_c": ambient_temp_c,
            "dc_power_w": round(dc_power, 2),
            "ac_power_w": round(ac_power, 2),
            "flow_rate_lpm": round(flow_lpm, 2)
        }

# --- Quick Test Block ---
if __name__ == "__main__":
    twin = SolarPumpDigitalTwin()
    
    print("Testing Digital Twin Pipeline...\n")
    
    # Scenario 1: Bright, cool morning
    morning_telemetry = twin.simulate_timestep(600, 20)
    print(f"Morning Scenario: {morning_telemetry}")