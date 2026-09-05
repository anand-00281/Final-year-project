import sys
import os
import pandas as pd

# Ensure python can find our config and modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import (
    SimulationParameters, WeatherParameters, PVParameters,
    MotorParameters, PumpParameters, HydraulicParameters,
    FaultParameters, SensorParameters
)

from src.weather.weather_sim import WeatherEngine, CloudEvent
from src.physics.pv_array import PVModel
from src.physics.motor_model import MotorModel
from src.physics.pump_model import PumpModel
from src.sensors.sensor_model import SensorModel
from src.faults.scenarios import Scenario, get_scenario_library

class ResearchDigitalTwin:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        
        # 1. Initialize Configuration
        self.sim_p = SimulationParameters()
        self.weath_p = WeatherParameters()
        self.pv_p = PVParameters()
        self.mot_p = MotorParameters()
        self.pump_p = PumpParameters()
        self.hyd_p = HydraulicParameters()
        self.sens_p = SensorParameters()
        
        # 2. Configure Faults based on Scenario parameters
        self.fault_p = FaultParameters()
        if scenario.fault_type == "pv_degradation":
            self.fault_p.pv_degradation_severity = scenario.fault_severity
        elif scenario.fault_type == "bearing_wear":
            self.fault_p.bearing_wear_severity = scenario.fault_severity
        elif scenario.fault_type == "impeller_blockage":
            self.fault_p.impeller_blockage_severity = scenario.fault_severity
            
        self.fault_p.is_dry_running = scenario.is_dry_running

        # 3. Instantiate Subsystems (using explicit keyword arguments)
        self.weather_engine = WeatherEngine(weather_params=self.weath_p, sim_params=self.sim_p)
        self.pv = PVModel(self.pv_p, self.fault_p)
        self.motor = MotorModel(self.mot_p, self.fault_p, self.sim_p)
        self.pump = PumpModel(self.pump_p, self.hyd_p, self.fault_p)
        self.sensor = SensorModel(self.sens_p)
        
        self._last_load_torque = 0.0

    def simulate_scenario(self, cloud_events=None):
        """Executes a full 24-hour physical simulation at 1-Hz."""
        print(f"Executing Scenario: {self.scenario.scenario_id}...")
        
        # Generate the weather for the day
        weather_df = self.weather_engine.generate_24h_profile(cloud_events=cloud_events)
        time_array, irrad_array, temp_array = weather_df
        
        telemetry_records = []
        
        # Run the physics loop at 1-Hz
        for i in range(len(time_array)):
            time_s = time_array[i]
            irrad = irrad_array[i]
            temp = temp_array[i]
            
            # Step 1: PV Array converts weather to electricity
            pv_state = self.pv.calculate_state(irrad, temp)
            
            # Step 2: Motor consumes electricity and mechanical load
            motor_state = self.motor.calculate_state(v_in=pv_state['v_dc_true'], t_load=self._last_load_torque)
            
            # Step 3: Pump consumes RPM and generates load
            pump_state = self.pump.calculate_operating_point(motor_state['omega_true'])
            
            # Update load torque for the next timestep to break the algebraic loop
            self._last_load_torque = pump_state['load_torque_true']
            
            # Step 4: Assemble True Physical State
            ideal_state = {
                'time_sec': time_s,
                'scenario_id': self.scenario.scenario_id,
                'irradiance_true': irrad,
                'ambient_temp_true': temp,
                **pv_state,
                **motor_state,
                **pump_state,
                'fault_type': self.scenario.fault_type,
                'fault_severity': self.scenario.fault_severity
            }
            
            # Step 5: Apply Sensor Noise to generate SCADA Telemetry
            measured_state = self.sensor.apply_noise(ideal_state)
            telemetry_records.append(measured_state)
            
        df_final = pd.DataFrame(telemetry_records)
        print(f"Simulation Complete. Generated {len(df_final)} samples.")
        return df_final

# --- Quick Test Block ---
if __name__ == "__main__":
    # Fetch a scenario from our new library
    library = get_scenario_library()
    scen = library["SC-02"]  # SC-02 is Normal + Cloud Transient
    
    # Define the explicit cloud event
    cloud = CloudEvent(
        start_time=43200,    # 12:00 PM
        duration=900,        # 15 minutes total duration
        depth=0.8,           # 80% drop severity
        onset_time=300,      # 5 minutes to ramp down
        recovery_time=300    # 5 minutes to ramp up
    )
    
    twin = ResearchDigitalTwin(scen)
    
    # Apply clouds only if the scenario calls for weather stress
    clouds_to_apply = [cloud] if scen.weather_type == "cloud_transient" else None
    
    df_result = twin.simulate_scenario(cloud_events=clouds_to_apply)
    
    print("\nSimulation successful!")
    print(f"Total rows generated: {len(df_result)}")
    print("Sample columns:", [col for col in df_result.columns if 'rpm' in col or 'flow' in col or 'measured' in col][:4])