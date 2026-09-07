import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import numpy as np
from src.simulation.digital_twin_v2 import ResearchDigitalTwin
from config.parameters import SimulationParameters, WeatherParameters, MotorParameters, PumpParameters, HydraulicParameters, FaultParameters
from src.faults.scenarios import Scenario
from src.validation.physics_checks import run_physical_acceptance_tests

def generate_master_dataset():
    print("Initializing Master Dataset Generation...")
    
    sim_p = SimulationParameters()
    weather_p = WeatherParameters()
    motor_p = MotorParameters()
    pump_p = PumpParameters()
    hyd_p = HydraulicParameters()

    scenarios_config = [
        {"id": "SC-02_Normal", "fault": "impeller_blockage", "sev": 0.0, "seed": 42, "cloud_depth": 0.5},
        {"id": "SC-02_Blockage", "fault": "impeller_blockage", "sev": 0.8, "seed": 42, "cloud_depth": 0.5},
        {"id": "SC-03_Bearing", "fault": "bearing_wear", "sev": 0.7, "seed": 100, "cloud_depth": 0.9},
        {"id": "SC-03_DryRun", "fault": "dry_running", "sev": 1.0, "seed": 200, "cloud_depth": 0.9}
    ]

    all_dfs = []

    class CloudTransient:
        def __init__(self, depth):
            self.start_time = 43200.0
            self.duration = 3600.0
            self.depth = depth
            self.onset_time = 300.0
            self.recovery_time = 300.0

    for cfg in scenarios_config:
        print(f"\nProcessing Scenario: {cfg['id']} (Fault: {cfg['fault']}, Severity: {cfg['sev']})")
        
        scenario = Scenario(
            scenario_id=cfg['id'],
            fault_type=cfg['fault'],
            fault_severity=cfg['sev'],
            seed=cfg['seed'],
            weather_type="variable",
            is_dry_running=(cfg['fault'] == 'dry_running')
        )
        
        twin = ResearchDigitalTwin(
            scenario=scenario,
            weather_params=weather_p,
            motor_params=motor_p,
            pump_params=pump_p,
            hydraulic_params=hyd_p,
            sim_params=sim_p
        )
        
        df_sim = twin.simulate_scenario(cloud_events=[CloudTransient(cfg['cloud_depth'])])
        
        passed = run_physical_acceptance_tests(df_sim)
        if not passed:
            print(f"Warning: Scenario {cfg['id']} triggered physical validation boundary warnings.")
            
        all_dfs.append(df_sim)

    master_df = pd.concat(all_dfs, ignore_index=True)
    output_path = os.path.join(root_dir, "master_simulation_dataset.csv")
    master_df.to_csv(output_path, index=False)
    print(f"\nMaster dataset successfully compiled and saved to: {output_path}")
    print(f"Total Rows Generated: {len(master_df):,}")

if __name__ == '__main__':
    generate_master_dataset()