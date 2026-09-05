import sys
import os
import pandas as pd

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.simulation.digital_twin_v2 import ResearchDigitalTwin
from src.faults.scenarios import get_scenario_library
from src.weather.weather_sim import CloudEvent

def generate_research_dataset():
    print("Starting Full Research Dataset Generation from Scenario Library...")
    
    library = get_scenario_library()
    all_dataframes = []
    
    # Standard cloud transient configuration for cloud scenarios
    standard_cloud = CloudEvent(
        start_time=43200,    # 12:00 PM
        duration=900,        # 15 minutes
        depth=0.8,           # 80% drop
        onset_time=300,      # 5 min ramp down
        recovery_time=300    # 5 min ramp up
    )
    
    for key, scen in library.items():
        twin = ResearchDigitalTwin(scen)
        
        # Apply cloud event if scenario requires weather stress
        clouds = [standard_cloud] if scen.weather_type == "cloud_transient" else None
        
        df_scen = twin.simulate_scenario(cloud_events=clouds)
        
        # Assign machine learning / analytical label
        if scen.fault_type == 'none' and not scen.is_dry_running:
            df_scen['ml_label'] = 'normal'
        elif scen.is_dry_running:
            df_scen['ml_label'] = 'dry_running'
        else:
            df_scen['ml_label'] = scen.fault_type
            
        all_dataframes.append(df_scen)
        
    # Combine all 11 scenarios into the master dataset
    master_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save to raw data folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'master_multiscenario_dataset.csv')
    
    master_df.to_csv(output_path, index=False)
    print(f"\nResearch Master Dataset Generation Complete!")
    print(f"Total rows generated: {len(master_df)}")
    print(f"Saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_research_dataset()
    