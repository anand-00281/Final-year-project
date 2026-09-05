import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.simulation.digital_twin_v2 import ResearchDigitalTwin
from src.faults.scenarios import get_scenario_library
from src.weather.weather_sim import CloudEvent

def run_physics_sanity_checks():
    print("Running Physics Sanity Checks: Analyzing Q/N and H/N^2 Ratios...")
    
    library = get_scenario_library()
    
    # Scenarios to compare
    scenarios_to_test = {
        "Normal Clear": library["SC-01"],
        "Cloud Transient": library["SC-02"],
        "Impeller Blockage": library["SC-05"],
        "Bearing Wear": library["SC-04"]
    }
    
    results = {}
    cloud = CloudEvent(start_time=43200, duration=900, depth=0.8, onset_time=300, recovery_time=300)
    
    for name, scen in scenarios_to_test.items():
        twin = ResearchDigitalTwin(scen)
        clouds = [cloud] if scen.weather_type == "cloud_transient" else None
        df = twin.simulate_scenario(cloud_events=clouds)
        
        # Calculate physical ratios
        # Avoid division by zero by filtering out zero RPM values at night
        active_mask = df['rpm_true'] > 100
        
        df_active = df[active_mask].copy()
        df_active['Q_over_N'] = df_active['flow_true'] / df_active['rpm_true']
        df_active['H_over_N2'] = df_active['head_true'] / (df_active['rpm_true'] ** 2)
        
        results[name] = df_active
        
    # Plotting the comparison around midday (Hours 10 to 16)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for name, df in results.items():
        hours = df['time_sec'] / 3600.0
        axes[0].plot(hours, df['Q_over_N'], label=name, linewidth=1.5)
        axes[1].plot(hours, df['H_over_N2'], label=name, linewidth=1.5)
        
    axes[0].set_ylabel("Q / N (Flow / RPM Ratio)")
    axes[0].set_title("Scientific Sanity Check: Emergent Physical Ratios")
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].set_ylabel("H / N² (Head / RPM² Ratio)")
    axes[1].set_xlabel("Time of Day (Hours)")
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    
    # Save graph
    graphs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/graphs/'))
    os.makedirs(graphs_dir, exist_ok=True)
    plot_path = os.path.join(graphs_dir, 'physics_sanity_ratios.png')
    plt.savefig(plot_path, dpi=300)
    print(f"\nPhysics sanity check plot saved successfully to: {plot_path}")
    print("Verification complete. Check the graph to ensure Q/N drops during blockage while remaining stable during cloud transients.")

if __name__ == "__main__":
    run_physics_sanity_checks()