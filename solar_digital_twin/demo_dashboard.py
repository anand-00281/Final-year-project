import sys
import os
import matplotlib.pyplot as plt
import pandas as pd

# Ensure Python can find our modules from the root directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.simulation.digital_twin_v2 import ResearchDigitalTwin
from src.faults.scenarios import get_scenario_library
from src.weather.weather_sim import CloudEvent

def run_dashboard():
    print("Initializing Digital Twin Dashboard...")
    library = get_scenario_library()
    
    # We will compare a Normal cloudy day vs an Impeller Blockage cloudy day
    scen_normal = library["SC-02"] 
    scen_fault = library["SC-09"]  
    
    # Create a massive cloud at noon
    cloud = CloudEvent(start_time=43200, duration=1200, depth=0.85, onset_time=200, recovery_time=200)
    
    print(f"Simulating Baseline: {scen_normal.scenario_id} (Normal Operation)...")
    twin_normal = ResearchDigitalTwin(scen_normal)
    df_normal = twin_normal.simulate_scenario(cloud_events=[cloud])
    
    print(f"Simulating Fault: {scen_fault.scenario_id} (Impeller Blockage)...")
    twin_fault = ResearchDigitalTwin(scen_fault)
    df_fault = twin_fault.simulate_scenario(cloud_events=[cloud])

    print("\nGenerating Visual Dashboard...")
    
    # Filter to daylight hours (8 AM to 4 PM) so we can see the data clearly
    mask = (df_normal['time_sec'] >= 8 * 3600) & (df_normal['time_sec'] <= 16 * 3600)
    df_n = df_normal[mask].copy()
    df_f = df_fault[mask].copy()
    
    hours = df_n['time_sec'] / 3600.0
    
    # Create a beautiful 4-panel dashboard
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Solar Digital Twin: Normal vs. Impeller Blockage under Cloud Transient", fontsize=16)
    
    # 1. Weather / Irradiance
    axes[0].plot(hours, df_n['irradiance_true'], color='orange', label='Solar Irradiance (W/m²)')
    axes[0].set_ylabel("Irradiance")
    axes[0].set_title("1. Weather Engine (Notice the cloud at 12:00 PM)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Motor RPM
    axes[1].plot(hours, df_n['rpm_true'], color='blue', label='Normal RPM')
    axes[1].plot(hours, df_f['rpm_true'], color='red', linestyle='--', label='Blockage RPM')
    axes[1].set_ylabel("Motor RPM")
    axes[1].set_title("2. Motor Physics (RPM drops during cloud, but stays similar during blockage)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    # 3. Water Flow
    axes[2].plot(hours, df_n['flow_true'], color='blue', label='Normal Flow (LPM)')
    axes[2].plot(hours, df_f['flow_true'], color='red', linestyle='--', label='Blockage Flow (LPM)')
    axes[2].set_ylabel("Flow (LPM)")
    axes[2].set_title("3. Hydraulic Pump (Flow crashes due to blockage!)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    # 4. Q/N Ratio (The Machine Learning Feature)
    df_n['q_n'] = df_n['flow_true'] / df_n['rpm_true']
    df_f['q_n'] = df_f['flow_true'] / df_f['rpm_true']
    
    axes[3].plot(hours, df_n['q_n'], color='green', label='Normal Q/N Ratio')
    axes[3].plot(hours, df_f['q_n'], color='purple', linestyle='--', label='Blockage Q/N Ratio')
    axes[3].set_ylabel("Q / N Ratio")
    axes[3].set_xlabel("Time of Day (Hours)")
    axes[3].set_title("4. Physics-Informed Feature (How XGBoost detects the fault despite the cloud)")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save and show
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/graphs/demo_dashboard.png'))
    plt.savefig(save_path, dpi=300)
    print(f"\nDashboard saved to: {save_path}")
    
    # This pops the window open on your screen!
    plt.show()

if __name__ == "__main__":
    run_dashboard()