import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.simulation.digital_twin_v2 import ResearchDigitalTwin
from config.parameters import SimulationParameters, WeatherParameters, MotorParameters, PumpParameters, HydraulicParameters
from src.faults.scenarios import Scenario

def run_dashboard():
    print("Initializing Digital Twin Dashboard...")
    
    sim_p = SimulationParameters()
    weather_p = WeatherParameters()
    motor_p = MotorParameters()
    pump_p = PumpParameters()
    hyd_p = HydraulicParameters()

    print("Simulating Baseline: SC-02 (Normal Operation)...")
    scen_normal = Scenario(
        scenario_id="SC-02_Normal",
        fault_type="impeller_blockage",
        fault_severity=0.0,
        seed=42,
        weather_type="variable",
        is_dry_running=False
    )
    
    twin_normal = ResearchDigitalTwin(
        scenario=scen_normal,
        weather_params=weather_p,
        motor_params=motor_p,
        pump_params=pump_p,
        hydraulic_params=hyd_p,
        sim_params=sim_p
    )

    class Cloud:
        start_time = 43200.0
        duration = 3600.0
        depth = 0.8
        onset_time = 300.0
        recovery_time = 300.0

    print("Executing Scenario: SC-02 Normal...")
    df_normal = twin_normal.simulate_scenario(cloud_events=[Cloud()])

    print("Simulating Fault: SC-02 (Impeller Blockage)...")
    scen_block = Scenario(
        scenario_id="SC-02_Blockage",
        fault_type="impeller_blockage",
        fault_severity=0.8,
        seed=42,
        weather_type="variable",
        is_dry_running=False
    )
    
    twin_block = ResearchDigitalTwin(
        scenario=scen_block,
        weather_params=weather_p,
        motor_params=motor_p,
        pump_params=pump_p,
        hydraulic_params=hyd_p,
        sim_params=sim_p
    )
    
    print("Executing Scenario: SC-02 Blockage...")
    df_block = twin_block.simulate_scenario(cloud_events=[Cloud()])

    # Defensive column mapping to prevent KeyError across naming variations
    for df in [df_normal, df_block]:
        if 'flow_lpm' not in df.columns:
            for col in df.columns:
                if 'flow' in col.lower():
                    df['flow_lpm'] = df[col]
                    break
        if 'rpm_true' not in df.columns:
            for col in df.columns:
                if 'rpm' in col.lower() or 'omega' in col.lower():
                    df['rpm_true'] = df[col]
                    break

    # Plotting results
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Solar Digital Twin: Normal vs. Impeller Blockage under Cloud Transient", fontsize=16)

    axes[0].plot(df_normal['time_sec'] / 3600.0, df_normal['irradiance_true'], color='orange', label='Solar Irradiance (W/m²)')
    axes[0].set_ylabel('Irradiance')
    axes[0].set_title('1. Weather Engine (Notice the cloud at 12:00 PM)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(df_normal['time_sec'] / 3600.0, df_normal['rpm_true'], color='blue', label='Normal RPM')
    axes[1].plot(df_block['time_sec'] / 3600.0, df_block['rpm_true'], color='red', linestyle='--', label='Blockage RPM')
    axes[1].set_ylabel('Motor RPM')
    axes[1].set_title('2. Motor Physics (RPM drops during cloud, but stays similar during blockage)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(df_normal['time_sec'] / 3600.0, df_normal['flow_lpm'], color='blue', label='Normal Flow (LPM)')
    axes[2].plot(df_block['time_sec'] / 3600.0, df_block['flow_lpm'], color='red', linestyle='--', label='Blockage Flow (LPM)')
    axes[2].set_ylabel('Flow (LPM)')
    axes[2].set_title('3. Hydraulic Pump (Flow crashes due to blockage!)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    q_n_norm = df_normal['flow_lpm'] / (df_normal['rpm_true'] + 1e-5)
    q_n_block = df_block['flow_lpm'] / (df_block['rpm_true'] + 1e-5)
    axes[3].plot(df_normal['time_sec'] / 3600.0, q_n_norm, color='green', label='Normal Q/N Ratio')
    axes[3].plot(df_block['time_sec'] / 3600.0, q_n_block, color='purple', linestyle='--', label='Blockage Q/N Ratio')
    axes[3].set_ylabel('Q / N Ratio')
    axes[3].set_xlabel('Time of Day (Hours)')
    axes[3].set_title('4. Physics-Informed Feature (How XGBoost detects the fault despite the cloud)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_dashboard()