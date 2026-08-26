"""
Feature Engineering module.
Processes raw synthetic telemetry data into feature sets suitable for ML fault classification models.
"""
import pandas as pd
import numpy as np
import os

def process_data():
    print("Starting Feature Engineering...")
    
    # 1. Define file paths
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/'))
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    
    # 2. Load the three datasets
    df_normal = pd.read_csv(os.path.join(raw_dir, 'normal_operation_baseline.csv'))
    df_normal['label'] = 'normal' # Tag the baseline data
    
    df_wear = pd.read_csv(os.path.join(raw_dir, 'pump_wear_data.csv'))
    df_soiling = pd.read_csv(os.path.join(raw_dir, 'panel_soiling_data.csv'))
    
    # 3. Combine them into one massive dataset
    df_combined = pd.concat([df_normal, df_wear, df_soiling], ignore_index=True)
    print(f"Combined Dataset Size: {len(df_combined)} rows.")
    
    # 4. Remove nighttime/zero-sunlight data 
    # (The pump is off, so we don't want the ML model learning from zeroes)
    df_combined = df_combined[df_combined['irradiance_w_m2'] > 50].copy()
    
    # 5. ENGINEER PHYSICS-INFORMED FEATURES
    
    # Feature A: PV Efficiency (Power per unit of sunlight)
    # Avoid division by zero
    df_combined['feature_pv_efficiency'] = np.where(
        df_combined['irradiance_w_m2'] > 0,
        df_combined['dc_power_w'] / df_combined['irradiance_w_m2'],
        0
    )
    
    # Feature B: System Efficiency (Water pumped per watt of AC power)
    df_combined['feature_system_efficiency'] = np.where(
        df_combined['ac_power_w'] > 0,
        df_combined['flow_rate_lpm'] / df_combined['ac_power_w'],
        0
    )
    
    # 6. Save the final "Machine Learning Ready" dataset
    output_path = os.path.join(processed_dir, 'ml_ready_dataset.csv')
    df_combined.to_csv(output_path, index=False)
    
    print(f"Feature Engineering Complete! {len(df_combined)} daytime rows saved.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    process_data()