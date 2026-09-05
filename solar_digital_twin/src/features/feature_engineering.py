import sys
import os
import pandas as pd
import numpy as np

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def extract_features():
    print("Starting Research Feature Engineering Pipeline...")
    
    raw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/master_multiscenario_dataset.csv'))
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Master dataset not found at {raw_path}. Run dataset generator first.")
        
    df = pd.read_csv(raw_path)
    print(f"Loaded raw dataset with {len(df)} rows.")
    
    # 1. Use SCADA Measured Telemetry for Feature Extraction
    rho = 1000.0
    g = 9.81
    
    # Initialize feature columns with default zeros
    df['p_elec_measured'] = 0.0
    df['p_hyd_measured'] = 0.0
    df['efficiency_proxy'] = 0.0
    df['q_over_n'] = 0.0
    df['h_over_n2'] = 0.0
    
    active_mask = df['rpm_measured'] > 100
    
    # Extract subsets to prevent shape broadcasting mismatches
    v_dc_sub = df.loc[active_mask, 'v_dc_measured']
    i_dc_sub = df.loc[active_mask, 'i_dc_measured']
    flow_sub = df.loc[active_mask, 'flow_measured']
    head_sub = df.loc[active_mask, 'head_measured']
    rpm_sub = df.loc[active_mask, 'rpm_measured']
    
    # Calculate electrical power: P_elec = V_DC * I_DC
    p_elec_sub = v_dc_sub * i_dc_sub
    df.loc[active_mask, 'p_elec_measured'] = p_elec_sub
    
    # Calculate hydraulic power: P_hyd = rho * g * (Q in m^3/s) * (Head in meters)
    q_m3s = (flow_sub / 1000.0) / 60.0
    p_hyd_sub = rho * g * q_m3s * head_sub
    df.loc[active_mask, 'p_hyd_measured'] = p_hyd_sub
    
    # Calculate System Efficiency Proxy: eta = P_hyd / P_elec
    pelec_safe = np.where(p_elec_sub > 1.0, p_elec_sub, np.nan)
    efficiency_sub = p_hyd_sub / pelec_safe
    df.loc[active_mask, 'efficiency_proxy'] = np.nan_to_num(efficiency_sub, nan=0.0)
    
    # Calculate Core Research Physics Ratios: Q/N and H/N^2
    df.loc[active_mask, 'q_over_n'] = flow_sub / rpm_sub
    df.loc[active_mask, 'h_over_n2'] = head_sub / (rpm_sub ** 2)
    
    # 2. Rolling Time-Series Statistics (60-second rolling window per scenario)
    rolling_features = []
    for scen_id, group in df.groupby('scenario_id'):
        g = group.copy()
        g['flow_rolling_mean'] = g['flow_measured'].rolling(window=60, min_periods=1).mean()
        g['flow_rolling_std'] = g['flow_measured'].rolling(window=60, min_periods=1).std().fillna(0.0)
        g['rpm_rolling_mean'] = g['rpm_measured'].rolling(window=60, min_periods=1).mean()
        rolling_features.append(g)
        
    df_processed = pd.concat(rolling_features, ignore_index=True)
    
    # Save to processed data folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/'))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_master_dataset.csv')
    
    df_processed.to_csv(output_path, index=False)
    print(f"\nFeature Engineering Complete!")
    print(f"Total processed rows: {len(df_processed)}")
    print(f"Saved successfully to: {output_path}")

if __name__ == "__main__":
    extract_features()