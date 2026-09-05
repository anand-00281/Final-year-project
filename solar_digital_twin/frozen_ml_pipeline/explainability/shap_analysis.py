import sys
import os
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def run_shap_analysis():
    print("Starting SHAP Explainability Analysis...")
    
    # Load model and encoder
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/fault_detection_model.pkl'))
    encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/label_encoder.pkl'))
    
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)
        
    # Load processed dataset
    processed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/processed_master_dataset.csv'))
    df = pd.read_csv(processed_path)
    
    feature_columns = [
        'irradiance_true', 'ambient_temp_true', 'v_dc_measured', 'i_dc_measured', 'p_elec_measured',
        'rpm_measured', 'flow_measured', 'head_measured', 'p_hyd_measured',
        'efficiency_proxy', 'q_over_n', 'h_over_n2',
        'flow_rolling_mean', 'flow_rolling_std', 'rpm_rolling_mean'
    ]
    
    df_active = df[df['rpm_measured'] > 100].copy()
    X = df_active[feature_columns]
    
    # Sample 1000 background points to keep SHAP computation time reasonable
    print("Sampling 1000 background points for TreeExplainer...")
    X_sample = X.sample(n=1000, random_state=42)
    
    print("Calculating SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    # Generate SHAP Summary Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, class_names=le.classes_, show=False)
    
    # Save Graph
    graphs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/graphs/'))
    os.makedirs(graphs_dir, exist_ok=True)
    plot_path = os.path.join(graphs_dir, 'shap_summary_plot.png')
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSHAP Analysis Complete!")
    print(f"Summary plot saved to: {plot_path}")
    print("\n--- Research Conclusion (Aligned with Guide) ---")
    print("\"SHAP analysis was used to examine the contribution of physics-informed features to the predictions of the proposed classifier.\"")

if __name__ == "__main__":
    run_shap_analysis()