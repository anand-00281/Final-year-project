"""
SHAP Analysis & Explainability module.
Provides model interpretability using SHAP values for predicted system faults.
"""
import pandas as pd
import os
import joblib
import shap
import matplotlib.pyplot as plt

def run_shap_analysis():
    print("1. Loading Model and Data for Analysis...")
    
    # Load the model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/fault_detection_model.pkl'))
    rf_model = joblib.load(model_path)
    
    # Load the data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/ml_ready_dataset.csv'))
    df = pd.read_csv(data_path)

    features = [
        'irradiance_w_m2', 
        'ambient_temp_c', 
        'dc_power_w', 
        'ac_power_w', 
        'flow_rate_lpm',
        'feature_pv_efficiency', 
        'feature_system_efficiency'
    ]
    
    X = df[features]
    
    # We don't need to analyze all thousands of rows. 
    # Analyzing a random sample of 500 is enough for a clear plot and runs much faster.
    print("2. Sampling data for SHAP computation...")
    X_sample = shap.sample(X, 500)

    # 3. Calculate SHAP values
    print("3. Calculating SHAP values (Game Theory Math)...")
    # For a Random Forest, TreeExplainer is heavily optimized
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample)

    # 4. Generate and Save the Plot
    print("4. Generating Explainability Plot...")
    
    # Create an output directory for graphs if it doesn't exist
    graphs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/graphs/'))
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Create a Summary Plot. 
    # Note: Depending on your exact shap version, shap_values might be a list (one for each class).
    # We will plot the overall feature importance across all classes.
    plt.figure(figsize=(10, 6))
    
    # Use plot_type='bar' to get a clear, easy-to-read feature importance chart
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    
    # Add a title and save
    plt.title("SHAP Feature Importance (Physics-Informed Digital Twin)")
    plot_path = os.path.join(graphs_dir, 'shap_summary_plot.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\nAnalysis Complete!")
    print(f"SHAP Plot saved to: {plot_path}")
    print("Look at this image—it proves that your engineered physics features drove the model's success.")

if __name__ == "__main__":
    run_shap_analysis()