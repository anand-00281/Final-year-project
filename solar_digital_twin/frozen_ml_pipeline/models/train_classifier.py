import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap

# Ensure root directory is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def train_and_explain():
    print("Loading Master Dataset...")
    data_path = os.path.join(root_dir, "master_simulation_dataset.csv")
    df = pd.read_csv(data_path)

    # Defensive column mapping to standardize names across the pipeline
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

    # Filter for active daylight hours (faults are undetectable at night)
    df = df[df['irradiance_true'] > 50.0].copy()
    print(f"Active daylight rows for training: {len(df):,}")

    # Feature Engineering: Injecting Physics-Informed Features
    print("Engineering physics-informed features (Q/N ratio, System Efficiency)...")
    df['q_n_ratio'] = df['flow_lpm'] / (df['rpm_true'] + 1e-5)
    
    # Define feature set and target
    features = [
        'irradiance_true', 
        'ambient_temp_true', 
        'v_dc_measured', 
        'p_elec_measured', 
        'rpm_true', 
        'flow_lpm', 
        'q_n_ratio'
    ]
    
    X = df[features]
    y_labels = df['fault_type']

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    target_names = le.classes_
    print(f"Detected Classes: {target_names}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # Evaluation
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # SHAP Explainability
    print("Generating SHAP Explainability Plots...")
    # Sample background data for SHAP to compute faster
    X_sample = shap.sample(X_train, 1000)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Plot 1: Global Feature Importance (Bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=features, class_names=target_names, show=False)
    plt.title("SHAP Global Feature Importance by Fault Type")
    plt.tight_layout()
    plt.show()

    # Plot 2: Detailed SHAP Summary for a specific class (e.g., Impeller Blockage)
    fault_idx = 1 if len(target_names) > 1 else 0 
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[fault_idx], X_sample, feature_names=features, show=False)
    plt.title(f"SHAP Summary Impact: {target_names[fault_idx]}")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_and_explain()