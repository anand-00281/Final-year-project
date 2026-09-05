import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import classification_report

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def train_model():
    print("Starting Physics-Informed XGBoost Classifier Training...")
    
    processed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/processed_master_dataset.csv'))
    df = pd.read_csv(processed_path)
    print(f"Loaded processed dataset with {len(df)} rows.")
    
    # Include full environmental context and physics ratios
    feature_columns = [
        'irradiance_true', 'ambient_temp_true', 'v_dc_measured', 'i_dc_measured', 'p_elec_measured',
        'rpm_measured', 'flow_measured', 'head_measured', 'p_hyd_measured',
        'efficiency_proxy', 'q_over_n', 'h_over_n2',
        'flow_rolling_mean', 'flow_rolling_std', 'rpm_rolling_mean'
    ]
    
    # Filter inactive night hours
    df_active = df[df['rpm_measured'] > 100].copy()
    
    X = df_active[feature_columns]
    y_raw = df_active['ml_label']
    
    # XGBoost requires numeric class labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    print(f"Training samples: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the proposed XGBoost Model
    print("\nTraining Proposed Physics-Informed XGBoost Classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=12,
        learning_rate=0.1,
        random_state=42,
        tree_method='hist', # Faster training
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    print("\nEvaluating XGBoost model on test set...")
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save both the model and the label encoder
    model_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'fault_detection_model.pkl'))
    encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'))
    
    with open(model_output_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(encoder_path, 'wb') as f:
        pickle.dump(le, f)
        
    print(f"\nModel training complete! Saved artifact to: {model_output_path}")

if __name__ == "__main__":
    train_model()