"""
Fault Classifier Training module.
Trains and evaluates machine learning models for detecting solar digital twin system faults.
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    print("1. Loading processed dataset...")
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/ml_ready_dataset.csv'))
    df = pd.read_csv(data_path)

    # Define the exact inputs (Features) and output (Target) for the AI
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
    y = df['label']

    # 2. Split data: 80% for training, 20% for testing
    print("2. Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Initialize and Train the Random Forest algorithm
    print("3. Training Random Forest Classifier (this might take a moment)...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 4. Evaluate the Model on the 20% of data it hasn't seen yet
    print("\n--- MODEL EVALUATION ---")
    predictions = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, predictions))

    # 5. Save the trained model to disk so we can analyze it with SHAP later
    model_save_path = os.path.join(os.path.dirname(__file__), 'fault_detection_model.pkl')
    joblib.dump(rf_model, model_save_path)
    print(f"\nModel successfully saved to: {model_save_path}")

if __name__ == "__main__":
    train_model()