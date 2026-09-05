import sys
import os

# Ensure python can find our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import SensorParameters, rng

class SensorModel:
    def __init__(self, sensor_params: SensorParameters):
        self.params = sensor_params

    def apply_noise(self, ideal_state: dict):
        """
        Takes the true physical values and applies reproducible Gaussian noise 
        to simulate what a real-world SCADA/IoT sensor would measure.
        """
        # Copy the ideal state so we can keep the 'true' values for validation
        measured_telemetry = ideal_state.copy()
        
        # Apply Gaussian noise (mean=0, std=parameter) using the reproducible rng
        # Measurements cannot drop below 0
        v_noise = rng.normal(0, self.params.voltage_noise_std)
        measured_telemetry['v_dc_measured'] = max(0.0, ideal_state.get('v_dc_true', 0.0) + v_noise)
        
        i_noise = rng.normal(0, self.params.current_noise_std)
        measured_telemetry['i_dc_measured'] = max(0.0, ideal_state.get('i_dc_true', 0.0) + i_noise)
        
        rpm_noise = rng.normal(0, self.params.rpm_noise_std)
        measured_telemetry['rpm_measured'] = max(0.0, ideal_state.get('rpm_true', 0.0) + rpm_noise)
        
        flow_noise = rng.normal(0, self.params.flow_noise_std)
        measured_telemetry['flow_measured'] = max(0.0, ideal_state.get('flow_true', 0.0) + flow_noise)
        
        head_noise = rng.normal(0, self.params.head_noise_std)
        measured_telemetry['head_measured'] = max(0.0, ideal_state.get('head_true', 0.0) + head_noise)
        
        return measured_telemetry

# --- Quick Test Block ---
if __name__ == "__main__":
    s_p = SensorParameters()
    sensor = SensorModel(s_p)
    
    # A mock ideal physical state from the Digital Twin
    ideal = {
        'v_dc_true': 372.0,
        'i_dc_true': 8.5,
        'rpm_true': 2850.0,
        'flow_true': 120.0,
        'head_true': 50.0
    }
    
    measured = sensor.apply_noise(ideal)
    
    print("Ideal Physical State:")
    for k, v in ideal.items():
        print(f"  {k}: {v}")
        
    print("\nMeasured SCADA Telemetry (with noise):")
    for k, v in measured.items():
        if 'measured' in k:
            print(f"  {k}: {round(v, 2)}")