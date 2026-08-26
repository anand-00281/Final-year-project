import sys
import os
import pandas as pd

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.physics_model.digital_twin import SolarPumpDigitalTwin
from src.telemetry.weather_sim import WeatherSimulator

def generate_fault_data(days=3, fault_type="pump_wear"):
    print(f"Generating {days} days of data with fault: {fault_type}...")
    
    # 1. Initialize the Physics Engine
    twin = SolarPumpDigitalTwin()
    
    # 2. INJECT THE PHYSICAL FAULT
    if fault_type == "pump_wear":
        # Normal pump efficiency is 45% (0.45). We degrade it to 30% (0.30)
        twin.pump.pump_efficiency = 0.30
    elif fault_type == "panel_soiling":
        # Dust reduces the effective maximum power capacity of the panels by 25%
        twin.pv.p_max_stc = twin.pv.p_max_stc * 0.75
        
    all_days_data = []

    for day in range(1, days + 1):
        print(f"  Simulating Day {day}...")
        weather = WeatherSimulator(samples_per_hour=60)
        peak_irrad = 900 + (os.urandom(1)[0] % 200)
        weather.generate_clear_day(peak_irrad=peak_irrad)
        weather.add_cloud_cover(cloud_intensity=0.3)
        weather_df = weather.df

        day_telemetry = []
        for index, row in weather_df.iterrows():
            irrad = row['irradiance']
            temp = row['temperature']
            
            # Run the simulation through the degraded Digital Twin
            readings = twin.simulate_timestep(irradiance_w_m2=irrad, ambient_temp_c=temp)
            
            # Tag the data for the Machine Learning model
            readings['timestamp'] = index.strftime("%H:%M:%S")
            readings['day'] = day
            readings['label'] = fault_type
            
            day_telemetry.append(readings)
            
        all_days_data.extend(day_telemetry)

    # 3. Save the anomalous dataset
    final_df = pd.DataFrame(all_days_data)
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../data/raw/{fault_type}_data.csv'))
    final_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Generated {len(final_df)} rows of {fault_type} data.")
    print(f"Saved to: {output_path}\n")

if __name__ == "__main__":
    # Generate 3 days of Pump Wear data
    generate_fault_data(days=3, fault_type="pump_wear")
    
    # Generate 3 days of Panel Soiling data
    generate_fault_data(days=3, fault_type="panel_soiling")