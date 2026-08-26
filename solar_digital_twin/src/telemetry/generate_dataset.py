import sys
import os
import pandas as pd

# Ensure Python can find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.physics_model.digital_twin import SolarPumpDigitalTwin
from src.telemetry.weather_sim import WeatherSimulator

def generate_normal_operation_data(days=5):
    """
    Generates multi-day synthetic telemetry for baseline (healthy) operation.
    """
    print(f"Generating {days} days of normal operation data...")
    
    # 1. Initialize the Physics Engine
    twin = SolarPumpDigitalTwin()
    
    all_days_data = []

    for day in range(1, days + 1):
        print(f"  Simulating Day {day}...")
        
        # 2. Generate a new weather pattern for the day
        # We use 60 samples per hour (1 reading per minute)
        weather = WeatherSimulator(samples_per_hour=60)
        
        # Vary the peak irradiance slightly day-to-day for realism
        peak_irrad = 900 + (os.urandom(1)[0] % 200) # Random peak between 900 and 1100
        weather.generate_clear_day(peak_irrad=peak_irrad)
        weather.add_cloud_cover(cloud_intensity=0.3)
        weather_df = weather.df

        # 3. Create a list to hold the telemetry for this specific day
        day_telemetry = []

        # 4. Loop through every minute of the day
        for index, row in weather_df.iterrows():
            irrad = row['irradiance']
            temp = row['temperature']
            
            # Run the physics simulation for this exact minute
            readings = twin.simulate_timestep(irradiance_w_m2=irrad, ambient_temp_c=temp)
            
            # Add timestamp and day number to the readings
            readings['timestamp'] = index.strftime("%H:%M:%S")
            readings['day'] = day
            
            day_telemetry.append(readings)
            
        # Append this day's list to our master list
        all_days_data.extend(day_telemetry)

    # 5. Convert the massive list of dictionaries into a Pandas DataFrame
    final_df = pd.DataFrame(all_days_data)
    
    # 6. Save it to a CSV file in the data/raw directory
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw/normal_operation_baseline.csv'))
    final_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Generated {len(final_df)} rows of data.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Generate 5 days of healthy data
    generate_normal_operation_data(days=5)