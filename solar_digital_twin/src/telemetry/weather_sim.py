"""
Weather Simulator module.
Generates realistic solar irradiance and ambient temperature daily curves/series data.
"""

import numpy as np
import pandas as pd

class WeatherSimulator:
    def __init__(self, hours_in_day=24, samples_per_hour=60):
        self.hours = hours_in_day
        self.samples_per_hour = samples_per_hour
        self.total_samples = hours_in_day * samples_per_hour

        # Create a time index (e.g., 00:00 to 23:59)
        time_range = pd.date_range("00:00", "23:59", freq=f"{int(60/samples_per_hour)}min")
        self.df = pd.DataFrame(index=time_range)

    def generate_clear_day(self, peak_irrad=1000.0, peak_temp=35.0, base_temp=15.0):
        """
        Generates a perfect, clear-sky day using Gaussian curves.
        """
        # Create a time array (0 to 24)
        x = np.linspace(0, self.hours, self.total_samples)

        # Irradiance Bell Curve (Centered at noon, width adjusted)
        irrad_curve = peak_irrad * np.exp(-((x - 12)**2) / 10)
        
        # The sun sets, so clip anything below zero
        irrad_curve = np.clip(irrad_curve, 0, None)

        # Temperature Bell Curve (Peaks slightly after noon, around 2-3 PM)
        temp_curve = base_temp + (peak_temp - base_temp) * np.exp(-((x - 14.5)**2) / 15)

        self.df['irradiance'] = irrad_curve
        self.df['temperature'] = temp_curve
        return self.df

    def add_cloud_cover(self, cloud_intensity=0.3):
        """
        Adds random dips in irradiance to simulate clouds.
        """
        # Generate random noise (0 to 1)
        noise = np.random.rand(self.total_samples)
        
        # Only apply clouds where the noise is below the intensity threshold
        cloud_mask = noise < cloud_intensity
        
        # Drop irradiance randomly between 30% and 80% where clouds exist
        drop_factors = np.random.uniform(0.3, 0.8, self.total_samples)
        
        # Apply the drops to the dataframe
        self.df.loc[cloud_mask, 'irradiance'] = self.df['irradiance'] * drop_factors
        
        return self.df

# --- Quick Test Block ---
if __name__ == "__main__":
    sim = WeatherSimulator(samples_per_hour=1) # 1 sample per hour for easy reading
    
    # Generate a clear day, then add clouds
    sim.generate_clear_day(peak_irrad=1000)
    sim.add_cloud_cover(cloud_intensity=0.2)
    
    # Print midday values (10 AM to 2 PM) to verify
    print(sim.df.iloc[10:15])