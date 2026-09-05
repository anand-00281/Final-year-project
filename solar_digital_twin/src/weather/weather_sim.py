import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Ensure Python can find our config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config.parameters import WeatherParameters, SimulationParameters, rng

@dataclass
class CloudEvent:
    start_time: int     # Second of the day (e.g., 43200 for 12:00 PM)
    duration: int       # Total duration in seconds
    depth: float        # Drop severity (0.0 to 1.0, where 0.8 is an 80% drop)
    onset_time: int     # Seconds it takes to reach full depth
    recovery_time: int  # Seconds it takes to recover to full sun

class WeatherEngine:
    def __init__(self, weather_params: WeatherParameters, sim_params: SimulationParameters):
        self.wp = weather_params
        self.sp = sim_params
        
        # 24 hours * 3600 seconds = 86400 samples for a 1-Hz simulation
        self.total_seconds = int(24 * 3600 / self.sp.internal_dt)
        self.time_array = np.arange(0, self.total_seconds * self.sp.internal_dt, self.sp.internal_dt)

    def generate_day(self, cloud_events=None, noise_std=5.0):
        """
        Generates a 24-hour weather profile returned as a pandas DataFrame.
        Columns: time_hrs, irradiance, temperature
        """
        t, irrad, temp = self.generate_24h_profile(cloud_events=cloud_events, noise_std=noise_std)
        return pd.DataFrame({
            'time_hrs': t / 3600.0,
            'irradiance': irrad,
            'temperature': temp
        })

    def generate_24h_profile(self, cloud_events=None, noise_std=5.0):
        """
        Generates 1-Hz environmental data over 24 hours.
        Combines Layer A (Clear Sky), Layer B (Noise), and Layer C (Clouds).
        """
        
        # --- LAYER A: Clear Sky Deterministic Envelope ---
        # Sunrise at 6 AM (21,600s), Sunset at 6 PM (64,800s)
        sun_start, sun_end = 6 * 3600, 18 * 3600
        
        irrad = np.zeros(self.total_seconds)
        sun_mask = (self.time_array >= sun_start) & (self.time_array <= sun_end)
        
        # Ideal sine wave for irradiance
        irrad[sun_mask] = self.wp.peak_irradiance * np.sin(
            np.pi * (self.time_array[sun_mask] - sun_start) / (sun_end - sun_start)
        )
        
        # --- LAYER B: Stochastic Variability ---
        # Use the controlled RNG from parameters.py to ensure reproducibility
        noise = rng.normal(0, noise_std, self.total_seconds)
        irrad = np.clip(irrad + noise, 0, None)
        
        # --- LAYER C: Explicit Cloud Events ---
        cloud_multiplier = np.ones(self.total_seconds)
        
        if cloud_events:
            t = self.time_array
            for event in cloud_events:
                # Boolean masks for vectorized performance
                in_onset = (t >= event.start_time) & (t < event.start_time + event.onset_time)
                in_recovery = (t > event.start_time + event.duration - event.recovery_time) & (t <= event.start_time + event.duration)
                in_full = (t >= event.start_time + event.onset_time) & (t <= event.start_time + event.duration - event.recovery_time)
                
                multiplier = np.ones(self.total_seconds)
                
                # Apply V-shape or U-shape drops
                if event.onset_time > 0:
                    multiplier[in_onset] = 1.0 - event.depth * ((t[in_onset] - event.start_time) / event.onset_time)
                
                multiplier[in_full] = 1.0 - event.depth
                
                if event.recovery_time > 0:
                    multiplier[in_recovery] = 1.0 - event.depth * ((event.start_time + event.duration - t[in_recovery]) / event.recovery_time)
                    
                cloud_multiplier *= multiplier

        # Apply clouds to irradiance
        irrad = irrad * cloud_multiplier
        
        # --- TEMPERATURE MODEL ---
        # Temp lags behind irradiance by ~2 hours
        temp = np.full(self.total_seconds, self.wp.base_temp)
        temp_start, temp_end = 8 * 3600, 20 * 3600
        temp_mask = (self.time_array >= temp_start) & (self.time_array <= temp_end)
        
        temp[temp_mask] = self.wp.base_temp + (self.wp.peak_temp - self.wp.base_temp) * np.sin(
            np.pi * (self.time_array[temp_mask] - temp_start) / (temp_end - temp_start)
        )
        
        return self.time_array, irrad, temp

# --- Quick Test Block ---
if __name__ == "__main__":
    print("Generating 1-Hz Weather Profile (86,400 samples)...")
    engine = WeatherEngine(WeatherParameters(), SimulationParameters())
    
    # Define a severe cloud transient exactly at noon (43200 seconds)
    # Drops irradiance by 80% (0.8) over 5 minutes (300s), lasts 15 minutes (900s) total
    test_cloud = CloudEvent(start_time=43200, duration=900, depth=0.8, onset_time=300, recovery_time=300)
    
    t, irrad, temp = engine.generate_24h_profile(cloud_events=[test_cloud])
    
    print(f"Total samples generated: {len(irrad)}")
    print(f"Irradiance at 11:50 AM (Clear): {irrad[42600]:.2f} W/m2")
    print(f"Irradiance at 12:08 PM (Cloud): {irrad[43680]:.2f} W/m2")
    
    # Plot it visually so we can confirm the causal transient
    plt.figure(figsize=(10, 4))
    plt.plot(t / 3600, irrad, label="Irradiance (W/m2)", color='orange')
    plt.plot(t / 3600, temp * 10, label="Temp (C * 10)", color='red', linestyle='--')
    plt.title("Stage 2: 1-Hz Weather Engine with Controlled Cloud Transient")
    plt.xlabel("Hour of Day")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    # Save to visually verify
    os.makedirs(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/graphs/')), exist_ok=True)
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/graphs/stage2_weather.png'))
    plt.savefig(plot_path)
    print(f"Saved weather verification plot to: {plot_path}")