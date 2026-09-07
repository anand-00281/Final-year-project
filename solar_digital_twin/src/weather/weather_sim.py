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
    def __init__(self, start_time, duration, depth, onset_time, recovery_time):
        self.start_time = start_time
        self.duration = duration
        self.depth = depth
        self.onset_time = onset_time
        self.recovery_time = recovery_time

class WeatherEngine:
    def __init__(self, weather_params, sim_params, seed=42):
        self.wp = weather_params
        self.sp = sim_params
        self.rng = np.random.default_rng(seed)
        self.total_seconds = int(24 * 3600 / self.sp.output_dt_s)
        # Initialize time array explicitly here
        self.time_array = np.arange(0, self.total_seconds, self.sp.output_dt_s)

    def generate_24h_profile(self, cloud_events=None):
        n_steps = len(self.time_array)
        irradiance = np.zeros(n_steps)
        temperature = np.zeros(n_steps)
        
        sun_start = 21600.0
        sun_end = 64800.0
        
        sun_mask = (self.time_array >= sun_start) & (self.time_array <= sun_end)
        day_times = self.time_array[sun_mask]
        
        if len(day_times) > 0:
            peak_irrad = self.wp.peak_irradiance
            sin_curve = np.sin(np.pi * (day_times - sun_start) / (sun_end - sun_start))
            clear_irrad = peak_irrad * sin_curve
            
            noise = self.rng.normal(0, self.wp.noise_std, size=len(day_times))
            irradiance[sun_mask] = np.clip(clear_irrad + noise, 0.0, peak_irrad)
            
            base_temp = self.wp.base_temp
            temp_amplitude = self.wp.temp_amplitude
            temperature[sun_mask] = base_temp + temp_amplitude * sin_curve + self.rng.normal(0, 0.5, size=len(day_times))
        
        if cloud_events:
            for cloud in cloud_events:
                c_start = cloud.start_time
                c_end = c_start + cloud.duration
                
                for i, t in enumerate(self.time_array):
                    if c_start <= t <= c_end:
                        irradiance[i] *= (1.0 - cloud.depth)
                        
        irradiance = np.maximum(0.0, irradiance)
        return self.time_array, irradiance, temperature

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