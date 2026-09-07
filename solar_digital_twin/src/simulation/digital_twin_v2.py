import numpy as np
import pandas as pd

from config.parameters import WeatherParameters, MotorParameters, PumpParameters, HydraulicParameters, SimulationParameters, FaultParameters, SensorParameters
from src.weather.weather_sim import WeatherEngine
from src.physics.motor_model import MotorModel
from src.physics.pump_model import PumpModel
from src.sensors.sensor_model import SensorModel

class ResearchDigitalTwin:
    def __init__(self, scenario, weather_params=None, motor_params=None, pump_params=None, hydraulic_params=None, sim_params=None):
        self.scenario = scenario
        self.weath_p = weather_params or WeatherParameters()
        self.mot_p = motor_params or MotorParameters()
        self.pump_p = pump_params or PumpParameters()
        self.hyd_p = hydraulic_params or HydraulicParameters()
        self.sim_p = sim_params or SimulationParameters()
        
        # Guide Issue 10: Isolate RNGs per subsystem using scenario seed
        base_seed = getattr(self.scenario, 'seed', 42)
        self.weather_rng = np.random.default_rng(base_seed + 1)
        self.sensor_rng = np.random.default_rng(base_seed + 2)
        self.fault_rng = np.random.default_rng(base_seed + 3)
        
        # Pass isolated weather RNG
        self.weather_engine = WeatherEngine(weather_params=self.weath_p, sim_params=self.sim_p, seed=base_seed + 1)
        
        # SAFELY handle fault parameters (This specific line fixes your AttributeError)
        self.fault_p = getattr(self.scenario, 'fault_params', FaultParameters())
        
        self.motor = MotorModel(self.mot_p, self.fault_p, self.sim_p)
        self.pump = PumpModel(self.pump_p, self.hyd_p, self.fault_p)
        self.sensor = SensorModel(sensor_params=SensorParameters())
        
        self._last_load_torque = 0.0

    def _calculate_dynamic_severity(self, time_sec):
        """
        Guide Issue 8: Implements time-dependent fault severity envelope s(t).
        Healthy morning -> Ramp-up at 10:00 AM (36000s) -> Stable active fault.
        """
        base_severity = getattr(self.scenario, 'fault_severity', 0.0)
        if base_severity == 0.0:
            return 0.0
            
        onset_start = 36000.0
        ramp_duration = 3600.0
        
        if time_sec < onset_start:
            return 0.0
        elif time_sec < (onset_start + ramp_duration):
            progress = (time_sec - onset_start) / ramp_duration
            return base_severity * progress
        else:
            return base_severity

    def simulate_scenario(self, cloud_events=None):
        weather_df = self.weather_engine.generate_24h_profile(cloud_events=cloud_events)
        time_array, irrad_array, temp_array = weather_df
        
        steps_per_output = int(self.sim_p.output_dt_s / self.sim_p.internal_dt_s)
        telemetry_records = []
        
        self.motor.omega = 0.0
        self._last_load_torque = 0.0
        
        for i in range(len(time_array)):
            time_s = time_array[i]
            irrad = irrad_array[i]
            temp = temp_array[i]
            
            current_severity = self._calculate_dynamic_severity(time_s)
            
            if self.scenario.fault_type == 'bearing_wear':
                self.fault_p.bearing_wear_severity = current_severity
            elif self.scenario.fault_type == 'impeller_blockage':
                self.fault_p.impeller_blockage_severity = current_severity
            elif self.scenario.fault_type == 'pv_degradation':
                self.fault_p.pv_degradation_severity = current_severity
            elif self.scenario.fault_type == 'dry_running':
                self.fault_p.is_dry_running = (current_severity > 0.0)

            p_pv_available = irrad * 2.8
            v_dc = 372.0 if irrad > 10 else 0.0
            
            for _ in range(steps_per_output):
                motor_state = self.motor.calculate_state(
                    v_in=v_dc, 
                    p_pv_available=p_pv_available, 
                    t_load=self._last_load_torque,
                    dt=self.sim_p.internal_dt_s
                )
                pump_state = self.pump.calculate_operating_point(motor_state['omega_true'])
                self._last_load_torque = pump_state['load_torque_true']
            
            ideal_state = {
                'time_sec': time_s,
                'scenario_id': self.scenario.scenario_id,
                'irradiance_true': irrad,
                'ambient_temp_true': temp,
                'v_dc_measured': v_dc,
                'i_dc_true': motor_state['i_dc_true'],
                'p_elec_measured': v_dc * motor_state['i_dc_true'],
                **motor_state,
                **pump_state,
                'fault_type': self.scenario.fault_type,
                'fault_severity': current_severity
            }
            
            measured_state = self.sensor.apply_noise(ideal_state)
            telemetry_records.append(measured_state)
            
        return pd.DataFrame(telemetry_records)