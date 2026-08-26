# baseline_config.py
# Baseline Physics Parameters for 3 HP (2.2 kW) Solar Pumping System

# 1. PV Array Specifications (10x 330W Polycrystalline Panels in Series)
PV_P_MAX = 3300.0          # Watts (Total array)
PV_V_OC = 456.0            # Volts (10 panels * 45.6V)
PV_I_SC = 9.45             # Amps
PV_V_MP = 372.0            # Volts (10 panels * 37.2V)
PV_I_MP = 8.88             # Amps
PV_TEMP_COEFF_P = -0.0040  # -0.40% per degree C (Power temperature coefficient)
STC_TEMP = 25.0            # Standard Test Condition Temperature (Celsius)
STC_IRRAD = 1000.0         # Standard Test Condition Irradiance (W/m^2)

# 2. Inverter / VFD Specifications
INV_RATED_POWER = 3000.0   # Watts
INV_EFFICIENCY = 0.97      # 97% baseline conversion efficiency

# 3. Motor & Pump Specifications
MOTOR_RATED_KW = 2.2       # 3 HP
MOTOR_EFFICIENCY = 0.80    # 80% electromechanical efficiency
PUMP_RATED_HEAD_M = 50.0   # Meters
PUMP_RATED_FLOW_LPM = 100.0 # Liters per minute at rated head

# Physical Constants for Hydraulic Equations
WATER_DENSITY = 1000.0     # kg/m^3
GRAVITY = 9.81             # m/s^2
