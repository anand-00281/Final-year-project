"""
simulation/schemas.py

Defines the column-level contract for Digital Twin output, and enforces
a strict separation between:

    IDEAL_STATE_COLUMNS  -- noise-free physical/ideal simulation state
    MEASURED_COLUMNS     -- sensor-imperfect telemetry (what a real
                             deployed system would actually report)
    METADATA_COLUMNS     -- timestamps, scenario identifiers, weather
                             condition/transient flags
    LABEL_COLUMNS         -- ground-truth fault label & severity

This split exists specifically to prevent data leakage in the future ML
pipeline: METADATA_COLUMNS and LABEL_COLUMNS must never be used as model
input features, and IDEAL_STATE_COLUMNS must never be used as model
input either (a real deployment only ever has MEASURED_COLUMNS
available) -- they exist in the generated dataset purely for physics
validation / debugging / research diagnostics.
"""

from typing import List

# 'normal' plus every mechanistically distinct fault implemented in V2.
# NOTE (ambiguity flagged in PART D / PART C.4): 'partial_shading' is
# implemented as a fully distinct mechanism per Decision 5 of the
# migration approval, but it is NOT one of the 4 core research fault
# classes in Section 7 of the original spec, and none of the 11
# declared scenarios in scenarios/scenario_library.py currently use it.
# It is included here so the machinery and the label space are ready if
# it is added to the scenario library later; until then it will simply
# never appear in fault_label for the default 11 scenarios.
FAULT_LABELS = (
    "normal",
    "bearing_wear",
    "impeller_blockage",
    "pv_degradation",
    "dry_running",
    "partial_shading",
)

METADATA_COLUMNS: List[str] = [
    "timestamp",
    "scenario_id",
    "weather_condition",   # 'clear' | 'cloudy_stochastic' | 'cloud_transient'
    "weather_transient",   # bool: True while a scripted cloud transient is actively reducing irradiance
]

LABEL_COLUMNS: List[str] = [
    "fault_label",       # one of FAULT_LABELS
    "fault_severity",    # 0.0 when fault_label == 'normal'
]

IDEAL_STATE_COLUMNS: List[str] = [
    "Irradiance_W_m2_ideal",
    "DC_Voltage_V_ideal",
    "DC_Current_A_ideal",
    "DC_Power_W_ideal",
    "Motor_RPM_ideal",
    "Flow_Rate_LPM_ideal",
    "Pressure_Head_m_ideal",
    "Hydraulic_Power_W_ideal",
    "Efficiency_Proxy_ideal",
]

MEASURED_COLUMNS: List[str] = [
    "Irradiance_W_m2",
    "DC_Voltage_V",
    "DC_Current_A",
    "DC_Power_W",
    "Motor_RPM",
    "Flow_Rate_LPM",
    "Pressure_Head_m",
    "Hydraulic_Power_W",
    "Efficiency_Proxy",
]

ALL_COLUMNS: List[str] = METADATA_COLUMNS + LABEL_COLUMNS + IDEAL_STATE_COLUMNS + MEASURED_COLUMNS

# Columns a downstream ML pipeline is allowed to use as input features.
# Named explicitly so any future misuse (e.g. accidentally training on
# an _ideal column or a label column) is a visible, deliberate choice
# rather than a silent leak.
ML_SAFE_FEATURE_COLUMNS: List[str] = list(MEASURED_COLUMNS)


def validate_schema(df) -> None:
    """Raise AssertionError if df is missing any required column, or if
    any ML-safe feature column name collides with a metadata/label
    column name."""
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    assert not missing, f"Digital Twin output missing required columns: {missing}"
    overlap = set(ML_SAFE_FEATURE_COLUMNS) & set(METADATA_COLUMNS + LABEL_COLUMNS)
    assert not overlap, f"Leakage risk: feature columns overlap metadata/label columns: {overlap}"
