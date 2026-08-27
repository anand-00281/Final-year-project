"""
scenarios/scenario_library.py

Declarative definitions of the 11 core research scenarios (Section 11).
SCENARIO_LIBRARY is pure DATA (a list of dict "recipes") -- no
simulation logic lives here except the single generic build_scenario()
function that turns a recipe + a start_time into concrete
CloudTransientSpec / FaultEvent objects. digital_twin.py never branches
on scenario_id; it only ever calls build_scenario() and runs whatever
comes out.

AMBIGUITY FLAGGED (carried over from PART C.4 / schemas.py): Section 11
does not include a partial_shading-primary scenario, or a
"partial_shading + cloud" combination, even though partial_shading is
implemented as a fully distinct mechanism (Decision 4/5). The 11
scenarios below follow Section 11 literally. If you want partial_shading
exercised in the generated dataset, two more recipes (a #12 and #13,
mirroring #6/#10's pv_degradation pattern) would need to be added --
not done here without confirmation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from solar_pump_digital_twin.weather.weather_generator import CloudTransientSpec
from solar_pump_digital_twin.faults.fault_models import FaultEvent

# A single reusable "rapid cloud transient" recipe fragment, per Section 9's
# worked example (1000 -> ~500 W/m^2 and recovery). Reused verbatim across
# scenarios 3, 8, 9, 10, 11 so the weather confound is IDENTICAL across the
# fault-vs-no-fault comparisons -- this is a deliberate experimental control,
# not an attempt to make faults artificially separable (the equipment fault
# itself still evolves independently under this identical weather signal).
_RAPID_CLOUD = dict(onset_hour=11.0, onset_s=180, hold_s=600, recovery_s=300, depth=0.5, shape="smoothstep")

SCENARIO_LIBRARY: List[Dict[str, Any]] = [
    dict(scenario_id=1, name="normal_clear_sky", duration_hours=24.0,
         weather_condition="clear", clouds=[], faults=[]),

    dict(scenario_id=2, name="normal_cloudy", duration_hours=24.0,
         weather_condition="cloudy_stochastic",
         clouds=[dict(onset_hour=7.0, onset_s=1800, hold_s=6 * 3600, recovery_s=1800, depth=0.35, shape="smoothstep")],
         faults=[]),

    dict(scenario_id=3, name="normal_rapid_cloud_transient", duration_hours=24.0,
         weather_condition="cloud_transient", clouds=[_RAPID_CLOUD], faults=[]),

    dict(scenario_id=4, name="bearing_wear", duration_hours=24.0,
         weather_condition="clear", clouds=[],
         faults=[dict(fault_type="bearing_wear", onset_hour=10.0, duration_h=3.0,
                       max_severity=0.7, severity_form="transient", ramp_s=600)]),

    dict(scenario_id=5, name="impeller_blockage", duration_hours=24.0,
         weather_condition="clear", clouds=[],
         faults=[dict(fault_type="impeller_blockage", onset_hour=10.0, duration_h=3.0,
                       max_severity=0.7, severity_form="transient", ramp_s=600)]),

    dict(scenario_id=6, name="pv_degradation", duration_hours=24.0,
         weather_condition="clear", clouds=[],
         faults=[dict(fault_type="pv_degradation", onset_hour=10.0, duration_h=None,
                       max_severity=0.4, severity_form="persistent", ramp_s=3600)]),

    dict(scenario_id=7, name="dry_running", duration_hours=24.0,
         weather_condition="clear", clouds=[],
         faults=[dict(fault_type="dry_running", onset_hour=10.0, duration_h=2.0,
                       max_severity=0.9, severity_form="transient", ramp_s=300)]),

    dict(scenario_id=8, name="bearing_wear_plus_cloud", duration_hours=24.0,
         weather_condition="cloud_transient", clouds=[_RAPID_CLOUD],
         faults=[dict(fault_type="bearing_wear", onset_hour=10.0, duration_h=3.0,
                       max_severity=0.7, severity_form="transient", ramp_s=600)]),

    dict(scenario_id=9, name="impeller_blockage_plus_cloud", duration_hours=24.0,
         weather_condition="cloud_transient", clouds=[_RAPID_CLOUD],
         faults=[dict(fault_type="impeller_blockage", onset_hour=10.0, duration_h=3.0,
                       max_severity=0.7, severity_form="transient", ramp_s=600)]),

    dict(scenario_id=10, name="pv_degradation_plus_cloud", duration_hours=24.0,
         weather_condition="cloud_transient", clouds=[_RAPID_CLOUD],
         faults=[dict(fault_type="pv_degradation", onset_hour=10.0, duration_h=None,
                       max_severity=0.4, severity_form="persistent", ramp_s=3600)]),

    dict(scenario_id=11, name="dry_running_plus_cloud", duration_hours=24.0,
         weather_condition="cloud_transient", clouds=[_RAPID_CLOUD],
         faults=[dict(fault_type="dry_running", onset_hour=10.0, duration_h=2.0,
                       max_severity=0.9, severity_form="transient", ramp_s=300)]),
]

_BY_ID = {r["scenario_id"]: r for r in SCENARIO_LIBRARY}


def list_scenarios() -> List[Tuple[int, str, str]]:
    """Return (scenario_id, name, weather_condition) for every defined scenario."""
    return [(r["scenario_id"], r["name"], r["weather_condition"]) for r in SCENARIO_LIBRARY]


def build_scenario(scenario_id: int, start_time: pd.Timestamp, config) -> Tuple[pd.DatetimeIndex, List[CloudTransientSpec], List[FaultEvent], str, str]:
    """Instantiate a scenario recipe into concrete objects anchored at start_time.

    Returns: (times, cloud_specs, fault_events, weather_condition, scenario_name)
    """
    if scenario_id not in _BY_ID:
        raise KeyError(f"Unknown scenario_id {scenario_id}. Known ids: {sorted(_BY_ID)}")
    recipe = _BY_ID[scenario_id]

    n_samples = int(round(recipe["duration_hours"] * 3600.0 / config.simulation.output_dt_s))
    times = pd.date_range(start=start_time, periods=n_samples, freq=pd.Timedelta(seconds=config.simulation.output_dt_s))

    cloud_specs = []
    for c in recipe["clouds"]:
        t0 = start_time + pd.Timedelta(hours=c["onset_hour"])
        cloud_specs.append(CloudTransientSpec(
            t_start=t0, onset_s=c["onset_s"], hold_s=c["hold_s"],
            recovery_s=c["recovery_s"], depth=c["depth"], shape=c.get("shape", "smoothstep"),
        ))

    fault_events = []
    for f in recipe["faults"]:
        t0 = start_time + pd.Timedelta(hours=f["onset_hour"])
        if f["duration_h"] is None:
            t1 = times[-1]  # persistent fault: extends to end of scenario
        else:
            t1 = t0 + pd.Timedelta(hours=f["duration_h"])
        fault_events.append(FaultEvent(
            fault_type=f["fault_type"], start_time=t0, end_time=t1,
            max_severity=f["max_severity"], severity_form=f["severity_form"], ramp_s=f["ramp_s"],
        ))

    return times, cloud_specs, fault_events, recipe["weather_condition"], recipe["name"]
