from dataclasses import dataclass

@dataclass
class Scenario:
    scenario_id: str
    weather_type: str      # "clear" or "cloud_transient"
    fault_type: str        # "none", "bearing_wear", "impeller_blockage", "pv_degradation"
    fault_severity: float
    is_dry_running: bool
    seed: int = 42         # Strict reproducibility per scenario

def get_scenario_library():
    """Returns the 11 foundational research scenarios."""
    return {
        "SC-01": Scenario("SC-01", "clear", "none", 0.0, False),
        "SC-02": Scenario("SC-02", "cloud_transient", "none", 0.0, False), # Normal + Cloudy
        "SC-03": Scenario("SC-03", "cloud_transient", "none", 0.0, False), # Normal + Rapid Cloud
        "SC-04": Scenario("SC-04", "clear", "bearing_wear", 0.5, False),
        "SC-05": Scenario("SC-05", "clear", "impeller_blockage", 0.5, False),
        "SC-06": Scenario("SC-06", "clear", "pv_degradation", 0.2, False),
        "SC-07": Scenario("SC-07", "clear", "none", 0.0, True),
        "SC-08": Scenario("SC-08", "cloud_transient", "bearing_wear", 0.5, False),
        "SC-09": Scenario("SC-09", "cloud_transient", "impeller_blockage", 0.5, False),
        "SC-10": Scenario("SC-10", "cloud_transient", "pv_degradation", 0.2, False),
        "SC-11": Scenario("SC-11", "cloud_transient", "none", 0.0, True),
    }