"""
main.py

Thin driver only -- no equations, no scenario logic. Demonstrates the
intended Phase-1 workflow:

    1. Run the physics validation suite on small scenarios (Section 16 /
       constraint 14) BEFORE anything else.
    2. Run one example scenario end-to-end.
    3. Plot it.
    4. Export CSV.

This is NOT the 10-day master dataset generation script (that is a
separate, larger step for Phase 2 once V2 itself is approved) -- this
main.py is the "smoke test + example execution" deliverable for PART E.
"""

import pandas as pd
from pathlib import Path

from solar_pump_digital_twin.config.parameters import DigitalTwinConfig
from solar_pump_digital_twin.validation.physics_checks import run_full_validation_suite
from solar_pump_digital_twin.simulation.digital_twin import run_scenario
from solar_pump_digital_twin.visualization.plots import plot_scenario
from solar_pump_digital_twin.scenarios.scenario_library import list_scenarios

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "generated"


def main():
    cfg = DigitalTwinConfig()
    start_time = pd.Timestamp("2026-05-01")

    print("=" * 70)
    print("Available scenarios:")
    for sid, name, wc in list_scenarios():
        print(f"  {sid:2d}  {name:32s} ({wc})")

    print("\n" + "=" * 70)
    print("Running physics validation suite (small scenarios) BEFORE any")
    print("large dataset generation...")
    val_results = run_full_validation_suite(cfg, start_time)
    print(val_results.to_string(index=False))
    if (val_results["result"] == "FAIL").any():
        print("\n*** One or more validation checks FAILED. Do not proceed to ***")
        print("*** large-scale dataset generation until this is resolved.  ***")

    print("\n" + "=" * 70)
    scenario_id = 9
    print(f"Running example scenario {scenario_id}...")
    df = run_scenario(scenario_id, start_time, cfg)
    print("shape:", df.shape)
    print(df.head())

    out_csv = OUTPUT_DIR / f"scenario_{scenario_id}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    out_png = OUTPUT_DIR / f"scenario_{scenario_id}_plot.png"
    plot_scenario(df, title=f"Scenario {scenario_id}", savepath=str(out_png))


if __name__ == "__main__":
    main()
