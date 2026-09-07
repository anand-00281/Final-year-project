import pandas as pd
import numpy as np

def run_physical_acceptance_tests(df):
    """
    Automated validation suite checking fundamental conservation laws and operational bounds.
    """
    print("Running Automated Physics Acceptance Tests...")
    
    # Defensive column mapping to standardize names
    df = df.copy()
    if 'flow_lpm' not in df.columns:
        for col in df.columns:
            if 'flow' in col.lower():
                df['flow_lpm'] = df[col]
                break
    if 'rpm_true' not in df.columns:
        for col in df.columns:
            if 'rpm' in col.lower() or 'omega' in col.lower():
                df['rpm_true'] = df[col]
                break

    results = {}
    
    # Filter out inactive night periods where power is zero
    active_df = df[df['irradiance_true'] > 50.0].copy()
    
    if len(active_df) == 0:
        print("Warning: No active daylight rows found for testing.")
        return False

    # Test 1: Power Hierarchy Conservation (P_hyd < P_mech < P_elec <= P_PV)
    p_hyd = (active_df['flow_lpm'] / 60000.0) * 9810.0 * 20.0 # Approximate hydraulic power
    p_elec = active_df['p_elec_measured']
    
    power_violation_count = (p_hyd > p_elec).sum()
    results['Power Conservation'] = power_violation_count == 0
    print(v_status("1. Power Hierarchy (P_hyd <= P_elec)", results['Power Conservation']))

    # Test 2: Plausible RPM Bounds (0 to 6000 RPM for BLDC pump)
    rpm_bounds_ok = (active_df['rpm_true'] >= 0.0).all() and (active_df['rpm_true'] <= 6000.0).all()
    results['RPM Bounds'] = rpm_bounds_ok
    print(v_status("2. Motor RPM Limits [0, 6000]", results['RPM Bounds']))

    # Test 3: Flow Non-Negativity
    flow_ok = (active_df['flow_lpm'] >= 0.0).all()
    results['Flow Non-Negativity'] = flow_ok
    print(v_status("3. Flow Non-Negativity (>= 0 LPM)", results['Flow Non-Negativity']))

    all_passed = all(results.values())
    if all_passed:
        print("\nSTATUS: ALL PHYSICAL ACCEPTANCE TESTS PASSED SUCCESSFULLY [PASS].")
    else:
        print("\nSTATUS: ONE OR MORE PHYSICAL ACCEPTANCE TESTS FAILED [FAIL].")
        
    return all_passed

def v_status(test_name, passed):
    status = "[ PASS ]" if passed else "[ FAIL ]"
    return f"  {test_name}: {status}"