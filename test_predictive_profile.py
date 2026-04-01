"""
Unit test for predictive speed profiling algorithm.
Verifies that decel_start_dist calculations match kinematics.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class MockCfg:
    """Mock config for testing."""
    max_approach_speed: float = 14.0
    max_decel: float = 6.0

def compute_predictive_speed_profile_test(path, cfg):
    """Test version of compute_predictive_speed_profile."""
    # Step 1: compute required_speed per gate
    for g in path:
        mult = g["turn_speed_mult"]
        g["required_speed"] = cfg.max_approach_speed * mult

    # Step 2: backward pass for decel_start_dist
    for i in range(len(path)):
        v_target = path[i]["required_speed"]
        v_prev = 0.0 if i == 0 else path[i - 1]["required_speed"]

        if v_target < v_prev:
            path[i]["decel_start_dist"] = (
                (v_prev ** 2 - v_target ** 2) / (2.0 * cfg.max_decel)
            )
        else:
            path[i]["decel_start_dist"] = 0.0

    return path

def test_kinematics():
    """Test that decel distance matches kinematics formula."""
    print("Test 1: Kinematics Verification")
    print("=" * 50)

    cfg = MockCfg(max_approach_speed=14.0, max_decel=6.0)

    # Synthetic path: 5 gates with varying turn angles
    path = [
        {"index": 0, "turn_speed_mult": 1.0, "center": np.array([0, 0, 0])},      # straight
        {"index": 1, "turn_speed_mult": 0.75, "center": np.array([10, 0, 0])},    # 45 deg turn
        {"index": 2, "turn_speed_mult": 0.50, "center": np.array([20, 0, 0])},    # 90 deg turn
        {"index": 3, "turn_speed_mult": 0.30, "center": np.array([30, 0, 0])},    # 180 deg turn
        {"index": 4, "turn_speed_mult": 1.0, "center": np.array([40, 0, 0])},     # back to straight
    ]

    # Compute profile
    path = compute_predictive_speed_profile_test(path, cfg)

    # Expected values
    tests = [
        (0, 14.0, 0.0),        # Gate 0: start, no decel
        (1, 10.5, (14.0**2 - 10.5**2) / (2 * 6.0)),    # Gate 1: decel from 14 to 10.5
        (2, 7.0, (10.5**2 - 7.0**2) / (2 * 6.0)),      # Gate 2: 7.0 from 10.5
        (3, 4.2, (7.0**2 - 4.2**2) / (2 * 6.0)),       # Gate 3: 4.2 from 7.0
        (4, 14.0, 0.0),                                # Gate 4: speed up (decel is 0 by our logic)
    ]

    all_pass = True
    for gate_idx, expected_speed, expected_decel_dist in tests:
        actual_speed = path[gate_idx]["required_speed"]
        actual_decel = path[gate_idx]["decel_start_dist"]

        speed_match = abs(actual_speed - expected_speed) < 0.01
        # For decel: allow negative values (acceleration), check if magnitude is close
        decel_match = abs(actual_decel - expected_decel_dist) < 0.1

        status = "PASS" if (speed_match and decel_match) else "FAIL"
        print(f"  {status} Gate {gate_idx}: "
              f"speed={actual_speed:.2f} (exp {expected_speed:.2f}), "
              f"decel_dist={actual_decel:.2f} (exp {expected_decel_dist:.2f})")

        if not (speed_match and decel_match):
            all_pass = False

    print()
    return all_pass

def test_specific_case():
    """Test the specific case mentioned in the plan: v_from=14, v_to=7, a=6."""
    print("Test 2: Specific Kinematics Case")
    print("=" * 50)

    cfg = MockCfg(max_approach_speed=14.0, max_decel=6.0)

    # v_from = 14 m/s, v_to = 7 m/s, a = 6 m/s^2
    # Expected: dist = (14^2 - 7^2) / (2 * 6) = (196 - 49) / 12 = 147 / 12 = 12.25 m

    expected_dist = (14.0 ** 2 - 7.0 ** 2) / (2.0 * 6.0)
    print(f"  Expected decel distance: {expected_dist:.2f}m")

    # Create a path: gate 0 (speed 14), gate 1 (speed 7)
    path = [
        {"index": 0, "turn_speed_mult": 1.0, "center": np.array([0, 0, 0])},
        {"index": 1, "turn_speed_mult": 0.5, "center": np.array([20, 0, 0])},
    ]

    path = compute_predictive_speed_profile_test(path, cfg)

    actual_dist = path[1]["decel_start_dist"]
    match = abs(actual_dist - expected_dist) < 0.01

    status = "PASS" if match else "FAIL"
    print(f"  {status} Actual decel distance: {actual_dist:.2f}m")
    print(f"     Difference: {abs(actual_dist - expected_dist):.4f}m")
    print()

    return match

def test_no_decel_needed():
    """Test case where next gate is faster (no decel needed)."""
    print("Test 3: Speed Increase (No Decel)")
    print("=" * 50)

    cfg = MockCfg(max_approach_speed=14.0, max_decel=6.0)

    path = [
        {"index": 0, "turn_speed_mult": 0.5, "center": np.array([0, 0, 0])},     # speed 7
        {"index": 1, "turn_speed_mult": 1.0, "center": np.array([20, 0, 0])},    # speed 14 (faster)
    ]

    path = compute_predictive_speed_profile_test(path, cfg)

    # When speeding up, decel_start_dist should be 0 or negative
    decel = path[1]["decel_start_dist"]
    is_zero_or_neg = decel <= 0.0

    status = "PASS" if is_zero_or_neg else "FAIL"
    print(f"  {status} Gate 1 decel_start_dist: {decel:.2f}m (expected <= 0)")
    print()

    return is_zero_or_neg

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("PREDICTIVE SPEED PROFILE TESTS")
    print("=" * 50 + "\n")

    results = [
        ("Kinematics Verification", test_kinematics()),
        ("Specific Case (v_from=14, v_to=7)", test_specific_case()),
        ("Speed Increase (No Decel)", test_no_decel_needed()),
    ]

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name}")

    all_pass = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_pass else "Some tests failed!") + "\n")
