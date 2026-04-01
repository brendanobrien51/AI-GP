"""
Parameter Optimizer for V16 Drone Racer
========================================
Uses Optuna to perform Bayesian hyperparameter optimization on V16.

Each trial:
1. Writes a trial config to config_trial.yaml
2. Spawns V16 subprocess with --config config_trial.yaml --headless --telemetry_out
3. Reads back JSON telemetry with lap_time and missed_gates
4. Returns objective score = lap_time + 5.0 * missed_gates
5. Saves best config to config_best.yaml when new best is found

Objective: minimize lap time while avoiding missed gates
"""

import json
import math
import optuna
import subprocess
import sys
import time
import yaml
from pathlib import Path


# Configuration paths
CONFIG_PATH = Path("C:/Users/brend/AI-GP/config.yaml")
CONFIG_TRIAL_PATH = Path("C:/Users/brend/AI-GP/config_trial.yaml")
CONFIG_BEST_PATH = Path("C:/Users/brend/AI-GP/config_best.yaml")
TRIAL_RESULT_PATH = Path("C:/Users/brend/AI-GP/trial_result.json")

# Optuna settings
OPTUNA_STORAGE = "sqlite:///optuna_racing.db"
STUDY_NAME = "v16_racing"
N_TRIALS = 100
MISSED_GATE_PENALTY = 100.0   # HEAVILY penalize missed gates
INCOMPLETE_COURSE_PENALTY = 500.0  # Massive penalty if not all gates completed
COLLISION_RETRY_PENALTY = 20.0  # Penalize each backup/retry (indicates bad params)
CV_DETECTION_REWARD = -0.5  # Reward high CV detection rate (negative = reduces score)
MIN_GATES_FOR_VALID_TRIAL = 10  # Trial is only valid if at least this many gates completed
MAX_TRIAL_TIME = 300  # Kill trial if it takes >5 minutes (stuck drone)

def load_config(path: Path) -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)

def save_config(cfg: dict, path: Path) -> None:
    """Save dict to YAML configuration file."""
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

def suggest_params(trial: optuna.Trial, base_cfg: dict) -> dict:
    """
    Suggest all tunable parameters using Optuna TPE sampler.
    Uses FINER STEPS and TIGHTER RANGES for precise learning.
    """
    cfg = base_cfg.copy()

    # Speed envelope - TIGHTER RANGES, FINER STEPS
    cfg["max_approach_speed"] = trial.suggest_float(
        "max_approach_speed", 12.0, 16.0, step=0.1  # Was 10-18, step 0.5
    )
    cfg["max_center_speed"] = trial.suggest_float(
        "max_center_speed", 8.0, 12.0, step=0.1    # Was 6-14, step 0.5
    )
    cfg["max_exit_speed"] = trial.suggest_float(
        "max_exit_speed", 14.0, 18.0, step=0.1     # Was 12-20, step 0.5
    )
    cfg["min_center_speed"] = trial.suggest_float(
        "min_center_speed", 4.0, 7.0, step=0.1     # Was 3-8, step 0.5
    )

    # Turn-angle adaptive speed curve - FINER CONTROL
    cfg["turn_speed_90deg"] = trial.suggest_float(
        "turn_speed_90deg", 0.40, 0.70, step=0.02  # Was 0.30-0.70, step 0.05
    )
    cfg["turn_speed_180deg"] = trial.suggest_float(
        "turn_speed_180deg", 0.20, 0.40, step=0.02 # Was 0.18-0.45, step 0.05
    )
    cfg["turn_taper_dist"] = trial.suggest_float(
        "turn_taper_dist", 10.0, 16.0, step=0.2    # Was 8-20, step 1.0
    )

    # Predictive braking - FINER TUNING
    cfg["max_decel"] = trial.suggest_float(
        "max_decel", 5.0, 8.0, step=0.1            # Was 3-10, step 0.5
    )

    # Lookahead - TIGHTER RANGES
    cfg["lookahead_min"] = trial.suggest_float(
        "lookahead_min", 5.0, 8.0, step=0.1        # Was 4-10, step 0.5
    )
    cfg["lookahead_max"] = trial.suggest_float(
        "lookahead_max", 18.0, 24.0, step=0.2      # Was 14-28, step 1.0
    )
    cfg["lookahead_speed_scale"] = trial.suggest_float(
        "lookahead_speed_scale", 1.2, 2.0, step=0.05  # Was 0.8-2.5, step 0.1
    )

    # Lateral gains - FINER STEPS
    cfg["lateral_gain_approach"] = trial.suggest_float(
        "lateral_gain_approach", 2.0, 6.0, step=0.2    # Was 1-8, step 0.5
    )
    cfg["lateral_gain_center"] = trial.suggest_float(
        "lateral_gain_center", 4.0, 10.0, step=0.2     # Was 2-12, step 0.5
    )
    cfg["lateral_speed_scale"] = trial.suggest_float(
        "lateral_speed_scale", 10.0, 16.0, step=0.2    # Was 6-20, step 1.0
    )

    # Cushion (gate centering force) - FOCUSED RANGE
    cfg["cushion_gain"] = trial.suggest_float(
        "cushion_gain", 6.0, 12.0, step=0.2        # Was 4-16, step 0.5
    )
    cfg["cushion_dist"] = trial.suggest_float(
        "cushion_dist", 8.0, 12.0, step=0.2        # Was 5-14, step 0.5
    )
    cfg["cushion_speed_scale"] = trial.suggest_float(
        "cushion_speed_scale", 6.0, 10.0, step=0.2 # Was 4-14, step 0.5
    )

    # CV detection - FINER CONTROL
    cfg["canny_low"] = trial.suggest_int("canny_low", 40, 70, step=2)     # Was 30-80
    cfg["canny_high"] = trial.suggest_int("canny_high", 100, 160, step=2) # Was 90-180
    cfg["cv_steer_blend"] = trial.suggest_float(
        "cv_steer_blend", 0.3, 0.7, step=0.05      # Was 0.2-0.8, step 0.1
    )
    cfg["cv_steer_gain_center"] = trial.suggest_float(
        "cv_steer_gain_center", 2.0, 6.0, step=0.2 # Was 1-8, step 0.5
    )

    # Dynamic CV search window - FINE TUNING
    cfg["sky_mask_near"] = trial.suggest_float(
        "sky_mask_near", 0.25, 0.45, step=0.02     # Was 0.20-0.50, step 0.05
    )
    cfg["sky_mask_far"] = trial.suggest_float(
        "sky_mask_far", 0.10, 0.20, step=0.02      # Was 0.05-0.25, step 0.05
    )
    cfg["contour_min_area_far"] = trial.suggest_int(
        "contour_min_area_far", 50, 80, step=2     # Was 30-100, step 10
    )

    return cfg

def run_trial_lap(trial_cfg: dict) -> dict:
    """
    Write trial config, launch racer subprocess, and read back result.

    Returns dict with keys: lap_time (float), missed_gates (int), error (str or None)
    Also includes: gates_completed, total_gates, total_retries, avg_cv_detection_rate
    """
    save_config(trial_cfg, CONFIG_TRIAL_PATH)

    cmd = [
        sys.executable,
        "C:/Users/brend/AI-GP/airsim_contour_trackerV16.py",
        "--config", str(CONFIG_TRIAL_PATH),
        "--headless",
        "--telemetry_out", str(TRIAL_RESULT_PATH),
    ]

    try:
        print(f"  [SUBPROCESS] Launching V16 trial...")
        proc = subprocess.run(
            cmd, timeout=MAX_TRIAL_TIME, capture_output=True, text=True, cwd="C:/Users/brend/AI-GP"
        )
        if proc.returncode != 0:
            print(f"    Error: subprocess returned {proc.returncode}")
            if proc.stderr:
                print(f"    stderr: {proc.stderr[:200]}")
            return {"lap_time": float("inf"), "missed_gates": 99, "error": "nonzero_exit"}

    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Trial exceeded {MAX_TRIAL_TIME}s (drone stuck?)")
        return {"lap_time": float("inf"), "missed_gates": 99, "error": "timeout"}
    except Exception as e:
        print(f"  [ERROR] Failed to launch subprocess: {e}")
        return {"lap_time": float("inf"), "missed_gates": 99, "error": str(e)}

    if not TRIAL_RESULT_PATH.exists():
        print(f"  [ERROR] No result file at {TRIAL_RESULT_PATH}")
        return {"lap_time": float("inf"), "missed_gates": 99, "error": "no_result_file"}

    try:
        with open(TRIAL_RESULT_PATH) as f:
            result = json.load(f)
        return result
    except Exception as e:
        print(f"  [ERROR] Failed to read result JSON: {e}")
        return {"lap_time": float("inf"), "missed_gates": 99, "error": "json_parse_error"}

def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization.

    Scoring strategy:
    1. Penalize incomplete courses (not finishing all gates) heavily
    2. Penalize each missed gate heavily (100s per gate)
    3. Only reward fast lap times if all gates are completed cleanly
    4. Prefer stability: smoother, more reliable runs over risky fast runs
    """
    base_cfg = load_config(CONFIG_PATH)
    trial_cfg = suggest_params(trial, base_cfg)

    print(f"\n[TRIAL {trial.number}]")
    result = run_trial_lap(trial_cfg)

    lap_time = result.get("lap_time", float("inf"))
    missed_gates = result.get("missed_gates", 99)
    gates_completed = result.get("gates_completed", 0)
    total_gates = result.get("total_gates", 12)
    total_retries = result.get("total_retries", 0)
    avg_cv_detection = result.get("avg_cv_detection_rate", 0.0)
    error = result.get("error")

    # Scoring logic (multi-factor):
    if math.isinf(lap_time) or error:
        # Crash or failure - worst possible score
        score = 9999.0
        print(f"  FAILED: {error}")
    elif gates_completed < MIN_GATES_FOR_VALID_TRIAL:
        # Didn't complete minimum gates - heavily penalized
        score = 9999.0 - gates_completed  # Still rank by gates completed
        print(f"  INVALID: only {gates_completed}/{total_gates} gates (threshold: {MIN_GATES_FOR_VALID_TRIAL})")
    elif gates_completed < total_gates:
        # Incomplete course - massive penalty
        score = lap_time + (total_gates - gates_completed) * MISSED_GATE_PENALTY + INCOMPLETE_COURSE_PENALTY
        print(f"  INCOMPLETE: {gates_completed}/{total_gates} gates, "
              f"lap_time={lap_time:.2f}s, missed={missed_gates}, score={score:.2f}s")
    else:
        # Complete course - normal scoring with additional factors
        base_score = lap_time + missed_gates * MISSED_GATE_PENALTY
        retry_penalty = total_retries * COLLISION_RETRY_PENALTY
        cv_bonus = avg_cv_detection * CV_DETECTION_REWARD  # Negative = reward
        score = base_score + retry_penalty + cv_bonus

        print(f"  COMPLETE: {gates_completed}/{total_gates} gates, "
              f"lap={lap_time:.1f}s, missed={missed_gates}, retries={total_retries}, "
              f"cv_detect={avg_cv_detection:.1%}, score={score:.2f}s")

    trial.set_user_attr("lap_time", lap_time if not math.isinf(lap_time) else None)
    trial.set_user_attr("missed_gates", missed_gates)
    trial.set_user_attr("gates_completed", gates_completed)
    trial.set_user_attr("total_gates", total_gates)
    trial.set_user_attr("total_retries", total_retries)
    trial.set_user_attr("avg_cv_detection_rate", avg_cv_detection)

    # Auto-save best config (check if this is better than previous best)
    study = trial.study
    try:
        best_val = study.best_value
        is_new_best = score < best_val
    except Exception:
        # First trial or no valid trials yet
        is_new_best = not math.isinf(score) and score < 9999.0
        best_val = None

    if is_new_best:
        save_config(trial_cfg, CONFIG_BEST_PATH)
        if best_val is None:
            print(f"\n  [FIRST] FIRST VALID TRIAL: {score:.2f}s "
                  f"(lap={lap_time:.1f}s, gates={gates_completed}/{total_gates}, "
                  f"missed={missed_gates}, retries={total_retries}, cv={avg_cv_detection:.1%})")
        else:
            improvement = best_val - score
            print(f"\n  [BEST] NEW BEST: {score:.2f}s (improved by {improvement:.2f}s) "
                  f"(lap={lap_time:.1f}s, gates={gates_completed}/{total_gates}, "
                  f"missed={missed_gates}, retries={total_retries}, cv={avg_cv_detection:.1%})")
        print(f"    Saved to {CONFIG_BEST_PATH}\n")

    # Show rolling average trend (last 5 valid trials)
    recent_trials = study.trials[-5:] if len(study.trials) >= 5 else study.trials
    recent_scores = [t.value for t in recent_trials if t.value is not None and t.value < 9999]
    if len(recent_scores) >= 2:
        trend_avg = sum(recent_scores) / len(recent_scores)
        print(f"  [TREND] Last {len(recent_scores)} valid trials avg: {trend_avg:.2f}s")

    return score

def main():
    """Run optimization loop."""
    print(f"\n{'='*70}")
    print(f"V16 Drone Racing Parameter Optimizer")
    print(f"{'='*70}")
    print(f"Database: {OPTUNA_STORAGE}")
    print(f"Base config: {CONFIG_PATH}")
    print(f"Trials per run: {N_TRIALS}")
    print(f"{'='*70}\n")

    # Load TPE study if it exists to extract best params for CMA-ES warm-start
    base_cfg = load_config(CONFIG_PATH)
    best_params_from_tpe = None
    tpe_trials_count = 0

    try:
        tpe_study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=OPTUNA_STORAGE,
        )
        if len(tpe_study.trials) > 0:
            tpe_trials_count = len(tpe_study.trials)
            print(f"[WARM-START] Loaded TPE study: {tpe_trials_count} trials, best score {tpe_study.best_value:.2f}s")
            best_params_from_tpe = tpe_study.best_params
            print(f"[WARM-START] Will seed CMA-ES from trial #{tpe_study.best_trial.number}")
    except Exception as e:
        print(f"[WARM-START] No prior TPE study found, starting fresh")

    # Create CMA-ES study for refinement phase
    print(f"\n[CMA-ES] Starting refinement phase with Covariance Matrix Adaptation")
    sampler = optuna.samplers.CmaEsSampler(
        x0=best_params_from_tpe,  # Only pass the tuned params, not full config
        sigma0=0.2,               # Initial step size (shrinks each trial toward optimum)
        seed=42,
        warn_independent_sampling=False,
    )
    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_cma_phase",
        storage=OPTUNA_STORAGE,
        sampler=sampler,
        direction="minimize",
        load_if_exists=True,
    )

    # Check if resuming CMA phase
    if len(study.trials) > 0:
        print(f"[CMA-ES] Resuming refinement: {len(study.trials)} CMA trials already done")
        try:
            best_trial = study.best_trial
            print(f"[CMA-ES] Current best: {study.best_value:.2f}s (trial {best_trial.number})\n")
        except Exception:
            print(f"[CMA-ES] (No valid trials yet)\n")
    else:
        print(f"[CMA-ES] Starting fresh refinement phase\n")
        # Enqueue the best config from TPE as the first trial to warm-start CMA-ES
        if best_params_from_tpe:
            print(f"[CMA-ES] Enqueueing best-from-TPE as trial 0\n")
            study.enqueue_trial(best_params_from_tpe)

    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    # Print results
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total trials: {len(study.trials)}")

    try:
        best_trial = study.best_trial
        print(f"Best score: {study.best_value:.2f}s")
        print(f"Best trial: #{best_trial.number}")
        best_attrs = best_trial.user_attrs
        lap_time = best_attrs.get('lap_time')
        missed = best_attrs.get('missed_gates')
        gates_comp = best_attrs.get('gates_completed')
        gates_tot = best_attrs.get('total_gates')
        retries = best_attrs.get('total_retries', 0)
        cv_rate = best_attrs.get('avg_cv_detection_rate', 0.0)

        print(f"\nBest run stats:")
        print(f"  Lap time: {lap_time:.2f}s" if lap_time else "  Lap time: N/A")
        print(f"  Missed gates: {missed}")
        print(f"  Gates completed: {gates_comp}/{gates_tot}")
        print(f"  Collision retries: {retries}")
        print(f"  Avg CV detection: {cv_rate:.1%}")

        # Show scoring breakdown
        if lap_time and gates_comp == gates_tot:
            missed_penalty = missed * MISSED_GATE_PENALTY
            retry_penalty = retries * COLLISION_RETRY_PENALTY
            cv_bonus = cv_rate * CV_DETECTION_REWARD
            total = lap_time + missed_penalty + retry_penalty + cv_bonus

            if missed == 0 and retries == 0:
                print(f"\n  Scoring: {lap_time:.2f}s (perfect run!)")
            else:
                print(f"\n  Scoring breakdown:")
                print(f"    Lap time: {lap_time:.2f}s")
                if missed > 0:
                    print(f"    Missed gates penalty: {missed} x {MISSED_GATE_PENALTY:.0f}s = {missed_penalty:.0f}s")
                if retries > 0:
                    print(f"    Collision retry penalty: {retries} x {COLLISION_RETRY_PENALTY:.0f}s = {retry_penalty:.0f}s")
                if cv_bonus != 0:
                    print(f"    CV detection bonus: {cv_rate:.1%} x {CV_DETECTION_REWARD:.1f} = {cv_bonus:.0f}s")
                print(f"    Total score: {total:.2f}s")
    except Exception as e:
        print(f"[No completed trials yet: {e}]")

    print(f"\nTop 5 trials:")
    for i, trial in enumerate(sorted(study.trials, key=lambda t: t.value)[:5]):
        print(f"  {i+1}. Trial #{trial.number}: {trial.value:.2f}s")

    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Save best config
    best_cfg = {**load_config(CONFIG_PATH), **study.best_params}
    save_config(best_cfg, CONFIG_BEST_PATH)
    print(f"\nBest config saved to: {CONFIG_BEST_PATH}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
