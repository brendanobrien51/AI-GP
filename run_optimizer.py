"""
Auto-restart wrapper for optimizer that handles AirSim crashes and resets.
Monitors optimize_params.py and auto-restarts AirSim if needed.
"""

import subprocess
import time
import psutil
import os

def airsim_running():
    """Check if AirSim is running."""
    for proc in psutil.process_iter(['name']):
        try:
            if 'airsim' in proc.info['name'].lower() or 'drone_racing' in proc.info['name'].lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

def restart_airsim():
    """Kill AirSim and restart it."""
    print("\n[AUTO-RESTART] Restarting AirSim...")

    # Kill existing AirSim processes
    for proc in psutil.process_iter(['name']):
        try:
            if 'airsim' in proc.info['name'].lower() or 'drone_racing' in proc.info['name'].lower():
                proc.kill()
                print(f"  Killed {proc.info['name']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    time.sleep(2)
    print("  [NOTE] Please manually open AirSim, select 'Soccer Field - Easy', and press BACKSPACE")
    print("  Waiting for AirSim to be ready...")

    # Wait for user to manually launch AirSim
    for i in range(120):  # 2 minute timeout
        if airsim_running():
            print("  AirSim detected! Continuing...\n")
            time.sleep(5)
            return True
        time.sleep(1)
        if i % 10 == 0:
            print(f"  Still waiting ({i}s)...")

    print("  [ERROR] Timeout waiting for AirSim. Exiting.")
    return False

def main():
    """Run optimizer with auto-restart logic."""
    print("=" * 70)
    print("V16 Optimizer with Auto-Restart")
    print("=" * 70)
    print("\nIMPORTANT:")
    print("1. Open AirSim Drone Racing Lab")
    print("2. Select 'Soccer Field - Easy' as the course")
    print("3. Press BACKSPACE to reset drone position (if needed)")
    print("4. When ready, I'll start the optimizer")
    print("\nPress ENTER to continue...")
    input()

    # Check AirSim is running
    if not airsim_running():
        print("\n[ERROR] AirSim is not running. Please launch it first.")
        return

    print("\nStarting optimizer...\n")

    # Run optimizer
    while True:
        try:
            cmd = [
                "python",
                "C:/Users/brend/AI-GP/optimize_params.py"
            ]
            proc = subprocess.Popen(cmd, cwd="C:/Users/brend/AI-GP")
            proc.wait()

            # If optimizer exits normally, we're done
            print("\n[DONE] Optimizer completed successfully!")
            break

        except KeyboardInterrupt:
            print("\n[STOPPED] Optimizer stopped by user.")
            break
        except Exception as e:
            print(f"\n[ERROR] Optimizer failed: {e}")

            # Check if AirSim crashed
            if not airsim_running():
                print("[AUTO-RESTART] AirSim appears to have crashed.")
                if restart_airsim():
                    print("[AUTO-RESTART] Resuming optimizer...")
                    time.sleep(5)
                    continue
                else:
                    break
            else:
                # Try to continue
                print("[RETRY] Retrying optimizer...")
                time.sleep(5)
                continue

if __name__ == "__main__":
    main()
