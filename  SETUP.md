# SETUP

This guide is for a new contributor setting up this repository from scratch on Windows.

This project currently uses two simulator/API paths:

- **AirSim Drone Racing Lab**: the main and recommended workflow
- **Classic AirSim**: older scripts kept for experiments and debugging

For most users, start with **AirSim Drone Racing Lab**.

---

# 1. Install Required Software

## Python 3.10

Download Python 3.10 here:

- <https://www.python.org/downloads/release/python-31011/>

During installation:

- enable **Add Python to PATH**

Verify install:

```powershell
python --version


Git-
Download Git for Windows here:
	https://git-scm.com/download/win
Verify install:
	git --version


Clone the Repository-
	git clone <repo-url>
	cd C:\Users\<YourUser>\source\repos\brendanobrien51\AI-GP
Replace <repo-url> with your actual Git remote URL.

Create and Activate a Virtual Environment-
Create the virtual environment:
	python -m venv venv

Activate it:
	.\venv\Scripts\Activate.ps1

If PowerShell blocks activation, run this once:
	Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Then activate again:
	.\venv\Scripts\Activate.ps1

Upgrade pip and packaging tools:
	python -m pip install --upgrade pip setuptools wheel

Install Python Dependencies-
Install all Python packages used in this repository:
	pip install numpy
	pip install opencv-python
	pip install airsimdroneracinglab
	pip install gymnasium
	pip install stable-baselines3
	pip install tensorboard
	pip install tqdm
	pip install rich
	pip install airsim

Dependency list-
Required for the main AirSim Drone Racing Lab workflow:
	numpy
	opencv-python
	airsimdroneracinglab
	gymnasium
	stable-baselines3
	tensorboard
	tqdm
	rich
Required for older classic AirSim scripts
	airsim

Package links:
NumPy: https://pypi.org/project/numpy/
OpenCV Python: https://pypi.org/project/opencv-python/
AirSim Drone Racing Lab Python API: https://pypi.org/project/airsimdroneracinglab/
Gymnasium: https://pypi.org/project/gymnasium/
Stable-Baselines3: https://pypi.org/project/stable-baselines3/
TensorBoard: https://pypi.org/project/tensorboard/
tqdm: https://pypi.org/project/tqdm/
rich: https://pypi.org/project/rich/
Classic AirSim Python API: https://pypi.org/project/airsim/

Install AirSim Drone Racing Lab:
GitHub repo: https://github.com/microsoft/AirSim-Drone-Racing-Lab
Releases page: https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases
Python docs: https://microsoft.github.io/AirSim-Drone-Racing-Lab/
AirSim settings docs: https://microsoft.github.io/AirSim/settings/
Download the simulator
Go to the releases page and download the Windows Binaries release.
Releases:
	https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases
Extract the simulator somewhere simple, for example:
	C:\AirSim-Drone-Racing-Lab

Create the AirSim Settings File
	C:\Users\<YourUser>\Documents\AirSim\settings.json
Create the folder if it does not exist
	C:\Users\<YourUser>\Documents\AirSim
Create settings.json
	C:\Users\<YourUser>\Documents\AirSim\settings.json
Use this minimal starter config:
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ViewMode": "Fpv"
}
This is enough for most initial testing.

Start the Simulator
After extracting AirSim Drone Racing Lab:
open the simulator folder
launch the provided Windows executable or run.bat
wait until the environment is fully loaded
Do this before running the Python scripts in this repo.

Recommended Startup Order
Every time you want to work on the project:
start AirSim Drone Racing Lab
wait for the simulator to finish loading
open PowerShell in the repo
activate the virtual environment
run a sanity-check script
then run the CV or RL scripts

Main Repository Workflows
AirSim Drone Racing Lab workflow
This is the recommended path.
Main files
airsim_contour_tracker.py
adrl_env.py
train_rl.py
run_policy.py
test_env.py

First commands to try
Test the RL environment connection
	python test_env.py

Run the CV racer
	python airsim_contour_tracker.py

Train the RL model
	python train_rl.py

Run a saved PPO policy
	python run_policy.py

Classic AirSim workflow
These scripts use the older airsim package.
Main files
test_connection.py
test_flight.py
debug_vision.py
drone_vision.py
waypoint_flight.py

First commands to try
Simple connection/takeoff test
	python test_connection.py

Vision debugging
	python debug_vision.py
	python drone_vision.py

10. What Each Main File Does
Current recommended path
airsim_contour_tracker.py

main CV-based gate racing script
adrl_env.py

Gymnasium RL environment
train_rl.py

PPO training entry point
run_policy.py

runs a saved PPO model
test_env.py

quick environment sanity test
Older experimental path
test_connection.py

classic AirSim connection/takeoff test
test_flight.py
simple flight test
drone_vision.py

older CV experiments
debug_vision.py

debugging camera/vision output
waypoint_flight.py
waypoint flight testing

Full Install Command Summary
	git clone <repo-url>
	cd C:\Users\<YourUser>\source\repos\brendanobrien51\AI-GP
python -m venv venv
	.\venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip setuptools wheel
	pip install numpy opencv-python airsimdroneracinglab gymnasium stable-baselines3 tensorboard tqdm rich airsim
If PowerShell blocks activation
	Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
	.\venv\Scripts\Activate.ps1

If RL training complains about missing extras
	pip install tensorboard tqdm rich

If airsimdroneracinglab is missing
	pip install airsimdroneracinglab

If airsim is missing
	pip install airsim

Important Note About the Existing venv
Do not rely on the checked-in venv folder.
It is machine-specific and may point to a Python path that does not exist on another machine.
A new user should always create a fresh local venv.
If needed, delete and recreate it:
Remove-Item -Recurse -Force .\venv
	python -m venv venv
	.\venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip setuptools wheel
	pip install numpy opencv-python airsimdroneracinglab gymnasium stable-baselines3 tensorboard tqdm rich airsim

Troubleshooting
The simulator is open but the script does nothing
Check:
AirSim Drone Racing Lab fully loaded before starting Python
the correct virtual environment is activated
you are using the correct API package for the script you ran
ModuleNotFoundError: No module named 'airsimdroneracinglab'
Install the ADRL package:
	pip install airsimdroneracinglab
ModuleNotFoundError: No module named 'airsim'
Install classic AirSim support:
	pip install airsim
No gate objects found in scene
Usually means:
wrong simulator environment
simulator not fully loaded
the script connected before the level finished initializing
RL training complains about TensorBoard or progress bar packages
Install the missing training extras:
	pip install tensorboard tqdm rich
PowerShell says scripts are disabled
Run:
	Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
Then reactivate:
	.\venv\Scripts\Activate.ps1

Recommended Starting Points by Role
If you are working on perception / computer vision
Start here:
	python airsim_contour_tracker.py
If you are working on reinforcement learning
Start here:
	python test_env.py
	python train_rl.py

If you are just validating simulator connectivity
Start here:
python test_connection.py

Notes
This repo is still experimental and mixes:
CV-first control
simulator-specific control code
RL environment work
older prototype scripts
If something looks duplicated or inconsistent, start with the AirSim Drone Racing Lab workflow first.

References
AirSim Drone Racing Lab repo: https://github.com/microsoft/AirSim-Drone-Racing-Lab
AirSim Drone Racing Lab releases: https://github.com/microsoft/AirSim-Drone-Racing-Lab/releases
AirSim Drone Racing Lab docs: https://microsoft.github.io/AirSim-Drone-Racing-Lab/
AirSim settings docs: https://microsoft.github.io/AirSim/settings/
Stable-Baselines3: https://pypi.org/project/stable-baselines3/
Classic AirSim package: https://pypi.org/project/airsim/
Python 3.10: https://www.python.org/downloads/release/python-31011/
Git for Windows: https://git-scm.com/download/win
