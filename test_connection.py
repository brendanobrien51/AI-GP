# This script is a minimal classic AirSim takeoff-and-land test.
# It connects to the simulator, enables API control, arms the drone,
# performs a takeoff, waits a few seconds in the air, and then lands.
# It is the simplest possible flight check for verifying that the AirSim
# simulator is responding correctly to Python commands.

import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

import time
time.sleep(3)

print("Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
