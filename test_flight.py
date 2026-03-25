import airsim
import time

print("Initializing connection to AirSim...")
# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control and arm the drone's motors
client.enableApiControl(True)
client.armDisarm(True)

print("Motors armed. Taking off!")
# Take off and wait for the action to complete
client.takeoffAsync().join()

print("Hovering at target altitude...")
# Hold the hover state for 5 seconds
time.sleep(5)

print("Test complete. Landing...")
# Land the drone safely
client.landAsync().join()

# Disarm motors and release control
client.armDisarm(False)
client.enableApiControl(False)
print("Drone secured. Connection closed.")
