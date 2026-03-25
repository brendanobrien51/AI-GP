import airsim
import time

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

# Define waypoints as (x, y, z) in meters
# In AirSim, z is negative for altitude (NED coordinate system)
waypoints = [
    (5,  0,  -3),   # Gate 1 - fly forward and up
    (10, 5,  -3),   # Gate 2 - turn right
    (10, 10, -5),   # Gate 3 - go higher
    (5,  10, -5),   # Gate 4 - fly back left
    (0,  0,  -3),   # Gate 5 - return to start
]

speed = 5  # m/s

for i, (x, y, z) in enumerate(waypoints):
    print(f"Flying to gate {i+1}: x={x}, y={y}, z={z}")
    client.moveToPositionAsync(x, y, z, speed).join()
    time.sleep(1)  # brief pause at each gate

print("Course complete! Landing...")
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Done.")
