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
