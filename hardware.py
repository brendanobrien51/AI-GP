"""
Hardware Abstraction Layer for Drone Racing
============================================
Decouples flight logic from simulator/hardware specifics.

- DroneInterface: Abstract base class defining all methods
- AirSimInterface: Wraps airsimdroneracinglab.MultirotorClient
- HardwareInterface: Stub for real drone (MAVLink/ROS integration point)
"""

from abc import ABC, abstractmethod
import math
import time
import numpy as np
import re


class DroneInterface(ABC):
    """Abstract interface for drone control and sensing."""

    @abstractmethod
    def get_image(self) -> np.ndarray:
        """Return latest BGR camera frame as (H, W, 3) uint8 numpy array."""
        ...

    @abstractmethod
    def get_position(self) -> np.ndarray:
        """Return (x, y, z) position as float64 numpy array (NED frame)."""
        ...

    @abstractmethod
    def get_velocity(self) -> np.ndarray:
        """Return (vx, vy, vz) velocity as float64 numpy array."""
        ...

    @abstractmethod
    def get_orientation(self):
        """Return orientation as an object with w_val/x_val/y_val/z_val."""
        ...

    @abstractmethod
    def get_yaw(self) -> float:
        """Return current yaw in radians."""
        ...

    @abstractmethod
    def move_by_velocity(self, vx: float, vy: float, vz: float,
                         duration: float, yaw_deg: float) -> None:
        """Command velocity for `duration` seconds with heading `yaw_deg`."""
        ...

    @abstractmethod
    def move_by_roll_pitch_yawrate_z(
        self, roll_rad: float, pitch_rad: float,
        yaw_rate_rad: float, z: float, duration: float
    ) -> None:
        """Body-rate command (for RL policy output)."""
        ...

    @abstractmethod
    def move_to_position(self, x: float, y: float, z: float,
                         speed: float) -> None:
        """Blocking move to world position at given speed."""
        ...

    @abstractmethod
    def hover(self) -> None:
        """Stop and hover in place."""
        ...

    @abstractmethod
    def get_gate_poses(self) -> list:
        """Return list of (name, position_np) tuples for all gates in scene."""
        ...

    @abstractmethod
    def arm_and_takeoff(self, target_z: float) -> None:
        """Arm, take off, and climb to target_z (negative = up in NED)."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Graceful shutdown: hover, land, disarm, disable API control."""
        ...


class AirSimInterface(DroneInterface):
    """Wraps airsimdroneracinglab.MultirotorClient for simulation."""

    def __init__(self):
        """Initialize connection to AirSim."""
        try:
            import airsimdroneracinglab as airsim
            self._airsim = airsim
        except ImportError:
            raise ImportError("airsimdroneracinglab package not found. "
                            "Install: pip install airsim")

        self._client = airsim.MultirotorClient()
        self._client.race_tier = None
        self._client.level_name = ""
        self._client.confirmConnection()
        self._last_state = None
        self._last_image = None

    def _refresh_state(self):
        """Fetch latest state from AirSim."""
        self._last_state = self._client.getMultirotorState()
        return self._last_state

    def get_image(self) -> np.ndarray:
        """Capture RGB camera frame from AirSim."""
        try:
            responses = self._client.simGetImages(
                [self._airsim.ImageRequest("0", self._airsim.ImageType.Scene,
                                          False, False)]
            )
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8,
                                     dtype=np.uint8)
                frame = img1d.reshape(responses[0].height,
                                     responses[0].width, 3).copy()
                self._last_image = frame
                return frame
            else:
                if self._last_image is not None:
                    return self._last_image
                return np.zeros((480, 640, 3), dtype=np.uint8)
        except Exception as e:
            print(f"Warning: Failed to get image: {e}")
            if self._last_image is not None:
                return self._last_image
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_position(self) -> np.ndarray:
        """Return current position in NED frame."""
        state = self._refresh_state()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float64)

    def get_velocity(self) -> np.ndarray:
        """Return current velocity in NED frame."""
        state = self._refresh_state()
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val], dtype=np.float64)

    def get_orientation(self):
        """Return orientation quaternion."""
        state = self._refresh_state()
        return state.kinematics_estimated.orientation

    def get_yaw(self) -> float:
        """Extract yaw from quaternion (radians)."""
        q = self.get_orientation()
        siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy = 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val)
        return math.atan2(siny, cosy)

    def get_imu_data(self) -> dict:
        """Return IMU sensor data (linear acceleration + angular velocity).

        Returns dict with keys:
            linear_acceleration: np.ndarray [ax, ay, az] m/s²
            angular_velocity:    np.ndarray [gx, gy, gz] rad/s
        """
        try:
            imu = self._client.getImuData(imu_name="Imu", vehicle_name="")
        except Exception:
            try:
                imu = self._client.getImuData()
            except Exception:
                # Fall back to finite-difference acceleration from velocity
                return {
                    "linear_acceleration": np.zeros(3, dtype=np.float64),
                    "angular_velocity": np.zeros(3, dtype=np.float64),
                }
        return {
            "linear_acceleration": np.array([
                imu.linear_acceleration.x_val,
                imu.linear_acceleration.y_val,
                imu.linear_acceleration.z_val,
            ], dtype=np.float64),
            "angular_velocity": np.array([
                imu.angular_velocity.x_val,
                imu.angular_velocity.y_val,
                imu.angular_velocity.z_val,
            ], dtype=np.float64),
        }

    def move_by_velocity(self, vx: float, vy: float, vz: float,
                         duration: float, yaw_deg: float) -> None:
        """Command velocity with yaw heading."""
        self._client.moveByVelocityAsync(
            float(vx), float(vy), float(vz),
            float(duration),
            yaw_mode=self._airsim.YawMode(
                is_rate=False,
                yaw_or_rate=float(yaw_deg)
            )
        )

    def move_by_roll_pitch_yawrate_z(
        self, roll_rad: float, pitch_rad: float,
        yaw_rate_rad: float, z: float, duration: float
    ) -> None:
        """Body-rate command (used by RL policy output)."""
        self._client.moveByRollPitchYawrateZAsync(
            float(roll_rad), float(pitch_rad),
            float(yaw_rate_rad), float(z), float(duration)
        )

    def move_to_position(self, x: float, y: float, z: float,
                         speed: float) -> None:
        """Blocking move to world position at given speed."""
        self._client.moveToPositionAsync(
            float(x), float(y), float(z), float(speed)
        ).join()

    def hover(self) -> None:
        """Hover in place."""
        self._client.hoverAsync().join()

    def get_gate_poses(self) -> list:
        """Fetch all gate poses from scene."""
        # This mirrors get_gate_names() + get_object_pose_safe() from V14
        gate_names = self._get_gate_names()
        result = []
        for name in gate_names:
            pose = self._get_object_pose_safe(name)
            if pose is not None:
                result.append((name, pose))
        return result

    def _get_gate_names(self) -> list:
        """Query scene for all gate objects (mirrored from V14)."""
        objs = self._client.simListSceneObjects(".*[Gg]ate.*")
        gates = [o for o in objs if re.search(r'[Gg]ate', o)]
        gates.sort(key=lambda s: int(re.findall(r'\d+', s)[0])
                   if re.findall(r'\d+', s) else 0)
        return gates

    def _get_object_pose_safe(self, name: str) -> np.ndarray:
        """Safely fetch object pose, return None on error."""
        try:
            pose = self._client.simGetObjectPose(name)
            pos = pose.position
            return np.array([pos.x_val, pos.y_val, pos.z_val],
                          dtype=np.float64)
        except Exception:
            return None

    def arm_and_takeoff(self, target_z: float) -> None:
        """Arm, enable API control, takeoff, and climb to target_z."""
        print("Arming drone...")
        self._client.enableApiControl()
        self._client.arm()
        print("Taking off...")
        self._client.takeoffAsync().join()

        # Climb to takeoff altitude
        state = self._refresh_state()
        pos = state.kinematics_estimated.position
        current_z = pos.z_val
        print(f"Climbing to z={target_z:.1f}m...")
        self._client.moveToPositionAsync(
            float(pos.x_val), float(pos.y_val), float(target_z), 2.0
        ).join()
        print("Takeoff complete.")

    def shutdown(self) -> None:
        """Graceful shutdown sequence."""
        print("Shutting down...")
        try:
            self._client.hoverAsync().join()
            self._client.landAsync().join()
            self._client.disarm()
            self._client.disableApiControl()
        except Exception as e:
            print(f"Warning during shutdown: {e}")


class HardwareInterface(DroneInterface):
    """Stub for real drone hardware.

    Methods raise NotImplementedError with guidance on how to implement
    for real hardware (e.g., via MAVLink, ROS, or other autopilot APIs).
    """

    def get_image(self) -> np.ndarray:
        """
        TODO: Connect to real camera feed.
        Options:
        - USB camera via cv2.VideoCapture()
        - ROS topic subscriber
        - MAVProxy image stream
        """
        raise NotImplementedError(
            "Implement camera capture for your hardware platform"
        )

    def get_position(self) -> np.ndarray:
        """
        TODO: Read odometry from VIO/GPS/SLAM system.
        Options:
        - ROS /odom or /vio/pose topic
        - MAVLink LOCAL_POSITION_NED message
        - Custom SLAM output
        """
        raise NotImplementedError(
            "Implement position sensing for your hardware platform"
        )

    def get_velocity(self) -> np.ndarray:
        """
        TODO: Read velocity from state estimate.
        Options:
        - ROS topic
        - MAVLink message
        - Derived from position history
        """
        raise NotImplementedError(
            "Implement velocity sensing for your hardware platform"
        )

    def get_orientation(self):
        """
        TODO: Read attitude (quaternion or Euler angles).
        Options:
        - ROS /tf tree
        - MAVLink ATTITUDE message
        - IMU-based EKF state
        """
        raise NotImplementedError(
            "Implement orientation sensing for your hardware platform"
        )

    def get_yaw(self) -> float:
        """
        TODO: Extract yaw from attitude.
        """
        raise NotImplementedError(
            "Implement yaw extraction for your hardware platform"
        )

    def move_by_velocity(self, vx: float, vy: float, vz: float,
                         duration: float, yaw_deg: float) -> None:
        """
        TODO: Send velocity setpoint to autopilot.
        Options:
        - MAVLink SET_POSITION_TARGET_LOCAL_NED (velocity mode)
        - ROS cmd_vel topic
        - Custom flight controller command

        Note: This call should NOT block; the autopilot runs the
        controller and you poll for state updates separately.
        """
        raise NotImplementedError(
            "Implement velocity control for your hardware platform"
        )

    def move_by_roll_pitch_yawrate_z(
        self, roll_rad: float, pitch_rad: float,
        yaw_rate_rad: float, z: float, duration: float
    ) -> None:
        raise NotImplementedError(
            "Implement body-rate control for your hardware platform"
        )

    def move_to_position(self, x: float, y: float, z: float,
                         speed: float) -> None:
        """
        TODO: Send waypoint and wait for arrival.
        Options:
        - MAVLink SET_POSITION_TARGET_LOCAL_NED (position mode)
        - ROS action server
        - Custom waypoint handler
        """
        raise NotImplementedError(
            "Implement position control for your hardware platform"
        )

    def hover(self) -> None:
        """
        TODO: Command hover in place.
        Options:
        - MAVLink GUIDED_NOGPS mode (hold attitude/thrust)
        - Send zero velocity
        - Send current position as target
        """
        raise NotImplementedError(
            "Implement hover for your hardware platform"
        )

    def get_gate_poses(self) -> list:
        """
        TODO: Retrieve gate positions.
        On hardware, gates are not part of the simulator scene.
        You must provide gate positions via:
        - Manual calibration file (YAML, JSON)
        - SLAM-based detection and mapping
        - Fiducial markers (AprilTags, etc.)
        - Pre-surveyed coordinates

        Return: list of (name, position_np) tuples
        """
        raise NotImplementedError(
            "Implement gate localization for your hardware platform"
        )

    def arm_and_takeoff(self, target_z: float) -> None:
        """
        TODO: Arm drone and climb to target altitude.
        Options:
        - MAVLink commands: CMD_COMPONENT_ARM_DISARM, CMD_NAV_TAKEOFF
        - ROS service calls
        - Custom autopilot interface
        """
        raise NotImplementedError(
            "Implement arm/takeoff for your hardware platform"
        )

    def shutdown(self) -> None:
        """
        TODO: Graceful shutdown: land and disarm.
        Options:
        - MAVLink commands: CMD_NAV_LAND, CMD_COMPONENT_ARM_DISARM
        - ROS service calls
        """
        raise NotImplementedError(
            "Implement shutdown for your hardware platform"
        )
