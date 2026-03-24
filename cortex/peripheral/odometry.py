"""
Step Odometry
=============
Provides encoder-style step counting and IMU-based heading for the G1 robot.

Step detection uses foot contact force sensors (8 total, 4 per foot).
A heel-strike event is detected when total foot force rises above a
threshold after being below it — each heel-strike counts as one step.

Heading is read from the IMU framequat sensor on the ``imu`` site,
matching real-hardware behaviour (no direct access to the free-joint).

Cardinal direction convention: +X = East, +Y = North.
  North = 0°, East = 90°, South = 180°, West = 270°.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.spatial.transform

logger = logging.getLogger(__name__)

CARDINAL_HEADINGS: Dict[str, float] = {
    "north": 0.0,
    "n": 0.0,
    "northeast": 45.0,
    "ne": 45.0,
    "east": 90.0,
    "e": 90.0,
    "southeast": 135.0,
    "se": 135.0,
    "south": 180.0,
    "s": 180.0,
    "southwest": 225.0,
    "sw": 225.0,
    "west": 270.0,
    "w": 270.0,
    "northwest": 315.0,
    "nw": 315.0,
}

# +X = East, +Y = North → robot yaw 0 rad points along +X (East = 90° compass).
# Compass heading = 90° - yaw_deg (mod 360).
_YAW_TO_COMPASS_OFFSET = 90.0

DEFAULT_STRIDE_LENGTH = 0.20  # metres per step (Groot WBC at default speed)
FOOT_FORCE_THRESHOLD = 20.0  # Newtons — minimum total foot-force for a heel-strike


class StepOdometry:
    """Encoder-style step counter and IMU heading tracker.

    Must be initialised with a MuJoCo model so it can look up sensor
    addresses once, then ``tick(data)`` is called every sim step.
    """

    def __init__(self, model):
        self._model = model
        self._stride_length = DEFAULT_STRIDE_LENGTH

        self._step_count: int = 0
        self._total_distance: float = 0.0

        # IMU state (updated every tick)
        self._heading_rad: float = 0.0
        self._roll_deg: float = 0.0
        self._pitch_deg: float = 0.0
        self._gyro: np.ndarray = np.zeros(3)       # rad/s  (x, y, z)
        self._accel: np.ndarray = np.zeros(3)       # m/s²   (x, y, z)
        self._body_vel: np.ndarray = np.zeros(3)    # m/s    (x, y, z)
        self._left_foot_force: float = 0.0          # N
        self._right_foot_force: float = 0.0         # N

        # Look up sensor addresses in sensordata once at init
        self._imu_quat_adr = self._sensor_adr("imu_quat", dim=4)
        self._imu_gyro_adr = self._sensor_adr("imu_gyro", dim=3)
        self._imu_acc_adr = self._sensor_adr("imu_acc", dim=3)
        self._frame_vel_adr = self._sensor_adr("frame_vel", dim=3)
        self._left_force_adrs = [
            self._force_sensor_adr(f"left_foot_contact_{i}") for i in range(1, 5)
        ]
        self._right_force_adrs = [
            self._force_sensor_adr(f"right_foot_contact_{i}") for i in range(1, 5)
        ]

        # Heel-strike state machine (per foot).
        # Start True so that initial ground contact doesn't register as a step.
        self._left_was_in_contact = True
        self._right_was_in_contact = True

        self._stride_mean: float = DEFAULT_STRIDE_LENGTH

        logger.info(
            "StepOdometry initialised: imu_quat@%d gyro@%d acc@%d, "
            "%d left + %d right force sensors",
            self._imu_quat_adr, self._imu_gyro_adr, self._imu_acc_adr,
            len(self._left_force_adrs), len(self._right_force_adrs),
        )

    # ── Sensor address helpers ────────────────────────────────────────────

    def _sensor_adr(self, name: str, dim: int = 1) -> int:
        """Return the start index in ``data.sensordata`` for a named sensor."""
        sid = self._model.sensor(name).id
        return int(self._model.sensor_adr[sid])

    def _force_sensor_adr(self, site_name: str) -> int:
        """Find the force sensor attached to *site_name*.

        Force sensors are unnamed in the XML, so we match by site id.
        Returns the sensordata start index (3 values: fx, fy, fz).
        """
        site_id = self._model.site(site_name).id
        for i in range(self._model.nsensor):
            if (self._model.sensor_objtype[i] == 6  # mjOBJ_SITE
                    and self._model.sensor_objid[i] == site_id
                    and self._model.sensor_dim[i] == 3):
                return int(self._model.sensor_adr[i])
        raise ValueError(f"No force sensor found for site '{site_name}'")

    # ── Per-step update ───────────────────────────────────────────────────

    def tick(self, data) -> None:
        """Call once per simulation step to update all sensor readings."""
        self._update_imu(data)
        self._update_steps(data)

    def _update_imu(self, data) -> None:
        """Read all IMU sensors: orientation, gyro, accelerometer, body velocity."""
        sd = data.sensordata

        # Orientation quaternion → roll, pitch, yaw
        a = self._imu_quat_adr
        quat_wxyz = sd[a:a + 4]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        rot = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
        roll, pitch, yaw = rot.as_euler("xyz")
        self._heading_rad = float(yaw)
        self._roll_deg = float(math.degrees(roll))
        self._pitch_deg = float(math.degrees(pitch))

        # Gyroscope (angular velocity, rad/s)
        g = self._imu_gyro_adr
        self._gyro = sd[g:g + 3].copy()

        # Accelerometer (linear acceleration, m/s²)
        ac = self._imu_acc_adr
        self._accel = sd[ac:ac + 3].copy()

        # Body frame velocity (m/s)
        v = self._frame_vel_adr
        self._body_vel = sd[v:v + 3].copy()

    def _update_steps(self, data) -> None:
        """Detect heel-strike events from foot force sensors."""
        left_force = self._total_foot_force(data, self._left_force_adrs)
        right_force = self._total_foot_force(data, self._right_force_adrs)
        self._left_foot_force = left_force
        self._right_foot_force = right_force

        left_in_contact = left_force > FOOT_FORCE_THRESHOLD
        right_in_contact = right_force > FOOT_FORCE_THRESHOLD

        if left_in_contact and not self._left_was_in_contact:
            self._register_step(data)
        if right_in_contact and not self._right_was_in_contact:
            self._register_step(data)

        self._left_was_in_contact = left_in_contact
        self._right_was_in_contact = right_in_contact

    def _total_foot_force(self, data, adrs: list) -> float:
        """Sum the vertical (z) force magnitudes across all contact sites on one foot."""
        total = 0.0
        for adr in adrs:
            fz = abs(data.sensordata[adr + 2])
            total += fz
        return total

    def _register_step(self, data) -> None:
        """Record one step using fixed stride (no qpos dependency).

        On real hardware the stride length would come from a gait
        calibration table indexed by commanded speed, or from leg
        kinematics.  Here we use the constant DEFAULT_STRIDE_LENGTH.
        """
        self._step_count += 1
        self._total_distance += self._stride_mean

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def heading_rad(self) -> float:
        return self._heading_rad

    @property
    def heading_deg(self) -> float:
        """Compass heading in degrees: 0 = North (+Y), 90 = East (+X)."""
        yaw_deg = math.degrees(self._heading_rad)
        return (_YAW_TO_COMPASS_OFFSET - yaw_deg) % 360.0

    @property
    def stride_length(self) -> float:
        return self._stride_mean

    @property
    def total_distance(self) -> float:
        return self._total_distance

    def reset(self) -> None:
        """Zero the step counter and distance accumulator."""
        self._step_count = 0
        self._total_distance = 0.0

    def steps_for_distance(self, meters: float) -> int:
        """Estimate the number of steps needed to cover *meters*."""
        return max(1, int(math.ceil(abs(meters) / self._stride_mean)))

    @staticmethod
    def heading_for_cardinal(direction: str) -> Optional[float]:
        """Convert a cardinal direction string to compass degrees, or None."""
        return CARDINAL_HEADINGS.get(direction.lower().strip())

    def turn_needed(self, target_heading_deg: float) -> float:
        """Signed degrees to turn from current heading to *target_heading_deg*.

        Positive = counter-clockwise (left), negative = clockwise (right).
        Result is in [-180, 180].
        """
        diff = target_heading_deg - self.heading_deg
        diff = (diff + 180) % 360 - 180
        return diff

    @property
    def roll_deg(self) -> float:
        return self._roll_deg

    @property
    def pitch_deg(self) -> float:
        return self._pitch_deg

    @property
    def gyro(self) -> np.ndarray:
        """Angular velocity (rad/s) from IMU gyroscope [x, y, z]."""
        return self._gyro

    @property
    def accel(self) -> np.ndarray:
        """Linear acceleration (m/s²) from IMU accelerometer [x, y, z]."""
        return self._accel

    @property
    def body_velocity(self) -> np.ndarray:
        """Body-frame linear velocity (m/s) [x, y, z]."""
        return self._body_vel

    @property
    def left_foot_force(self) -> float:
        """Total vertical contact force on left foot (N)."""
        return self._left_foot_force

    @property
    def right_foot_force(self) -> float:
        """Total vertical contact force on right foot (N)."""
        return self._right_foot_force

    @property
    def body_speed(self) -> float:
        """Horizontal speed (m/s) from body velocity sensor."""
        return float(np.linalg.norm(self._body_vel[:2]))

    @property
    def heading_cardinal(self) -> str:
        """Convert compass heading to nearest cardinal/intercardinal label."""
        h = self.heading_deg
        labels = [
            (0, "North"), (45, "Northeast"), (90, "East"), (135, "Southeast"),
            (180, "South"), (225, "Southwest"), (270, "West"), (315, "Northwest"),
            (360, "North"),
        ]
        return min(labels, key=lambda pair: abs(pair[0] - h))[1]

    def format_state(self) -> str:
        return (
            f"Steps: {self._step_count}, "
            f"distance: {self._total_distance:.1f}m, "
            f"heading: {self.heading_deg:.0f}° "
            f"(stride: {self._stride_mean:.2f}m)"
        )
