"""
Scene Context
=============
Builds a static environment description and landmark map from the MuJoCo
scene XML at startup.  This is the robot's "venue briefing" — equivalent
to loading a floor plan or SLAM map on real hardware.

No live MuJoCo queries at runtime — only the pre-loaded context and
heading math are used for spatial answers.
"""

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCENE_DIR = Path(__file__).resolve().parents[2] / "scene"

_FRIENDLY_NAMES = {
    "stage": "the stage platform",
    "stairs": "the staircase",
    "water_table": "a small table",
    "water_bottle": "a water bottle (on the table)",
    "podium": "the podium",
    "floor": "the ground floor",
    "marker_1": "floating red number 1",
    "marker_2": "floating blue number 2",
    "marker_3": "floating green number 3",
}

_DIRECTION_LABELS = [
    ((-22.5, 22.5), "ahead"),
    ((22.5, 67.5), "to your front-left"),
    ((67.5, 112.5), "to your left"),
    ((112.5, 157.5), "behind-left"),
    ((157.5, 180.0), "behind you"),
    ((-180.0, -157.5), "behind you"),
    ((-157.5, -112.5), "behind-right"),
    ((-112.5, -67.5), "to your right"),
    ((-67.5, -22.5), "to your front-right"),
]


def _parse_pos(elem) -> Tuple[float, float, float]:
    """Parse the 'pos' attribute of an XML element, defaulting to (0,0,0)."""
    pos_str = elem.get("pos", "0 0 0")
    parts = pos_str.split()
    if len(parts) >= 3:
        return float(parts[0]), float(parts[1]), float(parts[2])
    return 0.0, 0.0, 0.0


def _walk_bodies(
    elem,
    parent_pos: Tuple[float, float, float],
    landmarks: Dict[str, Tuple[float, float, float]],
) -> None:
    """Recursively walk <body> elements, accumulating world positions."""
    for body in elem.findall("body"):
        local = _parse_pos(body)
        world = (
            parent_pos[0] + local[0],
            parent_pos[1] + local[1],
            parent_pos[2] + local[2],
        )
        name = body.get("name")
        if name:
            landmarks[name] = world
        _walk_bodies(body, world, landmarks)


def _parse_body_positions(xml_path: Path) -> Dict[str, Tuple[float, float, float]]:
    """Extract named body positions from a MuJoCo XML file (world coordinates)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    landmarks: Dict[str, Tuple[float, float, float]] = {}
    for wb in root.iter("worldbody"):
        _walk_bodies(wb, (0.0, 0.0, 0.0), landmarks)
    return landmarks


def get_landmark_map(scene_name: str) -> Dict[str, Tuple[float, float, float]]:
    """Return landmark name → (x, y, z) positions parsed from the scene XML.

    On real hardware this would come from a pre-loaded map or SLAM.
    """
    xml_path = _SCENE_DIR / f"{scene_name}.xml"
    if not xml_path.exists():
        logger.warning("Scene XML not found: %s", xml_path)
        return {}
    landmarks = _parse_body_positions(xml_path)
    logger.info("Loaded %d landmarks from %s", len(landmarks), scene_name)
    return landmarks


def build_environment_context(scene_name: str) -> str:
    """Build a natural language venue description from the scene XML.

    Called once at startup.  The result is a static string used as
    context for the planner — equivalent to a mission briefing.
    """
    landmarks = get_landmark_map(scene_name)
    if not landmarks:
        return "You are in an unknown environment."

    parts = [f"You are a humanoid robot (Unitree G1) in a '{scene_name}' environment."]

    for name, (x, y, z) in sorted(landmarks.items()):
        friendly = _FRIENDLY_NAMES.get(name, name.replace("_", " "))
        parts.append(f"- {friendly} is at position ({x:.1f}, {y:.1f}, z={z:.1f})")

    parts.append(
        "\nCoordinates: +X is forward (east), +Y is left (north). "
        "The robot spawns at approximately (-5, 0) facing east (+X direction)."
    )
    return "\n".join(parts)


def _angle_diff(a: float, b: float) -> float:
    """Signed angle difference in degrees, wrapped to [-180, 180]."""
    d = (a - b) % 360
    if d > 180:
        d -= 360
    return d


def _bearing_label(bearing_deg: float) -> str:
    """Convert a relative bearing (degrees) to a human-readable direction."""
    for (lo, hi), label in _DIRECTION_LABELS:
        if lo <= bearing_deg < hi:
            return label
    return "nearby"


def compose_spatial_response(
    landmark_map: Dict[str, Tuple[float, float, float]],
    query: str,
    robot_pos: Tuple[float, float],
    robot_yaw_rad: float,
) -> Optional[str]:
    """Answer 'where is X?' using the pre-loaded landmark map and heading math.

    Returns None if the query doesn't match any known landmark.
    """
    query_lower = query.lower()
    robot_yaw_deg = math.degrees(robot_yaw_rad)

    for name, (lx, ly, _lz) in landmark_map.items():
        if name.replace("_", " ") in query_lower or name in query_lower:
            dx = lx - robot_pos[0]
            dy = ly - robot_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            world_bearing = math.degrees(math.atan2(dy, dx))
            relative_bearing = _angle_diff(world_bearing, robot_yaw_deg)

            friendly = _FRIENDLY_NAMES.get(name, name.replace("_", " "))
            direction = _bearing_label(relative_bearing)

            return (
                f"{friendly.capitalize()} is approximately {dist:.1f}m away, "
                f"{direction} (bearing {relative_bearing:+.0f}° from your heading)."
            )

    return None


def describe_surroundings(
    landmark_map: Dict[str, Tuple[float, float, float]],
    robot_pos: Tuple[float, float],
    robot_yaw_rad: float,
    max_range: float = 50.0,
) -> str:
    """Describe all landmarks relative to the robot's current position and heading."""
    robot_yaw_deg = math.degrees(robot_yaw_rad)
    items: List[str] = []

    for name, (lx, ly, _lz) in sorted(landmark_map.items()):
        dx = lx - robot_pos[0]
        dy = ly - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > max_range:
            continue
        world_bearing = math.degrees(math.atan2(dy, dx))
        relative_bearing = _angle_diff(world_bearing, robot_yaw_deg)
        friendly = _FRIENDLY_NAMES.get(name, name.replace("_", " "))
        direction = _bearing_label(relative_bearing)
        items.append(f"- {friendly}: {dist:.1f}m {direction}")

    if not items:
        return "I don't detect any landmarks nearby."
    return "From my current position:\n" + "\n".join(items)
