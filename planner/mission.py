"""Mission definition and YAML loader.

Mission file format (Nav2-compatible conventions)::

    mission:
      name: "demo_podium"
      frame_id: "world"
      waypoints:
        - id: "start"
          pose: {x: 0.0, y: 0.0, yaw: 0.0}
          behavior: "walk"
          speed: 0.4
          arrival_radius: 0.3
        - id: "podium"
          pose: {x: 5.0, y: 0.0, yaw: 0.0}
          behavior: "climb"
          speed: 0.1
          arrival_radius: 0.3
          on_arrive: "handshake"
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Waypoint:
    """A single navigation waypoint."""
    id: str
    x: float
    y: float
    yaw: float = 0.0
    behavior: str = "walk"
    speed: float = 0.4
    arrival_radius: float = 0.3
    hold_duration: float = 0.0
    on_arrive: Optional[str] = None

    def distance_to(self, x: float, y: float) -> float:
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def bearing_from(self, x: float, y: float) -> float:
        """Bearing angle from (x, y) to this waypoint."""
        return math.atan2(self.y - y, self.x - x)


@dataclass
class Mission:
    """Ordered sequence of waypoints defining a navigation mission."""
    name: str = "unnamed"
    frame_id: str = "world"
    policies: Dict[str, str] = field(default_factory=dict)
    waypoints: List[Waypoint] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Mission:
        """Load a mission from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        m = data.get("mission", data)
        wps = []
        for wp_data in m.get("waypoints", []):
            pose = wp_data.get("pose", {})
            wps.append(Waypoint(
                id=wp_data["id"],
                x=pose.get("x", 0.0),
                y=pose.get("y", 0.0),
                yaw=pose.get("yaw", 0.0),
                behavior=wp_data.get("behavior", "walk"),
                speed=wp_data.get("speed", 0.4),
                arrival_radius=wp_data.get("arrival_radius", 0.3),
                hold_duration=wp_data.get("hold_duration", 0.0),
                on_arrive=wp_data.get("on_arrive"),
            ))
        return cls(
            name=m.get("name", "unnamed"),
            frame_id=m.get("frame_id", "world"),
            policies=m.get("policies", {}),
            waypoints=wps,
        )
