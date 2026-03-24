"""
Intent Matcher — Embedding-Based Command Classification
========================================================
Uses a lightweight sentence embedding model (all-MiniLM-L6-v2, ~80MB)
to semantically match user commands to robot actions.

Flow:
  1. User text is encoded into a 384-dim vector (~5ms on CPU)
  2. Cosine similarity against pre-computed intent templates
  3. Best-matching intent is selected (with confidence threshold)
  4. Lightweight regex extracts parameters (distance, degrees, landmark)

This replaces both brittle regex matching and slow VLM inference
for command interpretation.  VLM is reserved for visual tasks only.
"""

import logging
import math
import re
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.40


class ActionType(Enum):
    WALK = auto()
    TURN = auto()
    LOOK = auto()
    LOOK_AROUND = auto()
    NAVIGATE_TO = auto()
    STOP = auto()
    DESCRIBE_SCENE = auto()
    GET_POSITION = auto()
    GET_STATUS = auto()
    RELEASE_HARNESS = auto()
    START_MISSION = auto()
    HELP = auto()
    QUIT = auto()
    UNKNOWN = auto()


@dataclass
class ParsedAction:
    action_type: ActionType
    confidence: float
    meters: Optional[float] = None
    degrees: Optional[float] = None
    speed: Optional[float] = None
    direction: Optional[str] = None
    landmark: Optional[str] = None
    raw_text: str = ""


# Intent templates: each action type maps to example phrases
# that the embedding model uses for similarity matching.
INTENT_TEMPLATES: Dict[ActionType, List[str]] = {
    ActionType.WALK: [
        "walk forward",
        "move ahead",
        "go straight",
        "walk 5 meters",
        "move forward 2m",
        "go ahead 3 meters",
        "walk straight",
        "run forward",
        "go forward",
        "walk 1 meter",
        "move 10 meters",
        "walk slowly",
        "walk fast",
        "go backwards",
        "walk back 2 meters",
        "go 5 meters north",
        "walk 10m east",
        "move south 3 meters",
    ],
    ActionType.TURN: [
        "turn left",
        "turn right",
        "rotate",
        "turn around",
        "turn 90 degrees",
        "spin right 45",
        "turn 180",
        "rotate left 30 degrees",
        "face the other direction",
        "turn 40 degrees right",
        "rotate 90 degrees left",
    ],
    ActionType.LOOK: [
        "what do you see",
        "look ahead",
        "describe what you see",
        "what is in front of you",
        "can you see anything",
        "tell me what you see",
        "what's visible",
    ],
    ActionType.LOOK_AROUND: [
        "look around",
        "scan the area",
        "scan 360",
        "look in all directions",
        "survey the area",
        "check surroundings",
    ],
    ActionType.NAVIGATE_TO: [
        "go to the podium",
        "walk to the table",
        "navigate to the stage",
        "move to the bottle",
        "walk towards the podium",
        "head to the podium",
        "approach the table",
        "go to the backstage area",
        "walk over to the water bottle",
    ],
    ActionType.STOP: [
        "stop walking",
        "halt now",
        "stop moving",
        "don't move",
        "stand still",
        "stay where you are",
        "cease movement",
    ],
    ActionType.DESCRIBE_SCENE: [
        "describe the scene",
        "what is around me",
        "environment layout",
        "tell me about surroundings",
        "describe the environment",
        "what's in this room",
        "scene description",
        "where are things located",
    ],
    ActionType.GET_POSITION: [
        "where am i",
        "my position",
        "current location",
        "what is my heading",
        "what direction am i facing",
        "coordinates",
    ],
    ActionType.GET_STATUS: [
        "robot status",
        "system status",
        "what is your state",
        "are you okay",
        "diagnostics",
        "how are you doing",
    ],
    ActionType.RELEASE_HARNESS: [
        "release harness",
        "drop harness",
        "free me",
        "release the robot",
        "take off harness",
        "remove harness",
    ],
    ActionType.START_MISSION: [
        "start mission",
        "begin mission",
        "start the task",
        "begin the presentation",
        "execute mission",
    ],
    ActionType.HELP: [
        "help",
        "what can you do",
        "list commands",
        "available commands",
        "show me options",
        "what are your capabilities",
    ],
    ActionType.QUIT: [
        "quit simulation",
        "exit simulation",
        "end simulation",
        "shutdown",
        "close everything",
    ],
}

# Parameter extraction patterns
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_METERS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:m(?:et(?:er|re)s?)?)?",
    re.IGNORECASE,
)
_DEGREES_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:deg(?:ree)?s?)?",
    re.IGNORECASE,
)
_DIRECTION_LEFT_RE = re.compile(r"\b(left)\b", re.IGNORECASE)
_DIRECTION_RIGHT_RE = re.compile(r"\b(right)\b", re.IGNORECASE)
_CARDINAL_RE = re.compile(
    r"\b(northeast|northwest|southeast|southwest|north|south|east|west|ne|nw|se|sw)\b",
    re.IGNORECASE,
)
_SPEED_WORDS = {
    "slowly": 0.15, "slow": 0.15,
    "fast": 0.5, "quickly": 0.5, "run": 0.5,
    "normal": 0.3,
}
_BACKWARD_RE = re.compile(r"\b(back(?:ward)?s?|reverse|retreat)\b", re.IGNORECASE)
_LANDMARK_PREPS = re.compile(
    r"(?:to|towards|toward|at|near)\s+(?:the\s+)?(.+?)(?:\s+and\b|\s*$)",
    re.IGNORECASE,
)


_STEP_SPLIT_RE = re.compile(
    r"\s*(?:,\s*(?:then\s+)?|;\s*|\s+then\s+|\s+and\s+then\s+|\s+after\s+that\s+)\s*",
    re.IGNORECASE,
)

# "walk 1m and what do you see" → split before the perception clause
_PERCEPTION_BOUNDARY_RE = re.compile(
    r"\s+(?:and\s+)?(?=what\s+do\s+you\s+see|what'?s\s+around|what\s+can\s+you\s+see|"
    r"describe\s+what\s+you\s+see|tell\s+me\s+what\s+you\s+see|look\s+around\b)",
    re.IGNORECASE,
)

# "what do you see and go 3m" → split before second motion clause
_MOTION_CHAIN_AND_RE = re.compile(
    r"\s+and\s+(?=(?:again\s+)?(?:go|walk|turn|move|run)\b)",
    re.IGNORECASE,
)

# "go 1 m turn left" (no comma) → split before "turn ..."
_TURN_AFTER_DISTANCE_RE = re.compile(
    r"\s+(?=turn\s+(?:left|right|around|\d))",
    re.IGNORECASE,
)

# "turn left go 3 m" / "turn right then go" already handled by _STEP_SPLIT_RE;
# this catches "turn left again go 3 m" without "then"
_MOTION_AFTER_TURN_RE = re.compile(
    r"\s+(?=(?:again\s+)?(?:go|walk|move|run)\b)",
    re.IGNORECASE,
)


def _flat_split(pieces: List[str], rx: re.Pattern) -> List[str]:
    out: List[str] = []
    for p in pieces:
        for chunk in rx.split(p):
            c = chunk.strip()
            if c:
                out.append(c)
    return out


def _subdivide_motion_within_clause(piece: str) -> List[str]:
    """Split 'go 1m turn left' and 'turn left go 3m' style glue inside one phrase."""
    piece = piece.strip()
    if not piece:
        return []
    parts = _TURN_AFTER_DISTANCE_RE.split(piece)
    result: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        for chunk in _MOTION_AFTER_TURN_RE.split(part):
            c = chunk.strip()
            if c:
                result.append(c)
    return result


def expand_pathway_segments(text: str) -> List[str]:
    """Flatten natural language into ordered atomic clauses for execution.

    Handles:
      - Commas / then / and then
      - Perception: "... and what do you see"
      - Motion chains: "... turn left" without comma after distance
      - "turn left ... go 3 m" without explicit "then"
    """
    text = text.strip()
    if not text:
        return []

    parts = [text]
    parts = _flat_split(parts, _STEP_SPLIT_RE)
    parts = _flat_split(parts, _PERCEPTION_BOUNDARY_RE)
    parts = _flat_split(parts, _MOTION_CHAIN_AND_RE)

    out: List[str] = []
    for p in parts:
        out.extend(_subdivide_motion_within_clause(p))
    return [x.strip() for x in out if x.strip()]


class IntentMatcher:
    """Semantic command classifier using sentence embeddings.

    Loads all-MiniLM-L6-v2 once, pre-computes embeddings for all
    intent templates, then matches user text in ~5ms.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = None
        self._model_name = model_name
        self._template_embeddings: Optional[np.ndarray] = None
        self._template_labels: List[ActionType] = []
        self._template_texts: List[str] = []
        self._loaded = False
        self._load_time_ms: float = 0

    def load(self) -> bool:
        """Load the embedding model and pre-compute template vectors."""
        if self._loaded:
            return True

        try:
            t0 = time.perf_counter()
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

            for action_type, texts in INTENT_TEMPLATES.items():
                for text in texts:
                    self._template_texts.append(text)
                    self._template_labels.append(action_type)

            self._template_embeddings = self._model.encode(
                self._template_texts, normalize_embeddings=True,
            )

            self._load_time_ms = (time.perf_counter() - t0) * 1000
            self._loaded = True
            logger.info(
                "IntentMatcher loaded: %s (%d templates, %.0fms)",
                self._model_name, len(self._template_texts), self._load_time_ms,
            )
            return True

        except Exception as e:
            logger.error("IntentMatcher failed to load: %s", e)
            return False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def match(self, text: str) -> ParsedAction:
        """Classify user text and extract parameters.

        Returns a ParsedAction with the matched intent and extracted params.
        """
        if not self._loaded:
            return ParsedAction(
                action_type=ActionType.UNKNOWN, confidence=0.0, raw_text=text,
            )

        t0 = time.perf_counter()

        query_emb = self._model.encode([text], normalize_embeddings=True)[0]
        similarities = self._template_embeddings @ query_emb

        # Per-intent max similarity
        intent_scores: Dict[ActionType, float] = {}
        for sim, label in zip(similarities, self._template_labels):
            if label not in intent_scores or sim > intent_scores[label]:
                intent_scores[label] = float(sim)

        best_intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[best_intent]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[INTENT] text='%s' -> %s (%.3f) in %.0fms",
            text[:60], best_intent.name, best_score, elapsed_ms,
        )

        if best_score < CONFIDENCE_THRESHOLD:
            return ParsedAction(
                action_type=ActionType.UNKNOWN, confidence=best_score, raw_text=text,
            )

        action = ParsedAction(
            action_type=best_intent, confidence=best_score, raw_text=text,
        )
        self._extract_params(action, text)
        return action

    def match_pathway(self, text: str) -> List[ParsedAction]:
        """Split compound instructions and classify each step.

        "go 10m north, then turn east and go 5m" →
            [WALK(10m, north), TURN_MAGNETIC("east"), WALK(5m)]
        """
        segments = expand_pathway_segments(text)

        if len(segments) <= 1:
            return [self.match(text)]

        logger.info("[INTENT:PATHWAY] %d segments: %s", len(segments), segments)
        return [self.match(seg) for seg in segments]

    def _extract_params(self, action: ParsedAction, text: str) -> None:
        """Extract numeric parameters and modifiers from the text."""
        text_lower = text.lower()

        if action.action_type == ActionType.WALK:
            # Distance
            m = _METERS_RE.search(text)
            if m:
                action.meters = float(m.group(1))
            else:
                action.meters = 1.0

            # Backward
            if _BACKWARD_RE.search(text_lower):
                action.meters = -(action.meters or 1.0)

            # Speed
            for word, spd in _SPEED_WORDS.items():
                if word in text_lower:
                    action.speed = spd
                    break
            if action.speed is None:
                action.speed = 0.3

            # Cardinal direction
            cm = _CARDINAL_RE.search(text_lower)
            if cm:
                action.direction = cm.group(1).lower()

        elif action.action_type == ActionType.TURN:
            # Degrees
            m = _DEGREES_RE.search(text)
            if m:
                action.degrees = float(m.group(1))
            else:
                action.degrees = 90.0

            # "turn around" special case
            if "around" in text_lower or "180" in text:
                action.degrees = 180.0

            # Left/right -> sign (positive=left, negative=right)
            if _DIRECTION_RIGHT_RE.search(text_lower):
                action.degrees = -abs(action.degrees)
            elif _DIRECTION_LEFT_RE.search(text_lower):
                action.degrees = abs(action.degrees)

        elif action.action_type == ActionType.NAVIGATE_TO:
            m = _LANDMARK_PREPS.search(text_lower)
            if m:
                action.landmark = m.group(1).strip().rstrip(".")
            else:
                # Fallback: everything after "to"
                parts = text_lower.split(" to ", 1)
                if len(parts) > 1:
                    action.landmark = parts[1].strip().rstrip(".")
