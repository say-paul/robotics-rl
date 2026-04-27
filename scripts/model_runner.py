#!/usr/bin/env python3
"""
Generic ONNX Model Runner — run any robot pipeline from a YAML config.

Reads the execution graph from a robot YAML, loads all ONNX models, places
them on the declared device (GPU/CPU), and chains them via a shared signal
bus driven by the dataflow section.  Controllers are first-class DAG nodes
that can be wired to any downstream node.

Usage:
    # Autonomous mode (VLA drives the pipeline)
    python scripts/model_runner.py --robot configs/robots/g1_sonic_wbc.yaml

    # With keyboard controller wired into the DAG
    python scripts/model_runner.py --robot configs/robots/g1_sonic_wbc.yaml \\
        --controller keyboard

    # Dry run — print the resolved graph without running
    python scripts/model_runner.py --robot configs/robots/g1_sonic_wbc.yaml \\
        --dry-run
"""

import argparse
import importlib
import json
import re
import signal as signal_mod
import sys
import threading
import time
from abc import ABC, abstractmethod
from math import gcd
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# YAML loading (shared with launch_robot.py)
# ---------------------------------------------------------------------------

def substitute_variables(obj, variables):
    """Recursively replace ${VAR_NAME} placeholders in a config tree."""
    if isinstance(obj, dict):
        return {k: substitute_variables(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_variables(item, variables) for item in obj]
    if isinstance(obj, str):
        result = obj
        for name, value in variables.items():
            result = result.replace(f"${{{name}}}", str(value))
        return result
    return obj


def load_robot_config(yaml_path):
    """Load a robot YAML and resolve all ${VAR} references."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"Error: robot config not found: {path}")
        sys.exit(1)
    with open(path) as f:
        raw = yaml.safe_load(f)
    variables = raw.get("variables", {})
    return substitute_variables(raw, variables)


# ---------------------------------------------------------------------------
# Signal Bus
# ---------------------------------------------------------------------------

class SignalBus:
    """Thread-safe shared memory for inter-node communication.

    Every signal is a named numpy array with a timestamp recording when it
    was last written.  Nodes read inputs from the bus and write outputs back.
    """

    def __init__(self):
        self._data: dict[str, np.ndarray] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def put(self, name: str, value: np.ndarray) -> None:
        with self._lock:
            self._data[name] = value
            self._timestamps[name] = time.monotonic()

    def get(self, name: str) -> np.ndarray | None:
        with self._lock:
            return self._data.get(name)

    def get_with_age(self, name: str) -> tuple[np.ndarray | None, float]:
        """Return (value, age_seconds).  Age is inf if signal was never written."""
        with self._lock:
            val = self._data.get(name)
            ts = self._timestamps.get(name)
        if val is None or ts is None:
            return None, float("inf")
        return val, time.monotonic() - ts

    def has(self, name: str) -> bool:
        with self._lock:
            return name in self._data

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return a shallow copy of all signals (for dry-run / debug)."""
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Dataflow Parser
# ---------------------------------------------------------------------------

_ARROW_RE = re.compile(r"\s*[→\->]+\s*")


def _split_signals(text: str) -> list[str]:
    """Split a comma-separated signal list, stripping whitespace."""
    return [s.strip() for s in text.split(",") if s.strip()]


class DataflowEdge:
    """One line of the dataflow section, parsed into structured form."""

    __slots__ = ("inputs", "node_name", "outputs", "source_tag")

    def __init__(self, inputs: list[str], node_name: str,
                 outputs: list[str], source_tag: str = "autonomous"):
        self.inputs = inputs
        self.node_name = node_name
        self.outputs = outputs
        self.source_tag = source_tag

    def __repr__(self):
        return (f"DataflowEdge({', '.join(self.inputs)} -> "
                f"{self.node_name} -> {', '.join(self.outputs)} "
                f"[{self.source_tag}])")


def parse_dataflow(raw_lines: list[str]) -> list[DataflowEdge]:
    """Parse YAML dataflow strings into structured edges.

    Supported format:
        input1, input2 → node_name → output1, output2
    Chained arrows are split into multiple edges.
    """
    edges: list[DataflowEdge] = []
    for line in raw_lines:
        parts = _ARROW_RE.split(line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 2:
            continue

        # Walk the chain: each triple (inputs, node, outputs) becomes an edge.
        # For chains like A → B → C → D, we treat B and C as nodes with
        # the preceding segment as inputs and following segment as outputs.
        # Convention: even indices are signal lists, odd indices are node names.
        # If there are only 2 parts, it's "inputs → node" with no declared outputs.
        i = 0
        while i + 1 < len(parts):
            inputs = _split_signals(parts[i])
            node_name = parts[i + 1].strip()
            if i + 2 < len(parts):
                outputs = _split_signals(parts[i + 2])
            else:
                outputs = []
            edges.append(DataflowEdge(inputs, node_name, outputs))
            i += 2
    return edges


def tag_controller_edges(edges: list[DataflowEdge],
                         controller_names: set[str]) -> None:
    """Mark edges whose inputs come from a controller node."""
    for edge in edges:
        for inp in edge.inputs:
            prefix = inp.split(".")[0] if "." in inp else ""
            if prefix in controller_names:
                edge.source_tag = "controller"
                break


# ---------------------------------------------------------------------------
# Handler Plugin System
# ---------------------------------------------------------------------------

def load_handler(dotted_path: str):
    """Import a handler callable from a dotted module path.

    Example: "handlers.sonic_encoder.build_obs" loads the build_obs function
    from scripts/handlers/sonic_encoder.py.
    """
    module_path, _, func_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Handler path must be module.function, got: {dotted_path}")
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise ImportError(f"Handler function '{func_name}' not found in {module_path}")
    return fn


# ---------------------------------------------------------------------------
# PolicyRunner base class (used by mujoco_engine.py)
# ---------------------------------------------------------------------------

class PolicyRunner(ABC):
    """Interface between the MuJoCo engine and a specific policy architecture.

    The engine calls setup() once, then step() every decimation sim-steps.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for status prints."""

    @abstractmethod
    def setup(self, model, data):
        """Called once after MuJoCo model/data are created.
        Must return n_actuated (int)."""

    @abstractmethod
    def step(self, model, data):
        """Called every decimation sim-steps when policy is active.
        Returns (target_pos, kps, kds) as np.arrays of length n_actuated."""

    def set_controller(self, controller):
        """Attach a runtime controller."""
        pass

    def info_lines(self):
        """Extra lines for the startup banner."""
        return []


# ---------------------------------------------------------------------------
# Controller Sources (DAG node type: "controller")
# ---------------------------------------------------------------------------

class ControllerSource(ABC):
    """Base class for human-input DAG nodes.

    A ControllerSource produces signals from a human input device (keyboard,
    gamepad, gRPC, MQTT, etc.) and writes them to the signal bus.  It is
    declared as a node with type "controller" in the execution graph and
    wired to any downstream node via dataflow.
    """

    def __init__(self):
        self._enabled = False
        self._lock = threading.Lock()

    @abstractmethod
    def setup(self, controller_cfg: dict) -> None:
        """Initialize from the YAML controller section."""

    @abstractmethod
    def poll(self) -> dict[str, np.ndarray]:
        """Return current output signals.  Called every scheduler tick."""

    @abstractmethod
    def handle_event(self, event) -> bool:
        """Handle a raw input event.  Return True if consumed."""

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Signal names this source produces."""

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, value: bool) -> None:
        with self._lock:
            self._enabled = value

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled

    def banner_lines(self) -> list[str]:
        return []


class KeyboardControllerSource(ControllerSource):
    """Keyboard controller that reads bindings from the YAML controller section.

    Ported from policy_runners.KeyboardController to work as a generic DAG
    node.  Key events are delivered via handle_event(keycode).
    """

    # GLFW key codes for arrow keys
    _GLFW_KEY_RIGHT = 262
    _GLFW_KEY_LEFT = 263
    _GLFW_KEY_DOWN = 264
    _GLFW_KEY_UP = 265

    _KEY_NAME_TO_GLFW = {
        "UP": 265, "DOWN": 264, "LEFT": 263, "RIGHT": 262,
    }

    def __init__(self):
        super().__init__()
        self._mode_map: dict[str, int] = {}
        self._command: dict[str, object] = {}
        self._movement_bindings: dict[int, dict] = {}
        self._style_bindings: dict[int, int] = {}
        self._active_style = 0
        self._cmd_lock = threading.Lock()

    @property
    def output_names(self) -> list[str]:
        return ["mode", "movement_direction", "facing_direction",
                "target_vel", "height"]

    def setup(self, controller_cfg: dict) -> None:
        modes = controller_cfg.get("locomotion_modes", {})
        self._mode_map = {str(k): int(v) for k, v in modes.items()}

        default = controller_cfg.get("default_command", {})
        self._command = {
            "mode": self._resolve_mode(default.get("locomotion_mode", "IDLE")),
            "movement_direction": list(default.get("movement_direction", [0, 0, 0])),
            "facing_direction": list(default.get("facing_direction", [1, 0, 0])),
            "target_vel": float(default.get("movement_speed", -1)),
            "height": float(default.get("height", -1)),
        }

        kb = controller_cfg.get("keyboard", {})
        for key_name, binding in kb.get("movement", {}).items():
            glfw_code = self._KEY_NAME_TO_GLFW.get(key_name.upper())
            if glfw_code is not None:
                self._movement_bindings[glfw_code] = self._parse_binding(binding)

        for key_char, mode_name in kb.get("styles", {}).items():
            code = ord(str(key_char))
            self._style_bindings[code] = self._resolve_mode(mode_name)

        self._active_style = self._command["mode"]

    def poll(self) -> dict[str, np.ndarray]:
        with self._cmd_lock:
            return {
                "mode": np.array([self._command["mode"]], dtype=np.int64),
                "movement_direction": np.array(
                    self._command["movement_direction"], dtype=np.float32),
                "facing_direction": np.array(
                    self._command["facing_direction"], dtype=np.float32),
                "target_vel": np.array([self._command["target_vel"]], dtype=np.float32),
                "height": np.array([self._command["height"]], dtype=np.float32),
            }

    @property
    def all_keycodes(self) -> set[int]:
        """All keycodes this controller handles (mujoco_engine compat)."""
        codes = set(self._movement_bindings.keys())
        codes |= set(self._style_bindings.keys())
        return codes

    def handle_key(self, keycode) -> bool:
        """Called by mujoco_engine on key press. Returns True if consumed."""
        return self.handle_event(keycode)

    def handle_event(self, event) -> bool:
        keycode = event
        with self._cmd_lock:
            if keycode in self._movement_bindings:
                b = self._movement_bindings[keycode]
                if "movement_direction" in b:
                    self._command["movement_direction"] = list(b["movement_direction"])
                self._command["mode"] = b.get("mode", self._active_style)
                self._print_state()
                return True
            if keycode in self._style_bindings:
                self._active_style = self._style_bindings[keycode]
                self._command["mode"] = self._active_style
                if self._active_style == 0:
                    self._command["movement_direction"] = [0.0, 0.0, 0.0]
                self._print_state()
                return True
        return False

    def banner_lines(self) -> list[str]:
        lines = ["Keyboard controller ready (activate with 'C' toggle):"]
        lines.append("  Arrow keys : move (forward/back/left/right)")
        inv = {v: k for k, v in self._mode_map.items()}
        for char_code, mode_id in sorted(self._style_bindings.items()):
            name = inv.get(mode_id, str(mode_id))
            suffix = " (stop)" if name == "IDLE" else ""
            lines.append(f"  {chr(char_code)}  - {name}{suffix}")
        return lines

    def _resolve_mode(self, name_or_int):
        if isinstance(name_or_int, int):
            return name_or_int
        return self._mode_map.get(str(name_or_int), 0)

    def _parse_binding(self, binding):
        result = {}
        if "movement_direction" in binding:
            result["movement_direction"] = [float(x) for x in binding["movement_direction"]]
        if "locomotion_mode" in binding:
            result["mode"] = self._resolve_mode(binding["locomotion_mode"])
        return result

    def _print_state(self):
        inv = {v: k for k, v in self._mode_map.items()}
        mode_name = inv.get(self._command["mode"], str(self._command["mode"]))
        d = self._command["movement_direction"]
        print(f"[Controller] mode={mode_name}  dir=[{d[0]:.1f},{d[1]:.1f},{d[2]:.1f}]")


CONTROLLER_REGISTRY: dict[str, type[ControllerSource]] = {
    "keyboard": KeyboardControllerSource,
}


# ---------------------------------------------------------------------------
# Execution Nodes
# ---------------------------------------------------------------------------

class ExecutionNode(ABC):
    """Base for anything that can appear in the execution graph."""

    def __init__(self, name: str, frequency_hz: float):
        self.name = name
        self.frequency_hz = frequency_hz

    @abstractmethod
    def tick(self, bus: SignalBus, input_names: list[str],
             output_names: list[str]) -> None:
        """Run one step: read inputs from bus, process, write outputs."""


class ModelNode(ExecutionNode):
    """An ONNX model node.  Loads the model on the declared device and runs
    inference each tick, reading inputs from the bus and writing outputs back."""

    def __init__(self, name: str, model_cfg: dict, node_cfg: dict):
        super().__init__(name, node_cfg.get("frequency_hz", 1))
        self.device = node_cfg.get("device", "cpu")
        self.hold_last_input = node_cfg.get("hold_last_input", True)
        self.max_staleness_ms = node_cfg.get("max_staleness_ms", None)

        model_path = model_cfg.get("model_path")
        handler_path = node_cfg.get("handler")

        self._handler = load_handler(handler_path) if handler_path else None
        self._session = None
        self._model_path = model_path
        self._model_cfg = model_cfg
        self._node_cfg = node_cfg
        self._last_outputs: dict[str, np.ndarray] = {}
        self._error_count: dict[str, int] = {}

    def load(self) -> None:
        """Load the ONNX session.  Called during graph setup (not __init__)
        so that dry-run can skip loading."""
        if self._model_path is None:
            return
        path = Path(self._model_path)
        if not path.exists():
            print(f"  Warning: ONNX file not found: {path}")
            print(f"           Node '{self.name}' will produce zeros until the file is available.")
            return

        try:
            import onnxruntime as ort
        except ImportError:
            print("Error: onnxruntime is required. Install with: pip install onnxruntime")
            sys.exit(1)

        provider = self._pick_provider()
        print(f"  Loading {self.name}: {path.name} on {self.device} ({provider})")
        self._session = ort.InferenceSession(str(path), providers=[provider])

    def _pick_provider(self) -> str:
        import onnxruntime as ort
        if self.device.startswith("gpu"):
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "CUDAExecutionProvider"
            print(f"  Warning: CUDAExecutionProvider not available, "
                  f"falling back to CPU for node '{self.name}'")
        return "CPUExecutionProvider"

    def tick(self, bus: SignalBus, input_names: list[str],
             output_names: list[str]) -> None:
        if self._session is None:
            for out_name in output_names:
                if not bus.has(out_name):
                    bus.put(out_name, np.array([0.0], dtype=np.float32))
            return

        if self._handler is not None:
            feed = self._handler(bus, self._model_cfg, self._node_cfg)
        else:
            feed = self._build_default_feed(bus, input_names)

        if feed is None:
            return

        feed = self._remap_feed_keys(feed)
        try:
            results = self._session.run(None, feed)
        except Exception as e:
            key = type(e).__name__
            self._error_count[key] = self._error_count.get(key, 0) + 1
            if self._error_count[key] <= 1:
                print(f"  Warning: node '{self.name}' raised "
                      f"{key}: {e}")
            elif self._error_count[key] == 2:
                print(f"  Warning: node '{self.name}' — suppressing "
                      f"further {key} warnings")
            return

        session_outputs = self._session.get_outputs()
        for i, out_meta in enumerate(session_outputs):
            signal_name = output_names[i] if i < len(output_names) else out_meta.name
            value = results[i]
            if isinstance(value, np.ndarray):
                bus.put(signal_name, value)
                self._last_outputs[signal_name] = value
            else:
                bus.put(signal_name, np.array(value))

    def _remap_feed_keys(self, feed: dict) -> dict:
        """Remap handler output keys to actual ONNX input tensor names.

        Handlers shouldn't need to know internal ONNX tensor names.  If the
        handler returns a single-key dict and the model has a single input,
        remap automatically.  For multi-input models, match by position.
        """
        session_inputs = self._session.get_inputs()
        onnx_names = [inp.name for inp in session_inputs]
        feed_keys = list(feed.keys())

        if set(feed_keys) == set(onnx_names):
            return feed

        if len(feed_keys) == len(onnx_names):
            return {onnx_names[i]: feed[feed_keys[i]]
                    for i in range(len(feed_keys))}

        return feed

    def _build_default_feed(self, bus: SignalBus,
                            input_names: list[str]) -> dict | None:
        """Build ONNX input dict by matching bus signals to session input names.

        For single-input models, falls back to trying dataflow input names
        if the ONNX input name isn't on the bus.  For multi-input models,
        only exact name matches are used (no blind fallback).
        """
        feed = {}
        session_inputs = self._session.get_inputs()
        single_input = len(session_inputs) == 1
        for inp_meta in session_inputs:
            val = bus.get(inp_meta.name)
            if val is None and single_input:
                for bus_name in input_names:
                    val = bus.get(bus_name)
                    if val is not None:
                        break
            if val is None:
                if self.hold_last_input and inp_meta.name in self._last_outputs:
                    val = self._last_outputs[inp_meta.name]
                else:
                    return None
            feed[inp_meta.name] = val
        return feed


class ControllerNode(ExecutionNode):
    """A controller DAG node.  Wraps a ControllerSource and writes its
    output signals to the bus when enabled."""

    def __init__(self, name: str, source: ControllerSource,
                 node_cfg: dict):
        super().__init__(name, node_cfg.get("frequency_hz", 50))
        self.source = source
        self._output_names_cfg = node_cfg.get("outputs", [])

    def tick(self, bus: SignalBus, input_names: list[str],
             output_names: list[str]) -> None:
        if not self.source.enabled:
            return
        signals = self.source.poll()
        for sig_name, value in signals.items():
            qualified = f"{self.name}.{sig_name}"
            bus.put(qualified, value)
            bus.put(sig_name, value)


class ClassicalControlNode(ExecutionNode):
    """Placeholder for non-ONNX nodes like PD control.  In the generic
    runner these are no-ops — the actual PD math lives in the simulation
    engine or hardware driver."""

    def __init__(self, name: str, node_cfg: dict):
        super().__init__(name, node_cfg.get("frequency_hz", 500))

    def tick(self, bus: SignalBus, input_names: list[str],
             output_names: list[str]) -> None:
        pass


# ---------------------------------------------------------------------------
# Execution Graph
# ---------------------------------------------------------------------------

class ExecutionGraph:
    """The full DAG: nodes, edges, and a topological schedule."""

    def __init__(self):
        self.nodes: dict[str, ExecutionNode] = {}
        self.edges: list[DataflowEdge] = []
        self._node_inputs: dict[str, list[str]] = {}
        self._node_outputs: dict[str, list[str]] = {}
        self._schedule: list[str] = []

    def add_node(self, node: ExecutionNode) -> None:
        self.nodes[node.name] = node

    def set_edges(self, edges: list[DataflowEdge]) -> None:
        self.edges = edges
        self._node_inputs.clear()
        self._node_outputs.clear()
        for edge in edges:
            self._node_inputs.setdefault(edge.node_name, []).extend(edge.inputs)
            self._node_outputs.setdefault(edge.node_name, []).extend(edge.outputs)
        self._build_schedule()

    def get_inputs(self, node_name: str) -> list[str]:
        return self._node_inputs.get(node_name, [])

    def get_outputs(self, node_name: str) -> list[str]:
        return self._node_outputs.get(node_name, [])

    def _build_schedule(self) -> None:
        """Topological sort of nodes based on dataflow edges."""
        produces: dict[str, str] = {}
        for edge in self.edges:
            for out_name in edge.outputs:
                produces[out_name] = edge.node_name

        deps: dict[str, set[str]] = {name: set() for name in self.nodes}
        for edge in self.edges:
            for inp in edge.inputs:
                clean = inp.split(".")[-1] if "." in inp else inp
                producer = produces.get(clean) or produces.get(inp)
                if producer and producer != edge.node_name:
                    deps.setdefault(edge.node_name, set()).add(producer)

        visited: set[str] = set()
        order: list[str] = []

        def visit(n: str):
            if n in visited:
                return
            visited.add(n)
            for dep in deps.get(n, set()):
                visit(dep)
            order.append(n)

        for n in self.nodes:
            visit(n)

        self._schedule = order

    @property
    def schedule(self) -> list[str]:
        return self._schedule


# ---------------------------------------------------------------------------
# Multi-rate Scheduler
# ---------------------------------------------------------------------------

def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


class Scheduler:
    """Tick-based multi-rate scheduler.

    Computes a master tick rate from the LCM of all node frequencies, then
    fires each node at the correct divisor.
    """

    def __init__(self, graph: ExecutionGraph, bus: SignalBus):
        self._graph = graph
        self._bus = bus

        freqs = [int(n.frequency_hz) for n in graph.nodes.values()
                 if n.frequency_hz > 0]
        if not freqs:
            freqs = [1]
        self._master_hz = freqs[0]
        for f in freqs[1:]:
            self._master_hz = _lcm(self._master_hz, f)
        self._master_hz = min(self._master_hz, 1000)

        self._divisors: dict[str, int] = {}
        for name, node in graph.nodes.items():
            freq = max(int(node.frequency_hz), 1)
            self._divisors[name] = max(self._master_hz // freq, 1)

        self._tick = 0
        self._running = False

    @property
    def master_hz(self) -> int:
        return self._master_hz

    def run(self) -> None:
        """Run the scheduler loop until stopped."""
        self._running = True
        dt = 1.0 / self._master_hz
        print(f"\nScheduler running at {self._master_hz} Hz master tick")
        print(f"  Press Ctrl+C to stop\n")

        while self._running:
            t0 = time.monotonic()

            for node_name in self._graph.schedule:
                if not self._running:
                    break
                divisor = self._divisors.get(node_name, 1)
                if self._tick % divisor != 0:
                    continue
                node = self._graph.nodes[node_name]
                inputs = self._graph.get_inputs(node_name)
                outputs = self._graph.get_outputs(node_name)
                try:
                    node.tick(self._bus, inputs, outputs)
                except Exception as e:
                    print(f"  Warning: node '{node_name}' raised {type(e).__name__}: {e}")

            self._tick += 1
            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        self._running = False

    def info_lines(self) -> list[str]:
        lines = [f"Master tick: {self._master_hz} Hz"]
        for name in self._graph.schedule:
            node = self._graph.nodes[name]
            div = self._divisors.get(name, 1)
            lines.append(f"  {name}: {node.frequency_hz} Hz (every {div} ticks)")
        return lines


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_graph(config: dict, controller_type: str | None = None) -> tuple[
    ExecutionGraph, SignalBus, ControllerSource | None,
]:
    """Build the execution graph from a robot config dict."""
    bus = SignalBus()
    graph = ExecutionGraph()
    controller: ControllerSource | None = None

    models_by_name = {m["name"]: m for m in config.get("models", [])}
    execution = config.get("execution", {})
    node_cfgs = execution.get("nodes", [])
    dataflow_lines = execution.get("dataflow", [])

    cfg_section = config.get("configuration", {})

    for nc in node_cfgs:
        name = nc["name"]
        node_type = nc.get("type", "model")

        if node_type == "controller":
            source_name = nc.get("source", "keyboard")
            cls = CONTROLLER_REGISTRY.get(source_name)
            if cls is None:
                print(f"Error: unknown controller source '{source_name}'. "
                      f"Available: {list(CONTROLLER_REGISTRY.keys())}")
                sys.exit(1)
            source = cls()
            config_key = nc.get("config_key", "controller")
            ctrl_cfg = cfg_section.get(config_key, {})
            source.setup(ctrl_cfg)
            if controller_type and source_name == controller_type:
                source.set_enabled(True)
            controller = source
            graph.add_node(ControllerNode(name, source, nc))
            continue

        model_name = nc.get("model", name)
        model_cfg = models_by_name.get(model_name, {})
        model_type = model_cfg.get("type", "")

        if model_type == "classical_control":
            graph.add_node(ClassicalControlNode(name, nc))
        else:
            graph.add_node(ModelNode(name, model_cfg, nc))

    # If --controller was requested but no controller node exists in the YAML,
    # auto-create one from the controller config section.
    if controller_type and controller is None:
        cls = CONTROLLER_REGISTRY.get(controller_type)
        if cls is None:
            print(f"Error: unknown controller type '{controller_type}'. "
                  f"Available: {list(CONTROLLER_REGISTRY.keys())}")
            sys.exit(1)
        source = cls()
        ctrl_cfg = cfg_section.get("controller", {})
        if not ctrl_cfg:
            print(f"Error: --controller {controller_type} requested but no "
                  f"controller section found in configuration")
            sys.exit(1)
        source.setup(ctrl_cfg)
        source.set_enabled(True)
        controller = source
        auto_name = f"{controller_type}_control"
        node_cfg = {
            "frequency_hz": 50,
            "outputs": source.output_names,
        }
        graph.add_node(ControllerNode(auto_name, source, node_cfg))

    edges = parse_dataflow(dataflow_lines)
    controller_node_names = {
        n for n, node in graph.nodes.items()
        if isinstance(node, ControllerNode)
    }
    tag_controller_edges(edges, controller_node_names)
    graph.set_edges(edges)

    return graph, bus, controller


def load_models(graph: ExecutionGraph) -> None:
    """Load ONNX sessions for all ModelNode instances in the graph."""
    for node in graph.nodes.values():
        if isinstance(node, ModelNode):
            node.load()


# ---------------------------------------------------------------------------
# GraphPolicyRunner — bridge between YAML graph and MuJoCo engine
# ---------------------------------------------------------------------------

class GraphPolicyRunner(PolicyRunner):
    """Generic PolicyRunner that delegates all robot-specific logic to a
    WBC handler class specified in the YAML (configuration.wbc.handler).

    The handler provides three methods:
      - setup(bus, config, model, data) -> n_actuated
      - pre_step(bus, model, data)
      - post_step(bus) -> (target_pos, kps, kds)
    """

    def __init__(self, graph: ExecutionGraph, bus: SignalBus, config: dict):
        self._graph = graph
        self._bus = bus
        self._config = config
        self._controller: ControllerSource | None = None
        self._wbc_handler = self._load_wbc_handler(config)

    def _load_wbc_handler(self, config):
        handler_path = (config.get("configuration", {})
                        .get("wbc", {}).get("handler"))
        if not handler_path:
            raise ValueError(
                "configuration.wbc.handler is required in the robot YAML. "
                "It should point to a WBC handler class, e.g. "
                "'handlers.sonic_wbc.SonicWBCHandler'")
        cls = load_handler(handler_path)
        return cls()

    @property
    def name(self) -> str:
        meta = self._config.get("metadata", {})
        return meta.get("name", "Graph Runner")

    def set_controller(self, controller):
        self._controller = controller

    def setup(self, model, data):
        return self._wbc_handler.setup(self._bus, self._config, model, data)

    def step(self, model, data):
        self._wbc_handler.pre_step(self._bus, model, data)

        if self._controller is not None and self._controller.enabled:
            for sig_name, value in self._controller.poll().items():
                self._bus.put(sig_name, value)

        for node_name in self._graph.schedule:
            node = self._graph.nodes[node_name]
            inputs = self._graph.get_inputs(node_name)
            outputs = self._graph.get_outputs(node_name)
            try:
                node.tick(self._bus, inputs, outputs)
            except Exception as e:
                print(f"  Warning: node '{node_name}' raised "
                      f"{type(e).__name__}: {e}")

        return self._wbc_handler.post_step(self._bus)

    def info_lines(self):
        lines = [
            f"Graph: {len(self._graph.nodes)} nodes, "
            f"{len(self._graph.edges)} edges",
        ]
        if self._controller is not None:
            lines.append(
                f"Controller: "
                f"{'enabled' if self._controller.enabled else 'disabled'}")
        return lines


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def print_banner(config: dict, graph: ExecutionGraph,
                 scheduler: Scheduler, controller: ControllerSource | None) -> None:
    meta = config.get("metadata", {})
    print("=" * 60)
    print(f"  Model Runner")
    print(f"  Robot : {meta.get('name', '?')}")
    print(f"  Desc  : {meta.get('description', '')}")
    print("=" * 60)

    print(f"\n  Execution graph ({len(graph.nodes)} nodes, "
          f"{len(graph.edges)} edges):")
    for name in graph.schedule:
        node = graph.nodes[name]
        kind = type(node).__name__
        print(f"    {name} ({kind}, {node.frequency_hz} Hz)")

    print(f"\n  Dataflow:")
    for edge in graph.edges:
        tag = f" [{edge.source_tag}]" if edge.source_tag != "autonomous" else ""
        print(f"    {', '.join(edge.inputs)} -> {edge.node_name} "
              f"-> {', '.join(edge.outputs)}{tag}")

    print()
    for line in scheduler.info_lines():
        print(f"  {line}")

    if controller is not None:
        print()
        for line in controller.banner_lines():
            print(f"  {line}")
        status = "ENABLED" if controller.enabled else "DISABLED (press C to toggle)"
        print(f"  Controller status: {status}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generic ONNX model runner — execute any robot pipeline from YAML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--robot", required=True,
        help="Path to robot definition YAML",
    )
    parser.add_argument(
        "--controller", default=None,
        choices=list(CONTROLLER_REGISTRY.keys()),
        help="Activate a controller source (e.g. keyboard). "
             "The controller is wired into the DAG as declared in the YAML dataflow.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolved execution graph and exit without running",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    config = load_robot_config(args.robot)

    graph, bus, controller = build_graph(config, args.controller)

    if args.dry_run:
        scheduler = Scheduler(graph, bus)
        print_banner(config, graph, scheduler, controller)
        print("\n[dry-run] Resolved configuration:\n")
        print(json.dumps(config, indent=2, default=str))
        return

    print("\nLoading models...")
    load_models(graph)

    scheduler = Scheduler(graph, bus)
    print_banner(config, graph, scheduler, controller)

    signal_mod.signal(signal_mod.SIGINT, lambda *_: scheduler.stop())

    print("\nStarting execution loop...")
    scheduler.run()
    print("\nStopped.")


if __name__ == "__main__":
    main()
