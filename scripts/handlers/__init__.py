"""Handler plugins for model_runner.py.

Each handler is a callable with signature:

    def my_handler(bus: SignalBus, model_cfg: dict, node_cfg: dict) -> dict

It reads signals from the bus, applies custom preprocessing, and returns
a dict mapping ONNX input names to numpy arrays.  If it returns None,
the node skips inference for this tick.

Reference the handler in a YAML execution node:

    nodes:
      - name: "encoder_inference"
        model: "sonic_encoder"
        handler: "handlers.sonic_encoder.build_obs"
"""
