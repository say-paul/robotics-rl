from .stand import StandAction
from .walk import WalkAction
from .policy import OnnxPolicyAction

ACTION_REGISTRY = {
    "stand": StandAction,
    "walk": WalkAction,
    "policy": OnnxPolicyAction,
}
