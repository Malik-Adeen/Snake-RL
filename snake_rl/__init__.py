"""Snake RL — tabular Q-learning project package."""

from .agent import TabularQAgent
from .env import SnakeEnv
from .multi_env import MultiSnakeEnv
from .multi_train import multi_train
from .train import train

# Renderer imports pygame — only load when actually needed
# This prevents import failures in headless/non-visual environments
def _get_renderer():
    from .renderer import SnakeRenderer, MultiSnakeRenderer
    return SnakeRenderer, MultiSnakeRenderer

__all__ = [
    "SnakeEnv",
    "MultiSnakeEnv",
    "TabularQAgent",
    "train",
    "multi_train",
    "SnakeRenderer",
    "MultiSnakeRenderer",
]

# Lazy-load renderer so non-visual scripts work without pygame
try:
    from .renderer import SnakeRenderer, MultiSnakeRenderer
except ImportError:
    pass
