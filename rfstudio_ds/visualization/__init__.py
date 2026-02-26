import logging

from ._base import Visualizer


# Suppress INFO logs from the websockets library
logging.getLogger("websockets").setLevel(logging.WARNING)

__all__ = [
    'Visualizer',
]
