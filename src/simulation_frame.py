"""
Module simulation_frame
"""
import numpy as np


class SimulationFrames:
    def __init__(self, transitions):
        self.transitions = transitions
        self.event_time_series = None

    def run(self, start_at=None, size=1e5, frames=10, frame_time='5ms', seed=None):
         