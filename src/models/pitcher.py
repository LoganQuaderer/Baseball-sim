from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class Pitcher:
    """Class representing a baseball pitcher with relevant attributes and methods"""
    
    # Basic attributes
    name: str
    throws: str  # 'L' or 'R'
    
    # Core pitching attributes (0-100 scale)
    velocity: float
    control: float
    movement: float
    stamina: float
    
    # Pitch arsenal with confidence levels (0-100)
    pitch_types: Dict[str, float] = None
    
    def __post_init__(self):
        if self.pitch_types is None:
            # Default pitch arsenal
            self.pitch_types = {
                "4-Seam Fastball": 100,
                "Changeup": 70,
                "Slider": 60
            }