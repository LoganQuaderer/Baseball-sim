from dataclasses import dataclass
from typing import Dict

@dataclass
class BatterAttributes:
    """Core attributes for a batter"""
    contact: float  # 0-100
    power: float  # 0-100
    eye: float  # plate discipline, 0-100
    speed: float  # 0-100

@dataclass
class Batter:
    """Class representing a baseball batter with relevant attributes"""
    name: str
    bats: str  # 'L', 'R', or 'S' for switch
    attributes: BatterAttributes