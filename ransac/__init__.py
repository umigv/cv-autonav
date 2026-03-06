from dataclasses import dataclass


@dataclass
class Intrinsics:
    cx: float
    cy: float
    fx: float
    fy: float
    tx: float = 0


@dataclass
class GridConfiguration:
    gw: float  # grid width in mm
    gh: float  # grid height in mm
    cw: float  # cell width in mm

