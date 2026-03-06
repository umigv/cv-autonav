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


@dataclass
class CameraPosition:
    x: float = 0.0 # mm, positive is right of wheelbase
    y: float = 0.0 # mm, positive is forward from wheelbase
    h: float = 0.0 # radians, positive is anticlockwise
