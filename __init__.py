"""
raytracer - A sequential ray tracing engine in Python

A portfolio project demonstrating optical system design and analysis,
replicating core functionality of commercial software like ZEMAX and CODE V.

Author: Avinash Kumar Singh
"""

from .rays import Ray, normalize
from .rays import create_ray_bundle_parallel, create_ray_fan_meridional, create_ray_grid

__version__ = "0.1.0"
__author__ = "Avinash Kumar Singh"

__all__ = [
    "Ray",
    "normalize",
    "create_ray_bundle_parallel",
    "create_ray_fan_meridional",
    "create_ray_grid",
]
