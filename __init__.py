"""
raytracer - A sequential ray tracing engine in Python

A portfolio project demonstrating optical system design and analysis,
replicating core functionality of commercial software like ZEMAX and CODE V.

Author: Avinash Kumar Singh
"""

from .rays import Ray, normalize
from .rays import create_ray_bundle_parallel, create_ray_fan_meridional, create_ray_grid

from .surfaces import Surface, SphericalSurface, ConicSurface, AsphereSurface
from .surfaces import SurfaceType
from .surfaces import create_flat_surface, create_mirror, create_parabolic_mirror

__version__ = "0.1.0"
__author__ = "Avinash Kumar Singh"

__all__ = [
    # Rays
    "Ray",
    "normalize",
    "create_ray_bundle_parallel",
    "create_ray_fan_meridional",
    "create_ray_grid",
    # Surfaces
    "Surface",
    "SphericalSurface",
    "ConicSurface",
    "AsphereSurface",
    "SurfaceType",
    "create_flat_surface",
    "create_mirror",
    "create_parabolic_mirror",
]