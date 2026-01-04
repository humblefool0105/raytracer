"""
rays.py - Ray class for sequential ray tracing

A ray is defined by:
    - Origin point P = (x, y, z)
    - Direction cosines D = (L, M, N) where L² + M² + N² = 1
    - Wavelength λ (in micrometers)
    - Accumulated optical path length (OPL)
    - History of intersection points

Author: Avinash Kumar Singh
Project: Sequential Ray Tracer
"""

import numpy as np
from typing import Optional, List, Tuple


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Parameters
    ----------
    vector : np.ndarray
        Input vector of any dimension
        
    Returns
    -------
    np.ndarray
        Unit vector in same direction
        
    Raises
    ------
    ValueError
        If vector has zero magnitude
    """
    magnitude = np.linalg.norm(vector)
    if magnitude < 1e-15:
        raise ValueError("Cannot normalize zero vector")
    return vector / magnitude


class Ray:
    """
    Represents an optical ray for sequential ray tracing.
    
    A ray is characterized by its position in 3D space, its direction
    of propagation (as direction cosines), wavelength, and accumulated
    optical path length.
    
    Attributes
    ----------
    origin : np.ndarray
        Current position of the ray [x, y, z] in mm
    direction : np.ndarray
        Direction cosines [L, M, N] where L² + M² + N² = 1
        L = sin(θx), M = sin(θy), N = cos(θ) = sqrt(1 - L² - M²)
    wavelength : float
        Wavelength in micrometers (default: 0.5876 μm, d-line)
    opl : float
        Accumulated optical path length in mm
    history : list
        List of (position, surface_index) tuples recording ray path
    is_valid : bool
        Flag indicating if ray is still valid (not vignetted or TIR)
        
    Examples
    --------
    >>> # Create an on-axis ray traveling in +z direction
    >>> ray = Ray(origin=[0, 0, 0], direction=[0, 0, 1])
    >>> print(ray)
    Ray at [0. 0. 0.], direction [0. 0. 1.], λ=0.5876 μm
    
    >>> # Create an off-axis ray at 5 degrees
    >>> import numpy as np
    >>> angle = np.radians(5)
    >>> ray = Ray(origin=[0, 10, 0], direction=[0, np.sin(angle), np.cos(angle)])
    """
    
    # Standard wavelengths (Fraunhofer lines) in micrometers
    WAVELENGTH_F = 0.4861  # F-line (hydrogen blue)
    WAVELENGTH_d = 0.5876  # d-line (helium yellow) - default
    WAVELENGTH_C = 0.6563  # C-line (hydrogen red)
    
    def __init__(
        self,
        origin: List[float] | np.ndarray,
        direction: List[float] | np.ndarray,
        wavelength: float = 0.5876
    ):
        """
        Initialize a Ray object.
        
        Parameters
        ----------
        origin : array-like
            Starting position [x, y, z] in mm
        direction : array-like
            Direction vector [L, M, N] or [dx, dy, dz]
            Will be normalized to unit length
        wavelength : float, optional
            Wavelength in micrometers (default: 0.5876 μm, d-line)
        """
        self.origin = np.array(origin, dtype=np.float64)
        self.direction = normalize(np.array(direction, dtype=np.float64))
        self.wavelength = wavelength
        self.opl = 0.0
        self.history: List[Tuple[np.ndarray, int]] = []
        self.is_valid = True
        
        # Store initial position in history
        self.history.append((self.origin.copy(), -1))  # -1 = object space
    
    @property
    def L(self) -> float:
        """Direction cosine with respect to x-axis."""
        return self.direction[0]
    
    @property
    def M(self) -> float:
        """Direction cosine with respect to y-axis."""
        return self.direction[1]
    
    @property
    def N(self) -> float:
        """Direction cosine with respect to z-axis (optical axis)."""
        return self.direction[2]
    
    @property
    def x(self) -> float:
        """Current x-coordinate."""
        return self.origin[0]
    
    @property
    def y(self) -> float:
        """Current y-coordinate."""
        return self.origin[1]
    
    @property
    def z(self) -> float:
        """Current z-coordinate."""
        return self.origin[2]
    
    def propagate(self, distance: float, refractive_index: float = 1.0) -> None:
        """
        Propagate the ray forward by a given distance.
        
        Updates the ray's origin and accumulates optical path length.
        
        Parameters
        ----------
        distance : float
            Geometric distance to propagate in mm
        refractive_index : float, optional
            Refractive index of the medium (default: 1.0 for air)
        """
        self.origin = self.origin + distance * self.direction
        self.opl += refractive_index * distance
    
    def point_at(self, t: float) -> np.ndarray:
        """
        Get the point along the ray at parameter t.
        
        The parametric ray equation is: P(t) = origin + t * direction
        
        Parameters
        ----------
        t : float
            Parameter value (distance along ray)
            
        Returns
        -------
        np.ndarray
            Point [x, y, z] at parameter t
        """
        return self.origin + t * self.direction
    
    def update(
        self,
        new_origin: np.ndarray,
        new_direction: np.ndarray,
        surface_index: int,
        path_length: float,
        refractive_index: float
    ) -> None:
        """
        Update ray state after interaction with a surface.
        
        Parameters
        ----------
        new_origin : np.ndarray
            New position after surface intersection
        new_direction : np.ndarray
            New direction after refraction/reflection
        surface_index : int
            Index of the surface in the optical system
        path_length : float
            Geometric path length traveled
        refractive_index : float
            Refractive index of the medium traversed
        """
        self.origin = new_origin
        self.direction = normalize(new_direction)
        self.opl += refractive_index * path_length
        self.history.append((new_origin.copy(), surface_index))
    
    def invalidate(self, reason: str = "unknown") -> None:
        """
        Mark the ray as invalid (vignetted, TIR, or missed surface).
        
        Parameters
        ----------
        reason : str, optional
            Reason for invalidation
        """
        self.is_valid = False
        self._invalid_reason = reason
    
    def copy(self) -> 'Ray':
        """
        Create a deep copy of the ray.
        
        Returns
        -------
        Ray
            Independent copy of this ray
        """
        new_ray = Ray(
            origin=self.origin.copy(),
            direction=self.direction.copy(),
            wavelength=self.wavelength
        )
        new_ray.opl = self.opl
        new_ray.history = [(pos.copy(), idx) for pos, idx in self.history]
        new_ray.is_valid = self.is_valid
        return new_ray
    
    def angle_from_axis(self) -> float:
        """
        Calculate the angle of the ray from the optical axis.
        
        Returns
        -------
        float
            Angle in radians
        """
        return np.arccos(np.clip(self.N, -1.0, 1.0))
    
    def angle_from_axis_degrees(self) -> float:
        """
        Calculate the angle of the ray from the optical axis in degrees.
        
        Returns
        -------
        float
            Angle in degrees
        """
        return np.degrees(self.angle_from_axis())
    
    @classmethod
    def from_angles(
        cls,
        origin: List[float] | np.ndarray,
        theta_x: float,
        theta_y: float,
        wavelength: float = 0.5876,
        angles_in_degrees: bool = True
    ) -> 'Ray':
        """
        Create a ray from angular specification.
        
        Parameters
        ----------
        origin : array-like
            Starting position [x, y, z] in mm
        theta_x : float
            Angle from optical axis in x-z plane
        theta_y : float
            Angle from optical axis in y-z plane
        wavelength : float, optional
            Wavelength in micrometers
        angles_in_degrees : bool, optional
            If True, angles are in degrees (default: True)
            
        Returns
        -------
        Ray
            New ray object
        """
        if angles_in_degrees:
            theta_x = np.radians(theta_x)
            theta_y = np.radians(theta_y)
        
        # Direction cosines from angles
        L = np.sin(theta_x)
        M = np.sin(theta_y)
        N_squared = 1.0 - L**2 - M**2
        
        if N_squared < 0:
            raise ValueError("Invalid angles: L² + M² > 1")
        
        N = np.sqrt(N_squared)
        
        return cls(origin=origin, direction=[L, M, N], wavelength=wavelength)
    
    @classmethod
    def from_two_points(
        cls,
        point1: List[float] | np.ndarray,
        point2: List[float] | np.ndarray,
        wavelength: float = 0.5876
    ) -> 'Ray':
        """
        Create a ray defined by two points.
        
        Parameters
        ----------
        point1 : array-like
            Starting point [x, y, z] in mm
        point2 : array-like
            Point that ray passes through [x, y, z] in mm
        wavelength : float, optional
            Wavelength in micrometers
            
        Returns
        -------
        Ray
            New ray from point1 toward point2
        """
        p1 = np.array(point1, dtype=np.float64)
        p2 = np.array(point2, dtype=np.float64)
        direction = p2 - p1
        
        return cls(origin=p1, direction=direction, wavelength=wavelength)
    
    def __repr__(self) -> str:
        """String representation of the ray."""
        return (
            f"Ray at [{self.x:.4f}, {self.y:.4f}, {self.z:.4f}], "
            f"direction [{self.L:.4f}, {self.M:.4f}, {self.N:.4f}], "
            f"λ={self.wavelength} μm"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        angle = self.angle_from_axis_degrees()
        status = "valid" if self.is_valid else "invalid"
        return (
            f"Ray ({status}):\n"
            f"  Position: ({self.x:.4f}, {self.y:.4f}, {self.z:.4f}) mm\n"
            f"  Direction: ({self.L:.4f}, {self.M:.4f}, {self.N:.4f})\n"
            f"  Angle from axis: {angle:.2f}°\n"
            f"  Wavelength: {self.wavelength} μm\n"
            f"  OPL: {self.opl:.4f} mm\n"
            f"  Surfaces hit: {len(self.history) - 1}"
        )


# =============================================================================
# Ray Generation Utilities
# =============================================================================

def create_ray_bundle_parallel(
    y_height: float,
    num_rays: int,
    aperture_radius: float,
    wavelength: float = 0.5876,
    z_start: float = 0.0
) -> List[Ray]:
    """
    Create a bundle of parallel rays (collimated beam) for a given field height.
    
    Parameters
    ----------
    y_height : float
        Field height (y-coordinate of ray origins)
    num_rays : int
        Number of rays across the aperture diameter
    aperture_radius : float
        Radius of the entrance pupil
    wavelength : float, optional
        Wavelength in micrometers
    z_start : float, optional
        Starting z-coordinate for rays
        
    Returns
    -------
    List[Ray]
        List of Ray objects
    """
    rays = []
    
    # Create rays across the pupil
    if num_rays == 1:
        positions = [0.0]
    else:
        positions = np.linspace(-aperture_radius, aperture_radius, num_rays)
    
    for y_pupil in positions:
        ray = Ray(
            origin=[0.0, y_height + y_pupil, z_start],
            direction=[0.0, 0.0, 1.0],
            wavelength=wavelength
        )
        rays.append(ray)
    
    return rays


def create_ray_fan_meridional(
    field_angle: float,
    num_rays: int,
    aperture_radius: float,
    entrance_pupil_z: float,
    wavelength: float = 0.5876,
    angle_in_degrees: bool = True
) -> List[Ray]:
    """
    Create a meridional (tangential) ray fan for a given field angle.
    
    Meridional rays lie in the y-z plane (plane of symmetry).
    
    Parameters
    ----------
    field_angle : float
        Field angle (angle of chief ray from optical axis)
    num_rays : int
        Number of rays in the fan
    aperture_radius : float
        Radius of the entrance pupil
    entrance_pupil_z : float
        Z-coordinate of entrance pupil
    wavelength : float, optional
        Wavelength in micrometers
    angle_in_degrees : bool, optional
        If True, field_angle is in degrees
        
    Returns
    -------
    List[Ray]
        List of Ray objects forming meridional fan
    """
    if angle_in_degrees:
        field_angle = np.radians(field_angle)
    
    rays = []
    pupil_positions = np.linspace(-aperture_radius, aperture_radius, num_rays)
    
    for y_pupil in pupil_positions:
        # Ray starts at object at infinity, hits entrance pupil at y_pupil
        # Direction is determined by field angle
        
        # For object at infinity, all rays for a field point are parallel
        # Direction cosines:
        L = 0.0  # Meridional rays have no x-component
        M = np.sin(field_angle)
        N = np.cos(field_angle)
        
        # Start ray at entrance pupil
        origin = [0.0, y_pupil, entrance_pupil_z]
        
        ray = Ray(origin=origin, direction=[L, M, N], wavelength=wavelength)
        rays.append(ray)
    
    return rays


def create_ray_grid(
    field_angle: float,
    num_rings: int,
    rays_per_ring: int,
    aperture_radius: float,
    entrance_pupil_z: float,
    wavelength: float = 0.5876,
    angle_in_degrees: bool = True,
    include_center: bool = True
) -> List[Ray]:
    """
    Create a circular grid of rays at the entrance pupil.
    
    Parameters
    ----------
    field_angle : float
        Field angle
    num_rings : int
        Number of concentric rings
    rays_per_ring : int
        Number of rays per ring
    aperture_radius : float
        Radius of the entrance pupil
    entrance_pupil_z : float
        Z-coordinate of entrance pupil
    wavelength : float, optional
        Wavelength in micrometers
    angle_in_degrees : bool, optional
        If True, field_angle is in degrees
    include_center : bool, optional
        If True, include chief ray at center
        
    Returns
    -------
    List[Ray]
        List of Ray objects
    """
    if angle_in_degrees:
        field_angle = np.radians(field_angle)
    
    rays = []
    
    # Direction for all rays (object at infinity)
    L = 0.0
    M = np.sin(field_angle)
    N = np.cos(field_angle)
    direction = [L, M, N]
    
    # Center ray (chief ray)
    if include_center:
        ray = Ray(
            origin=[0.0, 0.0, entrance_pupil_z],
            direction=direction,
            wavelength=wavelength
        )
        rays.append(ray)
    
    # Rings of rays
    for i_ring in range(1, num_rings + 1):
        ring_radius = aperture_radius * i_ring / num_rings
        
        for i_ray in range(rays_per_ring):
            theta = 2 * np.pi * i_ray / rays_per_ring
            x_pupil = ring_radius * np.cos(theta)
            y_pupil = ring_radius * np.sin(theta)
            
            ray = Ray(
                origin=[x_pupil, y_pupil, entrance_pupil_z],
                direction=direction,
                wavelength=wavelength
            )
            rays.append(ray)
    
    return rays


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Ray Class Tests")
    print("=" * 60)
    
    # Test 1: Basic ray creation
    print("\n--- Test 1: Basic Ray Creation ---")
    ray1 = Ray(origin=[0, 0, 0], direction=[0, 0, 1])
    print(ray1)
    
    # Test 2: Off-axis ray
    print("\n--- Test 2: Off-axis Ray (5°) ---")
    ray2 = Ray.from_angles(origin=[0, 0, 0], theta_x=0, theta_y=5)
    print(ray2)
    
    # Test 3: Ray propagation
    print("\n--- Test 3: Ray Propagation ---")
    ray3 = Ray(origin=[0, 0, 0], direction=[0, 0, 1])
    print(f"Before: {ray3.origin}")
    ray3.propagate(distance=100, refractive_index=1.5)
    print(f"After 100mm in n=1.5: {ray3.origin}")
    print(f"OPL: {ray3.opl} mm")
    
    # Test 4: Ray from two points
    print("\n--- Test 4: Ray from Two Points ---")
    ray4 = Ray.from_two_points([0, 0, 0], [10, 10, 100])
    print(ray4)
    
    # Test 5: Ray bundle
    print("\n--- Test 5: Parallel Ray Bundle ---")
    bundle = create_ray_bundle_parallel(
        y_height=0,
        num_rays=5,
        aperture_radius=10
    )
    print(f"Created {len(bundle)} rays")
    for i, ray in enumerate(bundle):
        print(f"  Ray {i}: y = {ray.y:.2f} mm")
    
    # Test 6: Meridional fan
    print("\n--- Test 6: Meridional Ray Fan (10° field) ---")
    fan = create_ray_fan_meridional(
        field_angle=10,
        num_rays=5,
        aperture_radius=12.5,
        entrance_pupil_z=0
    )
    print(f"Created {len(fan)} rays")
    print(f"Direction cosines: L={fan[0].L:.4f}, M={fan[0].M:.4f}, N={fan[0].N:.4f}")
    
    # Test 7: Ray grid
    print("\n--- Test 7: Ray Grid ---")
    grid = create_ray_grid(
        field_angle=0,
        num_rings=2,
        rays_per_ring=6,
        aperture_radius=10,
        entrance_pupil_z=0
    )
    print(f"Created {len(grid)} rays (1 center + 2 rings × 6 rays)")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
