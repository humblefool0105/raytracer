"""
surfaces.py - Optical surface classes for sequential ray tracing

Surface types:
    - SphericalSurface: Basic spherical refracting/reflecting surface
    - ConicSurface: Conic sections (parabola, hyperbola, ellipse)
    - AsphereSurface: Conic + polynomial aspheric terms

Each surface knows:
    - Its shape (sag equation)
    - How to compute surface normals
    - Its aperture (clear diameter)
    - Material on each side (for refraction)

Author: Avinash Kumar Singh
Project: Sequential Ray Tracer
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum


class SurfaceType(Enum):
    """Enumeration of surface interaction types."""
    REFRACTING = "refracting"
    REFLECTING = "reflecting"
    DUMMY = "dummy"  # No interaction, just a reference plane


class Surface:
    """
    Base class for all optical surfaces.
    
    A surface is defined by:
        - Vertex position (on optical axis)
        - Radius of curvature (or curvature c = 1/R)
        - Clear aperture (semi-diameter)
        - Thickness to next surface
        - Refractive indices on each side
    
    Sign Convention:
        - Radius R > 0: Center of curvature to the RIGHT of vertex
        - Radius R < 0: Center of curvature to the LEFT of vertex
        - R = infinity (c = 0): Flat surface
    
    Attributes
    ----------
    radius : float
        Radius of curvature in mm (np.inf for flat)
    curvature : float
        Curvature c = 1/R in mm^-1 (0 for flat)
    thickness : float
        Distance to next surface in mm
    n1 : float
        Refractive index before surface (object side)
    n2 : float
        Refractive index after surface (image side)
    aperture : float
        Semi-diameter (clear radius) in mm
    surface_type : SurfaceType
        Refracting, reflecting, or dummy
    """
    
    def __init__(
        self,
        radius: float = np.inf,
        thickness: float = 0.0,
        n1: float = 1.0,
        n2: float = 1.5,
        aperture: float = 25.0,
        surface_type: SurfaceType = SurfaceType.REFRACTING
    ):
        """
        Initialize a Surface.
        
        Parameters
        ----------
        radius : float, optional
            Radius of curvature in mm (default: infinity = flat)
        thickness : float, optional
            Distance to next surface in mm (default: 0)
        n1 : float, optional
            Refractive index before surface (default: 1.0 = air)
        n2 : float, optional
            Refractive index after surface (default: 1.5 = typical glass)
        aperture : float, optional
            Semi-diameter in mm (default: 25.0)
        surface_type : SurfaceType, optional
            Type of surface interaction (default: REFRACTING)
        """
        self.radius = radius
        self.thickness = thickness
        self.n1 = n1
        self.n2 = n2
        self.aperture = aperture
        self.surface_type = surface_type
        
        # Compute curvature (handle infinity)
        if np.isinf(radius):
            self.curvature = 0.0
        else:
            self.curvature = 1.0 / radius
    
    @property
    def c(self) -> float:
        """Shorthand for curvature."""
        return self.curvature
    
    @property
    def R(self) -> float:
        """Shorthand for radius."""
        return self.radius
    
    @property
    def is_flat(self) -> bool:
        """Check if surface is flat (planar)."""
        return abs(self.curvature) < 1e-15
    
    @property
    def is_mirror(self) -> bool:
        """Check if surface is reflective."""
        return self.surface_type == SurfaceType.REFLECTING
    
    @property
    def power(self) -> float:
        """
        Optical power of the surface in mm^-1.
        
        Power φ = (n2 - n1) * c = (n2 - n1) / R
        """
        return (self.n2 - self.n1) * self.curvature
    
    def sag(self, x: float, y: float) -> float:
        """
        Calculate surface sag at point (x, y).
        
        Sag is the z-displacement from the vertex plane.
        Must be implemented by subclasses.
        
        Parameters
        ----------
        x : float
            X-coordinate in mm
        y : float
            Y-coordinate in mm
            
        Returns
        -------
        float
            Sag (z-displacement) in mm
        """
        raise NotImplementedError("Subclasses must implement sag()")
    
    def sag_array(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate surface sag for arrays of points.
        
        Parameters
        ----------
        x : np.ndarray
            X-coordinates in mm
        y : np.ndarray
            Y-coordinates in mm
            
        Returns
        -------
        np.ndarray
            Sag values in mm
        """
        raise NotImplementedError("Subclasses must implement sag_array()")
    
    def normal(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculate outward surface normal at point (x, y, z).
        
        Must be implemented by subclasses.
        
        Parameters
        ----------
        x : float
            X-coordinate of point on surface
        y : float
            Y-coordinate of point on surface
        z : float
            Z-coordinate of point on surface
            
        Returns
        -------
        np.ndarray
            Unit normal vector [nx, ny, nz]
        """
        raise NotImplementedError("Subclasses must implement normal()")
    
    def is_inside_aperture(self, x: float, y: float) -> bool:
        """
        Check if point (x, y) is within the clear aperture.
        
        Parameters
        ----------
        x : float
            X-coordinate in mm
        y : float
            Y-coordinate in mm
            
        Returns
        -------
        bool
            True if point is within aperture
        """
        r = np.sqrt(x**2 + y**2)
        return r <= self.aperture
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"R={self.radius:.2f}, "
            f"t={self.thickness:.2f}, "
            f"n={self.n1:.4f}/{self.n2:.4f}, "
            f"aperture={self.aperture:.2f})"
        )


class SphericalSurface(Surface):
    """
    Spherical optical surface.
    
    Sag equation:
        z(r) = c * r² / (1 + sqrt(1 - c² * r²))
        
    where:
        c = 1/R is the curvature
        r = sqrt(x² + y²) is the radial distance
        
    This is the exact sag equation for a sphere, not the paraxial
    approximation z ≈ r²/(2R).
    """
    
    def sag(self, x: float, y: float) -> float:
        """
        Calculate spherical surface sag at point (x, y).
        
        Parameters
        ----------
        x : float
            X-coordinate in mm
        y : float
            Y-coordinate in mm
            
        Returns
        -------
        float
            Sag (z-displacement) in mm
        """
        if self.is_flat:
            return 0.0
        
        r2 = x**2 + y**2
        c = self.curvature
        c2r2 = c**2 * r2
        
        # Check for valid range (ray must hit surface)
        if c2r2 >= 1.0:
            return np.nan  # Outside surface
        
        return c * r2 / (1.0 + np.sqrt(1.0 - c2r2))
    
    def sag_array(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate spherical sag for arrays of points.
        
        Parameters
        ----------
        x : np.ndarray
            X-coordinates in mm
        y : np.ndarray
            Y-coordinates in mm
            
        Returns
        -------
        np.ndarray
            Sag values in mm
        """
        if self.is_flat:
            return np.zeros_like(x)
        
        r2 = x**2 + y**2
        c = self.curvature
        c2r2 = c**2 * r2
        
        # Handle points outside surface
        result = np.zeros_like(r2)
        valid = c2r2 < 1.0
        result[valid] = c * r2[valid] / (1.0 + np.sqrt(1.0 - c2r2[valid]))
        result[~valid] = np.nan
        
        return result
    
    def normal(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculate outward surface normal at point (x, y, z).
        
        For a sphere centered at (0, 0, R), the normal points
        from the center toward the surface point.
        
        Parameters
        ----------
        x : float
            X-coordinate of point on surface
        y : float
            Y-coordinate of point on surface
        z : float
            Z-coordinate of point on surface
            
        Returns
        -------
        np.ndarray
            Unit normal vector [nx, ny, nz]
        """
        if self.is_flat:
            return np.array([0.0, 0.0, 1.0])
        
        # Sphere center is at (0, 0, R) where R = 1/c
        R = self.radius
        
        # Vector from center to surface point
        nx = x
        ny = y
        nz = z - R
        
        # Normalize
        mag = np.sqrt(nx**2 + ny**2 + nz**2)
        
        # Normal should point in +z direction for incident rays
        # For R > 0 (center to right), normal points left (-z at vertex)
        # For R < 0 (center to left), normal points right (+z at vertex)
        # We want normal pointing toward incoming light (into the surface)
        
        if R > 0:
            # Convex surface (as seen from left), normal points left
            return np.array([-nx/mag, -ny/mag, -nz/mag])
        else:
            # Concave surface (as seen from left), normal points right
            return np.array([nx/mag, ny/mag, nz/mag])


class ConicSurface(Surface):
    """
    Conic section surface (sphere, paraboloid, hyperboloid, ellipsoid).
    
    Sag equation:
        z(r) = c * r² / (1 + sqrt(1 - (1+k) * c² * r²))
        
    where:
        c = 1/R is the curvature
        k is the conic constant
        r = sqrt(x² + y²) is the radial distance
        
    Conic constant values:
        k = 0:     Sphere
        k = -1:    Paraboloid
        k < -1:    Hyperboloid
        -1 < k < 0: Prolate ellipsoid
        k > 0:     Oblate ellipsoid
        
    The conic constant relates to eccentricity e by: k = -e²
    """
    
    def __init__(
        self,
        radius: float = np.inf,
        thickness: float = 0.0,
        n1: float = 1.0,
        n2: float = 1.5,
        aperture: float = 25.0,
        conic: float = 0.0,
        surface_type: SurfaceType = SurfaceType.REFRACTING
    ):
        """
        Initialize a ConicSurface.
        
        Parameters
        ----------
        radius : float, optional
            Radius of curvature in mm (default: infinity = flat)
        thickness : float, optional
            Distance to next surface in mm (default: 0)
        n1 : float, optional
            Refractive index before surface (default: 1.0)
        n2 : float, optional
            Refractive index after surface (default: 1.5)
        aperture : float, optional
            Semi-diameter in mm (default: 25.0)
        conic : float, optional
            Conic constant k (default: 0 = sphere)
        surface_type : SurfaceType, optional
            Type of surface interaction
        """
        super().__init__(radius, thickness, n1, n2, aperture, surface_type)
        self.conic = conic
    
    @property
    def k(self) -> float:
        """Shorthand for conic constant."""
        return self.conic
    
    @property
    def is_sphere(self) -> bool:
        """Check if surface is spherical (k = 0)."""
        return abs(self.conic) < 1e-15
    
    @property
    def is_parabola(self) -> bool:
        """Check if surface is parabolic (k = -1)."""
        return abs(self.conic + 1.0) < 1e-15
    
    @property
    def conic_type(self) -> str:
        """Return string description of conic type."""
        k = self.conic
        if abs(k) < 1e-15:
            return "sphere"
        elif abs(k + 1) < 1e-15:
            return "paraboloid"
        elif k < -1:
            return "hyperboloid"
        elif k < 0:
            return "prolate ellipsoid"
        else:
            return "oblate ellipsoid"
    
    def sag(self, x: float, y: float) -> float:
        """
        Calculate conic surface sag at point (x, y).
        
        Parameters
        ----------
        x : float
            X-coordinate in mm
        y : float
            Y-coordinate in mm
            
        Returns
        -------
        float
            Sag (z-displacement) in mm
        """
        if self.is_flat:
            return 0.0
        
        r2 = x**2 + y**2
        c = self.curvature
        k = self.conic
        
        discriminant = 1.0 - (1.0 + k) * c**2 * r2
        
        # Check for valid range
        if discriminant <= 0:
            return np.nan  # Outside surface
        
        return c * r2 / (1.0 + np.sqrt(discriminant))
    
    def sag_array(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate conic sag for arrays of points.
        
        Parameters
        ----------
        x : np.ndarray
            X-coordinates in mm
        y : np.ndarray
            Y-coordinates in mm
            
        Returns
        -------
        np.ndarray
            Sag values in mm
        """
        if self.is_flat:
            return np.zeros_like(x)
        
        r2 = x**2 + y**2
        c = self.curvature
        k = self.conic
        
        discriminant = 1.0 - (1.0 + k) * c**2 * r2
        
        result = np.zeros_like(r2)
        valid = discriminant > 0
        result[valid] = c * r2[valid] / (1.0 + np.sqrt(discriminant[valid]))
        result[~valid] = np.nan
        
        return result
    
    def _dsag_dr(self, r: float) -> float:
        """
        Derivative of sag with respect to radial distance.
        
        dz/dr = cr / sqrt(1 - (1+k)c²r²)
        
        Parameters
        ----------
        r : float
            Radial distance in mm
            
        Returns
        -------
        float
            dz/dr at radius r
        """
        if self.is_flat or r < 1e-15:
            return 0.0
        
        c = self.curvature
        k = self.conic
        discriminant = 1.0 - (1.0 + k) * c**2 * r**2
        
        if discriminant <= 0:
            return np.nan
        
        return c * r / np.sqrt(discriminant)
    
    def normal(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculate outward surface normal at point (x, y, z).
        
        For a general surface z = f(x,y), the normal is:
            n = (-∂f/∂x, -∂f/∂y, 1) / |...|
            
        Parameters
        ----------
        x : float
            X-coordinate of point on surface
        y : float
            Y-coordinate of point on surface
        z : float
            Z-coordinate of point on surface
            
        Returns
        -------
        np.ndarray
            Unit normal vector [nx, ny, nz]
        """
        if self.is_flat:
            return np.array([0.0, 0.0, 1.0])
        
        r = np.sqrt(x**2 + y**2)
        
        if r < 1e-15:
            # At vertex, normal is along z-axis
            return np.array([0.0, 0.0, 1.0])
        
        # dz/dr
        dzdr = self._dsag_dr(r)
        
        # Convert to partial derivatives
        # dz/dx = (dz/dr) * (dr/dx) = (dz/dr) * (x/r)
        # dz/dy = (dz/dr) * (dr/dy) = (dz/dr) * (y/r)
        dzdx = dzdr * x / r
        dzdy = dzdr * y / r
        
        # Normal vector (unnormalized): (-dz/dx, -dz/dy, 1)
        nx = -dzdx
        ny = -dzdy
        nz = 1.0
        
        # Normalize
        mag = np.sqrt(nx**2 + ny**2 + nz**2)
        
        return np.array([nx/mag, ny/mag, nz/mag])
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"R={self.radius:.2f}, "
            f"k={self.conic:.4f} [{self.conic_type}], "
            f"t={self.thickness:.2f}, "
            f"n={self.n1:.4f}/{self.n2:.4f})"
        )


class AsphereSurface(ConicSurface):
    """
    Aspheric surface with polynomial deformation terms.
    
    Sag equation:
        z(r) = c*r² / (1 + sqrt(1 - (1+k)*c²*r²)) + Σ A_{2i} * r^{2i}
        
    where:
        c = 1/R is the curvature
        k is the conic constant
        A_4, A_6, A_8, ... are even aspheric coefficients
        
    The aspheric terms allow correction of higher-order aberrations
    beyond what conic sections can achieve.
    """
    
    def __init__(
        self,
        radius: float = np.inf,
        thickness: float = 0.0,
        n1: float = 1.0,
        n2: float = 1.5,
        aperture: float = 25.0,
        conic: float = 0.0,
        asph_coeffs: Optional[dict] = None,
        surface_type: SurfaceType = SurfaceType.REFRACTING
    ):
        """
        Initialize an AsphereSurface.
        
        Parameters
        ----------
        radius : float, optional
            Radius of curvature in mm
        thickness : float, optional
            Distance to next surface in mm
        n1 : float, optional
            Refractive index before surface
        n2 : float, optional
            Refractive index after surface
        aperture : float, optional
            Semi-diameter in mm
        conic : float, optional
            Conic constant k
        asph_coeffs : dict, optional
            Aspheric coefficients {4: A4, 6: A6, 8: A8, ...}
            Keys are even integers >= 4
        surface_type : SurfaceType, optional
            Type of surface interaction
        """
        super().__init__(radius, thickness, n1, n2, aperture, conic, surface_type)
        self.asph_coeffs = asph_coeffs or {}
        
        # Validate coefficients (must be even integers >= 4)
        for power in self.asph_coeffs.keys():
            if not isinstance(power, int) or power < 4 or power % 2 != 0:
                raise ValueError(f"Aspheric power must be even integer >= 4, got {power}")
    
    def _aspheric_term(self, r2: float) -> float:
        """
        Calculate the sum of aspheric polynomial terms.
        
        Parameters
        ----------
        r2 : float
            r² (radial distance squared)
            
        Returns
        -------
        float
            Sum of A_i * r^i terms
        """
        result = 0.0
        for power, coeff in self.asph_coeffs.items():
            # power is even: 4, 6, 8, ...
            # r^power = (r²)^(power/2)
            result += coeff * (r2 ** (power // 2))
        return result
    
    def _dasph_dr(self, r: float, r2: float) -> float:
        """
        Derivative of aspheric terms with respect to r.
        
        d/dr [A_i * r^i] = i * A_i * r^(i-1)
        
        Parameters
        ----------
        r : float
            Radial distance
        r2 : float
            r² (passed to avoid recomputation)
            
        Returns
        -------
        float
            Derivative of aspheric terms
        """
        result = 0.0
        for power, coeff in self.asph_coeffs.items():
            # d/dr [A * r^power] = power * A * r^(power-1)
            if power >= 4:
                result += power * coeff * (r ** (power - 1))
        return result
    
    def sag(self, x: float, y: float) -> float:
        """
        Calculate aspheric surface sag at point (x, y).
        
        Parameters
        ----------
        x : float
            X-coordinate in mm
        y : float
            Y-coordinate in mm
            
        Returns
        -------
        float
            Sag (z-displacement) in mm
        """
        # Get conic sag from parent class
        conic_sag = super().sag(x, y)
        
        if np.isnan(conic_sag):
            return np.nan
        
        # Add aspheric terms
        r2 = x**2 + y**2
        asph = self._aspheric_term(r2)
        
        return conic_sag + asph
    
    def sag_array(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate aspheric sag for arrays of points.
        
        Parameters
        ----------
        x : np.ndarray
            X-coordinates in mm
        y : np.ndarray
            Y-coordinates in mm
            
        Returns
        -------
        np.ndarray
            Sag values in mm
        """
        # Get conic sag from parent class
        conic_sag = super().sag_array(x, y)
        
        # Add aspheric terms
        r2 = x**2 + y**2
        asph = np.zeros_like(r2)
        for power, coeff in self.asph_coeffs.items():
            asph += coeff * (r2 ** (power // 2))
        
        return conic_sag + asph
    
    def normal(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Calculate outward surface normal at point (x, y, z).
        
        Parameters
        ----------
        x : float
            X-coordinate of point on surface
        y : float
            Y-coordinate of point on surface
        z : float
            Z-coordinate of point on surface
            
        Returns
        -------
        np.ndarray
            Unit normal vector [nx, ny, nz]
        """
        if self.is_flat and not self.asph_coeffs:
            return np.array([0.0, 0.0, 1.0])
        
        r2 = x**2 + y**2
        r = np.sqrt(r2)
        
        if r < 1e-15:
            return np.array([0.0, 0.0, 1.0])
        
        # Total dz/dr = conic part + aspheric part
        dzdr_conic = self._dsag_dr(r)
        dzdr_asph = self._dasph_dr(r, r2)
        dzdr = dzdr_conic + dzdr_asph
        
        # Convert to partial derivatives
        dzdx = dzdr * x / r
        dzdy = dzdr * y / r
        
        # Normal vector
        nx = -dzdx
        ny = -dzdy
        nz = 1.0
        
        mag = np.sqrt(nx**2 + ny**2 + nz**2)
        
        return np.array([nx/mag, ny/mag, nz/mag])
    
    def __repr__(self) -> str:
        """String representation."""
        asph_str = ", ".join([f"A{k}={v:.2e}" for k, v in sorted(self.asph_coeffs.items())])
        return (
            f"{self.__class__.__name__}("
            f"R={self.radius:.2f}, "
            f"k={self.conic:.4f}, "
            f"{asph_str}, "
            f"t={self.thickness:.2f})"
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_flat_surface(
    thickness: float,
    n1: float = 1.0,
    n2: float = 1.0,
    aperture: float = 25.0
) -> SphericalSurface:
    """
    Create a flat (planar) surface.
    
    Parameters
    ----------
    thickness : float
        Distance to next surface in mm
    n1 : float, optional
        Refractive index before surface
    n2 : float, optional
        Refractive index after surface
    aperture : float, optional
        Semi-diameter in mm
        
    Returns
    -------
    SphericalSurface
        Flat surface (R = infinity)
    """
    return SphericalSurface(
        radius=np.inf,
        thickness=thickness,
        n1=n1,
        n2=n2,
        aperture=aperture
    )


def create_mirror(
    radius: float,
    thickness: float = 0.0,
    aperture: float = 25.0,
    conic: float = 0.0
) -> ConicSurface:
    """
    Create a reflective surface (mirror).
    
    Parameters
    ----------
    radius : float
        Radius of curvature in mm
    thickness : float, optional
        Distance to next surface in mm
    aperture : float, optional
        Semi-diameter in mm
    conic : float, optional
        Conic constant (0=sphere, -1=parabola)
        
    Returns
    -------
    ConicSurface
        Reflective surface
    """
    return ConicSurface(
        radius=radius,
        thickness=thickness,
        n1=1.0,
        n2=1.0,  # Same medium (reflection)
        aperture=aperture,
        conic=conic,
        surface_type=SurfaceType.REFLECTING
    )


def create_parabolic_mirror(
    focal_length: float,
    aperture: float = 25.0
) -> ConicSurface:
    """
    Create a parabolic mirror with given focal length.
    
    For a parabolic mirror: f = R/2, so R = 2f
    
    Parameters
    ----------
    focal_length : float
        Focal length in mm (positive for concave)
    aperture : float, optional
        Semi-diameter in mm
        
    Returns
    -------
    ConicSurface
        Parabolic mirror
    """
    radius = 2.0 * focal_length
    return create_mirror(radius=radius, aperture=aperture, conic=-1.0)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Surface Class Tests")
    print("=" * 60)
    
    # Test 1: Spherical surface
    print("\n--- Test 1: Spherical Surface ---")
    sphere = SphericalSurface(radius=100, thickness=10, n1=1.0, n2=1.5, aperture=25)
    print(f"Surface: {sphere}")
    print(f"Curvature: {sphere.curvature:.6f} mm^-1")
    print(f"Power: {sphere.power:.6f} mm^-1")
    print(f"Sag at r=10mm: {sphere.sag(10, 0):.6f} mm")
    print(f"Sag at r=0: {sphere.sag(0, 0):.6f} mm")
    print(f"Normal at vertex: {sphere.normal(0, 0, 0)}")
    print(f"Normal at (10,0,z): {sphere.normal(10, 0, sphere.sag(10, 0))}")
    
    # Test 2: Flat surface
    print("\n--- Test 2: Flat Surface ---")
    flat = create_flat_surface(thickness=50, n1=1.5, n2=1.0)
    print(f"Surface: {flat}")
    print(f"Is flat: {flat.is_flat}")
    print(f"Sag at r=20mm: {flat.sag(20, 0):.6f} mm")
    print(f"Normal: {flat.normal(10, 10, 0)}")
    
    # Test 3: Conic surface (parabola)
    print("\n--- Test 3: Parabolic Surface ---")
    parabola = ConicSurface(radius=100, thickness=0, conic=-1.0, aperture=25)
    print(f"Surface: {parabola}")
    print(f"Conic type: {parabola.conic_type}")
    print(f"Sag at r=10mm (sphere): {sphere.sag(10, 0):.6f} mm")
    print(f"Sag at r=10mm (parabola): {parabola.sag(10, 0):.6f} mm")
    
    # Test 4: Different conic types
    print("\n--- Test 4: Conic Types ---")
    conics = [
        (0.0, "sphere"),
        (-0.5, "prolate ellipsoid"),
        (-1.0, "paraboloid"),
        (-1.5, "hyperboloid"),
        (0.5, "oblate ellipsoid"),
    ]
    for k, expected in conics:
        surf = ConicSurface(radius=100, conic=k)
        print(f"k={k:+.1f}: {surf.conic_type} (expected: {expected})")
    
    # Test 5: Aspheric surface
    print("\n--- Test 5: Aspheric Surface ---")
    asphere = AsphereSurface(
        radius=100,
        thickness=5,
        conic=-1.0,
        asph_coeffs={4: 1e-6, 6: -1e-9, 8: 1e-12},
        aperture=25
    )
    print(f"Surface: {asphere}")
    print(f"Sag at r=10mm (conic only): {parabola.sag(10, 0):.6f} mm")
    print(f"Sag at r=10mm (asphere): {asphere.sag(10, 0):.6f} mm")
    print(f"Difference: {(asphere.sag(10, 0) - parabola.sag(10, 0))*1000:.6f} μm")
    
    # Test 6: Parabolic mirror
    print("\n--- Test 6: Parabolic Mirror (f=500mm) ---")
    mirror = create_parabolic_mirror(focal_length=500, aperture=50)
    print(f"Surface: {mirror}")
    print(f"Radius: {mirror.radius:.1f} mm")
    print(f"Is mirror: {mirror.is_mirror}")
    
    # Test 7: Array sag calculation
    print("\n--- Test 7: Array Sag Calculation ---")
    x = np.linspace(-20, 20, 5)
    y = np.zeros_like(x)
    sags = sphere.sag_array(x, y)
    print("X positions:", x)
    print("Sag values:", np.round(sags, 4))
    
    # Test 8: Normal vectors
    print("\n--- Test 8: Normal Vectors on Sphere ---")
    for r in [0, 5, 10, 15]:
        z = sphere.sag(r, 0)
        n = sphere.normal(r, 0, z)
        print(f"r={r:2d}mm: normal = [{n[0]:+.4f}, {n[1]:+.4f}, {n[2]:+.4f}]")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)