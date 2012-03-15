'''Module that provides magnetic vector potentials.'''
import numpy as np
from math import cos, sin
cdef extern from "math.h":
    void sincosf(float x, float *sin, float *cos)
    float sqrtf(float x)
# ==============================================================================
def mvp_x( X ):
    '''Magnetic vector potential corresponding to the field B=(1,0,0).'''
    return [ 0.0, -0.5*X[2], 0.5*X[1] ]
# ==============================================================================
def mvp_y( X ):
    '''Magnetic vector potential corresponding to the field B=(0,1,0).'''
    return [ 0.5*X[2], 0.0, -0.5*X[0] ]
# ==============================================================================
def mvp_z( X ):
    '''Magnetic vector potential corresponding to the field B=(0,0,1).'''
    return [ -0.5*X[1], 0.5*X[0], 0.0 ]
# ==============================================================================
def field2potential( X, B ):
    '''Converts a constant magnetic field B at X into a corresponding potential.'''
    # This is one particular choice that works.
    return 0.5 * np.cross(B, X)
# ==============================================================================
def mvp_spherical( X, phi, theta ):
    '''Magnetic vector potential corresponding to the field
           B=( cos(theta)cos(phi), cos(theta)sin(phi), sin(theta) ),
       i.e., phi\in(0,2pi) being the azimuth, theta\in(-pi,pi) the altitude.
       The potentials parallel to the Cartesian axes can be recovered by
           mvp_x = mvp_spherical( ., 0   , 0    ),
           mvp_y = mvp_spherical( ., 0   , pi/2 ),
           mvp_z = mvp_spherical( ., pi/2, *    ).'''
    return [ -0.5 * np.sin(theta)               * X[1]
             +0.5 * np.cos(theta) * np.sin(phi) * X[2],
              0.5 * np.sin(theta)               * X[0]
             -0.5 * np.cos(theta) * np.cos(phi) * X[2],
              0.5 * np.cos(theta) * np.cos(phi) * X[1]
             -0.5 * np.cos(theta) * np.sin(phi) * X[0] ]
# ==============================================================================
def mvp_magnetic_dot(float x, float y,
                     float magnet_radius,
                     float height0,
                     float height1
                     ):
    '''Magnetic vector potential corresponding to the field
          B =
       This reprepresents the potential associated with a magnetic dot
       hovering over the domain, starting at height0, ending at height1.'''
    cdef float pi = 3.141592653589793

    # Span a cartesian grid over the sample, and integrate over it.

    # For symmetry, choose a number that is divided by 4.
    cdef int n_phi = 100
    # Choose such that the quads at radius/2 are approximately squares.
    cdef int n_radius = int( round( n_phi / pi ) )

    cdef float dr = magnet_radius / n_radius

    cdef float ax = 0.0
    cdef float ay = 0.0
    cdef float beta, rad, r, r_3D0, r_3D1, alpha, x0, y0, x_dist, y_dist
    cdef int i_phi, i_radius

    # Iterate over all all 2D 'boxes' of the magnetic dot.
    for i_phi in range(n_phi):
        beta = 2.0*pi/n_phi * i_phi
        sincosf(beta, &x0, &y0)
        for i_radius in range(n_radius):
            rad = magnet_radius / n_radius * (i_radius + 0.5)
            # r = squared distance between grid point X to the
            #     point (x,y) on the magnetic dot
            x_dist = x - rad * x0
            y_dist = y - rad * y0
            r = x_dist * x_dist + y_dist * y_dist
            if r > 1.0e-15:
                # 3D distance to point on lower edge (xi,yi,height0)
                r_3D0 = sqrtf( r + height0*height0 )
                # 3D distance to point on upper edge (xi,yi,height1)
                r_3D1 = sqrtf( r + height1*height1 )
                # Volume of circle segment = pi*r^2 * anglar_width,
                # so the volume of a building brick of the discretization is
                #   pi/n_phi * [(r+dr/2)^2 - (r-dr/2)^2]
                alpha = ( height1/r_3D1 - height0/r_3D0 ) / r \
                      * pi / n_phi * (2.0*rad*dr) # volume
                ax += y_dist * alpha
                ay -= x_dist * alpha

    return [ ax, ay, 0.0 ]
# ==============================================================================
