'''Module that provides magnetic vector potentials.'''
import math
# ==============================================================================
def mvp_x( X ):
    '''Magnetic vector potential corresponding to the field B=(1,0,0).'''
    return [ 0.0, -0.5*X[2], 0.5*X[1] ]
# ==============================================================================
def mvp_y( X ):
    '''Magnetic vector potential corresponding to the field B=(0,1,0).'''
    return [ 0.5*X[2], 0, -0.5*X[0] ]
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
    return [ -0.5 * math.sin(theta)                 * X[1]
             +0.5 * math.cos(theta) * math.sin(phi) * X[2],
              0.5 * math.sin(theta)                 * X[0]
             -0.5 * math.cos(theta) * math.cos(phi) * X[2],
              0.5 * math.cos(theta) * math.cos(phi) * X[1]
             -0.5 * math.cos(theta) * math.sin(phi) * X[0] ]
# ==============================================================================
def mvp_magnetic_dot( X, magnet_radius, height0, height1 ):
    '''Magnetic vector potential corresponding to the field
          B =
       This reprepresents the potential associated with a magnetic dot
       hovering over the domain, starting at height0, ending at height1.'''

    # Span a cartesian grid over the sample, and integrate over it.

    # For symmetry, choose a number that is divided by 4.
    n_phi = 100
    # Choose such that the quads at radius/2 are approximately squares.
    n_radius = int( round( n_phi / math.pi ) )

    dr = magnet_radius / n_radius

    ax = 0.0
    ay = 0.0
    # Iterate over all all 2D 'boxes' of the magnetic dot.
    for i_phi in xrange( 0, n_phi ):
        x0 = math.cos( 2.0*math.pi/n_phi * i_phi  )
        y0 = math.sin( 2.0*math.pi/n_phi * i_phi  )
        for i_radius in xrange( 0, n_radius ):
            rad = magnet_radius / n_radius * (i_radius + 0.5)
            x = rad * x0
            y = rad * y0
            # r = squared distance between grid point X to the
            #     point (x,y) on the magnetic dot
            x_dist = X[0] - x
            y_dist = X[1] - y
            r = x_dist*x_dist + y_dist*y_dist
            if  r > 1.0e-15:
                # 3D distance to point on lower edge (xi,yi,height0)
                r_3D0 = math.sqrt( r + height0*height0 )
                # 3D distance to point on upper edge (xi,yi,height1)
                r_3D1 = math.sqrt( r + height1*height1 )
                # Volume of circle segment = pi*r^2 * anglar_width,
                # so the volume of a building brick of the discretization is
                #   pi/n_phi * [(r+dr/2)^2 - (r-dr/2)^2]
                vol = math.pi / n_phi * (2.0*rad*dr)
                alpha = ( height1/r_3D1 - height0/r_3D0 ) / r * vol
                ax = ax + y_dist * alpha
                ay = ay - x_dist * alpha

    return [ ax, ay, 0.0 ]
# ==============================================================================
