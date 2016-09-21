# -*- coding: utf-8 -*-
#
import numpy
from pyfvm.base import \
        _base_mesh, \
        _row_dot, \
        compute_tri_areas_and_ce_ratios, \
        compute_triangle_circumcenters

__all__ = ['MeshTri']


def _mirror_point(p0, p1, p2):
    '''For any given triangle and local edge

            p0
          _/  \__
        _/       \__
       /            \
      p1-------------p2

     this method creates the point p0, mirrored along the edge, and the point q
     at the perpendicular intersection of the mirror

            p0
          _/| \__
        _/  |    \__
       /    |q      \
      p1----|--------p2
       \_   |     __/
         \_ |  __/
           \| /
           mirror

    '''
    # Create the mirror.
    # q: Intersection point of old and new edge
    # q = p1 + dot(p0-p1, (p2-p1)/||p2-p1||) * (p2-p1)/||p2-p1||
    #   = p1 + dot(p0-p1, p2-p1)/dot(p2-p1, p2-p1) * (p2-p1)
    #
    if len(p0) == 0:
        return numpy.empty(p0.shape), numpy.empty(p0.shape)
    alpha = _row_dot(p0-p1, p2-p1)/_row_dot(p2-p1, p2-p1)
    q = p1 + alpha[:, None] * (p2-p1)
    # mirror = p0 + 2*(q - p0)
    mirror = 2 * q - p0
    return mirror, q


def _isosceles_ce_ratios(p0, p1, p2):
    '''Compute the _two_ covolume-edge length ratios of the isosceles
    triaingle p0, p1, p2; the edges p0---p1 and p0---p2 are assumed to be
    equally long.:
               p0
             _/ \_
           _/     \_
         _/         \_
        /             \
       p1-------------p2
    '''
    e0 = p2 - p1
    e1 = p0 - p2
    e2 = p1 - p0
    assert all(abs(_row_dot(e2, e2) - _row_dot(e1, e1)) < 1.0e-14)
    _, ce_ratios = compute_tri_areas_and_ce_ratios(e0, e1, e2)
    assert all(abs(ce_ratios[:, 1] - ce_ratios[:, 2]) < 1.0e-14)
    return ce_ratios[:, [0, 1]]


class FlatBoundaryCorrector(object):
    '''For flat elements on the boundary, a couple of things need to be
    adjusted: The covolume-edge length ratio, the control volumes
    contributions, the centroid contributions etc. Since all of these
    computations share some data, that we couldp pass around, we might as well
    do that as part of a class. Enter `FlatBoundaryCorrector`.
    '''
    def __init__(self, cells, node_coords, cell_ids, local_edge_ids):
        self.cells = cells
        self.node_coords = node_coords
        self.cell_ids = cell_ids
        self.local_edge_ids = local_edge_ids
        self.create_data()
        return

    def create_data(self):
        # In each cell, edge k is opposite of vertex k.
        self.p0_local_id = self.local_edge_ids.copy()
        self.p1_local_id = (self.local_edge_ids + 1) % 3
        self.p2_local_id = (self.local_edge_ids + 2) % 3

        self.p0_id = self.cells['nodes'][self.cell_ids, self.p0_local_id]
        self.p1_id = self.cells['nodes'][self.cell_ids, self.p1_local_id]
        self.p2_id = self.cells['nodes'][self.cell_ids, self.p2_local_id]

        self.p0 = self.node_coords[self.p0_id]
        self.p1 = self.node_coords[self.p1_id]
        self.p2 = self.node_coords[self.p2_id]

        ghost, self.q = _mirror_point(self.p0, self.p1, self.p2)

        self.ce_ratios1 = _isosceles_ce_ratios(self.p1, self.p0, ghost)
        self.ce_ratios2 = _isosceles_ce_ratios(self.p2, self.p0, ghost)

        # The ce_ratios should all be greater than 0, but due to round-off
        # errors can be slightly smaller sometimes.
        # assert (self.ce_ratios1 > 0.0).all()
        # assert (self.ce_ratios2 > 0.0).all()

        self.ghostedge_length_2 = _row_dot(
                ghost - self.p0,
                ghost - self.p0
                )
        return

    def correct_ce_ratios(self):
        '''Return the covolume-edge length ratios for the flat boundary cells.
        '''
        vals = numpy.empty((len(self.cell_ids), 3), dtype=float)
        i = numpy.arange(len(vals))
        vals[i, self.p0_local_id] = 0.0
        vals[i, self.p1_local_id] = self.ce_ratios2[:, 1]
        vals[i, self.p2_local_id] = self.ce_ratios1[:, 1]
        return self.cell_ids, vals

    def control_volumes(self):
        '''Control volume contributions

                               p0
                               _^_
                           ___//|\\___
                   e2  ___/   / | \   \___  e1
                   ___/  p0 _/  |  \_ p0  \___
               ___/   \    /    |    \    /   \___
           ___/   p1   \  /  p0 | p0  \  /  p2    \___
          /_____________\/______|______\/_____________\
         p1                                           p2
        '''
        e1 = self.p0 - self.p2
        e2 = self.p1 - self.p0

        e1_length2 = _row_dot(e1, e1)
        e2_length2 = _row_dot(e2, e2)

        ids = numpy.stack([
            numpy.c_[self.p0_id, self.p0_id],
            numpy.c_[self.p0_id, self.p1_id],
            numpy.c_[self.p0_id, self.p2_id]
            ], axis=1)
        vals = numpy.stack([
            numpy.c_[
               0.25 * self.ce_ratios1[:, 0] * self.ghostedge_length_2,
               0.25 * self.ce_ratios2[:, 0] * self.ghostedge_length_2
               ],
            numpy.c_[
                0.25 * self.ce_ratios1[:, 1] * e2_length2,
                0.25 * self.ce_ratios1[:, 1] * e2_length2
                ],
            numpy.c_[
                0.25 * self.ce_ratios2[:, 1] * e1_length2,
                0.25 * self.ce_ratios2[:, 1] * e1_length2
                ],
            ], axis=1)

        return ids, vals

    def surface_areas(self):
        '''In the triangle

                               p0
                               _^_
                           ___//|\\___
                       ___/   / | \   \___
                   ___/     _/  |  \_     \___
               ___/   \    /    |    \    /   \___
           ___/        \  /     |     \  /        \___
          /_____________\/__cv1_|_cv2__\/_____________\
         p1                     q                     p2
                             OUTSIDE

        associate the lenght dist(p1, q1) with p1, dist(q2, p2) to p2, and
        dist(q1, q2) to p0.
        '''
        ghostedge_length = numpy.sqrt(self.ghostedge_length_2)

        cv1 = self.ce_ratios1[:, 0] * ghostedge_length
        cv2 = self.ce_ratios2[:, 0] * ghostedge_length

        ids0 = numpy.c_[self.p0_id, self.p0_id]
        vals0 = numpy.c_[cv1, cv2]

        ids1 = numpy.c_[self.p1_id, self.p2_id]
        vals1 = numpy.c_[
                numpy.linalg.norm(self.q - self.p1) - cv1,
                numpy.linalg.norm(self.q - self.p2) - cv2
                ]

        ids = numpy.r_[ids0, ids1]
        vals = numpy.r_[vals0, vals1]
        return ids, vals

    def integral_x(self):
        '''Computes the integral of x,

          \int_V x,

        over all "atomic" triangles

                               p0
                               _^_
                           ___//|\\___
                       ___/   / | \   \___
                   _em2     _/  |  \_     em1_
               ___/   \    /    |    \    /   \___
           ___/        \  /     |     \  /        \___
          /_____________\/______|______\/_____________\
         p1             q1      q      q2             p2
        '''
        ids = numpy.empty((len(self.cell_ids), 3, 2), dtype=int)
        vals = numpy.empty((len(self.cell_ids), 3, 2, 3))
        # TODO vectorize
        for k, (cell_id, local_edge_id) in \
                enumerate(zip(self.cell_ids, self.local_edge_ids)):
            # The long edge is opposite of p0 and has the same local index,
            # likewise for the other edges.
            e0 = self.p2[k] - self.p1[k]
            e1 = self.p0[k] - self.p2[k]
            e2 = self.p1[k] - self.p0[k]

            # The orthogonal projection of the point q1 (and likewise q2) is
            # the midpoint em2 of the edge e2, so
            #
            #     <q1 - p1, (p0 - p1)/||p0 - p1||> = 0.5 * ||p0 - p1||.
            #
            # Setting
            #
            #     q1 = p1 + lambda1 * (p2 - p1)
            #
            # gives
            #
            #     lambda1 = 0.5 * <p0-p1, p0-p1> / <p2-p1, p0-p1>.
            #
            lambda1 = 0.5 * numpy.dot(e2, e2) / numpy.dot(e0, -e2)
            lambda2 = 0.5 * numpy.dot(e1, e1) / numpy.dot(e0, -e1)
            q1 = self.p1[k] + lambda1 * (self.p2[k] - self.p1[k])
            q2 = self.p2[k] + lambda2 * (self.p1[k] - self.p2[k])

            em1 = 0.5 * (self.p0[k] + self.p2[k])
            em2 = 0.5 * (self.p1[k] + self.p0[k])

            e1_length2 = numpy.dot(e1, e1)
            e2_length2 = numpy.dot(e2, e2)

            # triangle areas
            # TODO take from control volume contributions
            area_p0_q_q1 = \
                0.25 * self.ce_ratios1[k, 0] * self.ghostedge_length_2[k]
            area_p0_q_q2 = \
                0.25 * self.ce_ratios2[k, 0] * self.ghostedge_length_2[k]
            area_p0_q1_em2 = 0.25 * self.ce_ratios1[k, 1] * e2_length2
            area_p1_q1_em2 = area_p0_q1_em2
            area_p0_q2_em1 = 0.25 * self.ce_ratios2[k, 1] * e1_length2
            area_p2_q2_em1 = area_p0_q2_em1

            # The integral of any linear function over a triangle is the
            # average of the values of the function in each of the three
            # corners, times the area of the triangle.
            ids[k, 0] = [self.p0_id[k], self.p0_id[k]]
            ids[k, 1] = [self.p0_id[k], self.p1_id[k]]
            ids[k, 2] = [self.p0_id[k], self.p2_id[k]]

            vals[k, 0] = [
                    area_p0_q_q1 * (self.p0[k] + self.q[k] + q1) / 3.0,
                    area_p0_q_q2 * (self.p0[k] + self.q[k] + q2) / 3.0
                    ]
            vals[k, 1] = [
                    area_p0_q1_em2 * (self.p0[k] + q1 + em2) / 3.0,
                    area_p1_q1_em2 * (self.p1[k] + q1 + em2) / 3.0
                    ]
            vals[k, 2] = [
                    area_p0_q2_em1 * (self.p0[k] + em1 + q2) / 3.0,
                    area_p2_q2_em1 * (self.p2[k] + em1 + q2) / 3.0
                    ]

        return ids, vals


class MeshTri(_base_mesh):
    '''Class for handling triangular meshes.

    .. inheritance-diagram:: MeshTri
    '''
    def __init__(self, nodes, cells):
        '''Initialization.
        '''
        # Make sure to only to include those vertices which are part of a cell
        uvertices, uidx = numpy.unique(cells, return_inverse=True)
        cells = uidx.reshape(cells.shape)
        nodes = nodes[uvertices]

        super(MeshTri, self).__init__(nodes, cells)
        self.cells = numpy.empty(
                len(cells),
                dtype=numpy.dtype([('nodes', (int, 3))])
                )
        self.cells['nodes'] = cells

        self.create_edges()
        self.compute_edge_lengths()
        self.mark_default_subdomains()

        self.cell_circumcenters = compute_triangle_circumcenters(
                self.node_coords[self.cells['nodes']]
                )

        self.cell_volumes, self.ce_ratios_per_half_edge = \
            self.compute_cell_volumes_and_ce_ratios()

        # Find out which boundary cells need special treatment
        is_flat_boundary = numpy.logical_and(
            self.ce_ratios_per_half_edge < 0.0,
            self.is_boundary_edge[self.cells['edges']]
            )
        cell_ids, local_edge_ids = numpy.where(is_flat_boundary)
        # regular boundary cells
        is_regular_boundary = numpy.logical_and(
            self.ce_ratios_per_half_edge >= 0.0,
            self.is_boundary_edge[self.cells['edges']]
            )
        regular_boundary_cell_ids, regular_local_edge_ids = \
            numpy.where(is_regular_boundary)
        # All rows which are completely not flat boundary
        regular_cell_ids = numpy.where(
                numpy.all(numpy.logical_not(is_flat_boundary), axis=1)
                )[0]

        fbc = FlatBoundaryCorrector(
                self.cells, self.node_coords, cell_ids, local_edge_ids
                )
        ids, vals = fbc.correct_ce_ratios()
        self.ce_ratios_per_half_edge[ids] = vals

        # Sum up all the partial entities for convenience.
        # covolume-edgelength ratios
        self.ce_ratios = numpy.zeros(len(self.edges['nodes']), dtype=float)
        cells_edges = self.cells['edges']
        numpy.add.at(self.ce_ratios, cells_edges, self.ce_ratios_per_half_edge)

        # control_volumes
        ids, vals = self.compute_control_volumes(regular_cell_ids)
        # flat boundary elements
        fb_ids, fb_vals = fbc.control_volumes()
        # add it all up
        self.control_volumes = numpy.zeros(len(self.node_coords), dtype=float)
        numpy.add.at(
                self.control_volumes,
                numpy.r_[ids, fb_ids],
                numpy.r_[vals, fb_vals]
                )

        # surface areas
        ids0, vals0 = self.compute_surface_areas(
                regular_boundary_cell_ids,
                regular_local_edge_ids
                )
        # flat boundary correction
        ids1, vals1 = fbc.surface_areas()
        # add them all up
        self.surface_areas = numpy.zeros(len(self.node_coords))
        numpy.add.at(
                self.surface_areas,
                numpy.r_[ids0, ids1],
                numpy.r_[vals0, vals1]
                )

        # Compute the control volume centroids.
        # This is actually only necessary for special applications like Lloyd's
        # smoothing <https://en.wikipedia.org/wiki/Lloyd%27s_algorithm>.
        #
        # The centroid of any volume V is given by
        #
        #   c = \int_V x / \int_V 1.
        #
        # The denominator is the control volume. The numerator can be computed
        # by making use of the fact that the control volume around any vertex
        # v_0 is composed of right triangles, two for each adjacent cell.
        ids0, vals0 = \
            self.compute_integral_x(self.cell_circumcenters, regular_cell_ids)
        # flat boundary contributions
        ids1, vals1 = fbc.integral_x()
        # add them all up
        self.centroids = numpy.zeros((len(self.node_coords), 3))
        numpy.add.at(
            self.centroids,
            numpy.vstack([ids0, ids1]),
            numpy.vstack([vals0, vals1])
            )
        # Don't forget to divide by the control volume for the centroids
        self.centroids /= self.control_volumes[:, None]

        return

    def mark_default_subdomains(self):
        self.subdomains = {}
        self.subdomains['everywhere'] = {
                'vertices': range(len(self.node_coords)),
                'edges': range(len(self.edges['nodes']))
                }

        # Get vertices on the boundary edges
        boundary_edges = numpy.where(self.is_boundary_edge)[0]
        boundary_vertices = numpy.unique(
                self.edges['nodes'][boundary_edges].flatten()
                )

        self.subdomains['boundary'] = {
                'vertices': boundary_vertices,
                'edges': boundary_edges
                }

        return

    def create_edges(self):
        '''Setup edge-node and edge-cell relations.
        '''
        self.cells['nodes'].sort(axis=1)
        # Order the edges such that node 0 doesn't occur in edge 0 etc., i.e.,
        # node k is opposite of edge k.
        a = numpy.vstack([
            self.cells['nodes'][:, [1, 2]],
            self.cells['nodes'][:, [0, 2]],
            self.cells['nodes'][:, [0, 1]]
            ])

        # Find the unique edges
        b = numpy.ascontiguousarray(a).view(
                numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
                )
        _, idx, inv, cts = numpy.unique(
                b,
                return_index=True,
                return_inverse=True,
                return_counts=True
                )
        edge_nodes = a[idx]

        self.is_boundary_edge = (cts == 1)

        self.edges = {
            'nodes': edge_nodes,
            }

        # cell->edges relationship
        num_cells = len(self.cells['nodes'])
        cells_edges = inv.reshape([3, num_cells]).T
        cells_nodes = self.cells['nodes']
        self.cells = {
            'nodes': cells_nodes,
            'edges': cells_edges
            }

        # store inv for possible later use in create_edge_cells
        self._inv = inv

        return

    def compute_edge_cells(self):
        '''This creates edge->cell relations. As an upstream relation, this is
        relatively expensive to compute and hardly ever necessary.
        '''
        num_cells = len(self.cells['nodes'])
        edge_cells = [[] for k in range(len(self.edges['nodes']))]
        for k, edge_id in enumerate(self._inv):
            edge_cells[edge_id].append(k % num_cells)
        return edge_cells

    def compute_cell_volumes_and_ce_ratios(self):
        edge_nodes = self.edges['nodes']
        edges = \
            self.node_coords[edge_nodes[:, 1]] - \
            self.node_coords[edge_nodes[:, 0]]
        cells_edges = edges[self.cells['edges']]
        e0 = cells_edges[:, 0, :]
        e1 = cells_edges[:, 1, :]
        e2 = cells_edges[:, 2, :]
        return compute_tri_areas_and_ce_ratios(e0, e1, e2)

    def compute_control_volumes(self, cell_ids):
        edge_nodes = self.edges['nodes']

        edges = \
            self.node_coords[edge_nodes[:, 1]] - \
            self.node_coords[edge_nodes[:, 0]]
        # Compute the control volumes. Note that
        #   0.5 * (0.5 * edge_length) * covolume
        # = 0.5 * (0.5 * edge_length**2) * ce_ratio_edge_ratio
        edge_lengths_squared = _row_dot(edges, edges)
        el2 = edge_lengths_squared[self.cells['edges'][cell_ids]]

        ids = self.edges['nodes'][self.cells['edges'][cell_ids]]
        v = 0.25 * el2 * self.ce_ratios_per_half_edge[cell_ids]
        vals = numpy.stack([v, v], axis=2)

        return ids, vals

    def compute_integral_x(self, cell_circumcenters, cell_ids):
        '''Computes the integral of x,

          \int_V x,

        over all "triangles", i.e., areas enclosed by half of an edge and the
        covolume,
        '''
        # The integral of any linear function over a triangle is the average of
        # the values of the function in each of the three corners, times the
        # area of the triangle.
        edge_nodes = self.edges['nodes']
        edges = \
            self.node_coords[edge_nodes[:, 1]] - \
            self.node_coords[edge_nodes[:, 0]]
        edge_lengths_squared = _row_dot(edges, edges)

        edge_midpoints = 0.5 * (
            self.node_coords[edge_nodes[:, 1]] +
            self.node_coords[edge_nodes[:, 0]]
            )

        cells_edges = self.cells['edges'][cell_ids]

        edge_lengths_per_cell = edge_lengths_squared[cells_edges]
        right_triangle_vols = \
            0.25 * \
            edge_lengths_per_cell * \
            self.ce_ratios_per_half_edge[cell_ids]

        pt_idx = self.edges['nodes'][cells_edges]
        average = (
            cell_circumcenters[cell_ids, None, None, :] +
            edge_midpoints[cells_edges, None, :] +
            self.node_coords[pt_idx]
            ) / 3.0
        contribs = right_triangle_vols[:, :, None, None] * average

        return pt_idx, contribs

    def compute_surface_areas(self, regular_boundary_cell_ids, local_edge_ids):
        edge_ids = \
            self.cells['edges'][regular_boundary_cell_ids, local_edge_ids]
        ids = self.edges['nodes'][edge_ids]
        vals = numpy.c_[
                0.5 * self.edge_lengths[edge_ids],
                0.5 * self.edge_lengths[edge_ids]
                ]
        return ids, vals

    def compute_gradient(self, u):
        '''Computes an approximation to the gradient :math:`\\nabla u` of a
        given scalar valued function :math:`u`, defined in the node points.
        This is taken from :cite:`NME2187`,

           Discrete gradient method in solid mechanics,
           Lu, Jia and Qian, Jing and Han, Weimin,
           International Journal for Numerical Methods in Engineering,
           http://dx.doi.org/10.1002/nme.2187.
        '''
        if self.cell_circumcenters is None:
            X = self.node_coords[self.cells['nodes']]
            self.cell_circumcenters = self.compute_triangle_circumcenters(X)

        if 'cells' not in self.edges:
            self.create_edge_cells()

        # This only works for flat meshes.
        assert (abs(self.node_coords[:, 2]) < 1.0e-10).all()
        node_coords2d = self.node_coords[:, :2]
        cell_circumcenters2d = self.cell_circumcenters[:, :2]

        num_nodes = len(node_coords2d)
        assert len(u) == num_nodes

        gradient = numpy.zeros((num_nodes, 2), dtype=u.dtype)

        # Create an empty 2x2 matrix for the boundary nodes to hold the
        # edge correction ((17) in [1]).
        boundary_matrices = {}
        for node in self.get_vertices('boundary'):
            boundary_matrices[node] = numpy.zeros((2, 2))

        for edge_id, edge in enumerate(self.edges['cells']):
            # Compute edge length.
            node0 = self.edges['nodes'][edge_id][0]
            node1 = self.edges['nodes'][edge_id][1]

            # Compute coedge length.
            if len(self.edges['cells'][edge_id]) == 1:
                # Boundary edge.
                edge_midpoint = 0.5 * (
                        node_coords2d[node0] +
                        node_coords2d[node1]
                        )
                cell0 = self.edges['cells'][edge_id][0]
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[cell0] +
                        edge_midpoint
                        )
            elif len(self.edges['cells'][edge_id]) == 2:
                cell0 = self.edges['cells'][edge_id][0]
                cell1 = self.edges['cells'][edge_id][1]
                # Interior edge.
                coedge_midpoint = 0.5 * (
                        cell_circumcenters2d[cell0] +
                        cell_circumcenters2d[cell1]
                        )
            else:
                raise RuntimeError(
                        'Edge needs to have either one or two neighbors.'
                        )

            # Compute the coefficient r for both contributions
            coeffs = self.ce_ratios[edge_id] / \
                self.control_volumes[self.edges['nodes'][edge_id]]

            # Compute R*_{IJ} ((11) in [1]).
            r0 = (coedge_midpoint - node_coords2d[node0]) * coeffs[0]
            r1 = (coedge_midpoint - node_coords2d[node1]) * coeffs[1]

            diff = u[node1] - u[node0]

            gradient[node0] += r0 * diff
            gradient[node1] -= r1 * diff

            # Store the boundary correction matrices.
            edge_coords = node_coords2d[node1] - node_coords2d[node0]
            if node0 in boundary_matrices:
                boundary_matrices[node0] += numpy.outer(r0, edge_coords)
            if node1 in boundary_matrices:
                boundary_matrices[node1] += numpy.outer(r1, -edge_coords)

        # Apply corrections to the gradients on the boundary.
        for k, value in boundary_matrices.items():
            gradient[k] = numpy.linalg.solve(value, gradient[k])

        return gradient

    def compute_curl(self, vector_field):
        '''Computes the curl of a vector field. While the vector field is
        point-based, the curl will be cell-based. The approximation is based on

        .. math::
            n\cdot curl(F) = \lim_{A\\to 0} |A|^{-1} \int_{dGamma} F dr;

        see <https://en.wikipedia.org/wiki/Curl_(mathematics)>. Actually, to
        approximate the integral, one would only need the projection of the
        vector field onto the edges at the midpoint of the edges.
        '''
        edge_coords = \
            self.node_coords[self.edges['nodes'][:, 1]] - \
            self.node_coords[self.edges['nodes'][:, 0]]

        barycenters = 1./3. * numpy.sum(
                self.node_coords[self.cells['nodes']],
                axis=1
                )

        # Compute the projection of A on the edge at each edge midpoint.
        nodes = self.edges['nodes']
        x = self.node_coords[nodes]
        A = 0.5 * numpy.sum(vector_field[nodes], axis=1)
        edge_dot_A = _row_dot(edge_coords, A)

        directions = numpy.cross(
                x[self.cells['edges'], 0] - barycenters[:, None, :],
                x[self.cells['edges'], 1] - barycenters[:, None, :]
                )
        dir_nrms = numpy.sqrt(numpy.sum(directions**2, axis=2))
        directions /= dir_nrms[..., None]

        # a: directions scaled with edge_dot_a
        a = directions * edge_dot_A[self.cells['edges']][..., None]

        # sum over all local edges
        curl = numpy.sum(a, axis=1)
        # Divide by cell volumes
        curl /= self.cell_volumes[..., None]
        return curl

    def num_delaunay_violations(self):
        # Delaunay violations are present exactly on the interior edges where
        # the ce_ratio is negative. Count those.
        return numpy.sum(self.ce_ratios[~self.is_boundary_edge] < 0.0)

    def show(self, show_ce_ratios=True):
        '''Show the mesh using matplotlib.

        :param show_ce_ratios: If true, show all ce_ratios of the mesh, too.
        :type show_ce_ratios: bool, optional
        '''
        # Importing matplotlib takes a while, so don't do that at the header.
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib as mpl
            # headless mode, for remote executions (and travis)
            mpl.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.collections import LineCollection

        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.gca()
        plt.axis('equal')

        xmin = numpy.amin(self.node_coords[:, 0])
        xmax = numpy.amax(self.node_coords[:, 0])
        ymin = numpy.amin(self.node_coords[:, 1])
        ymax = numpy.amax(self.node_coords[:, 1])

        width = xmax - xmin
        xmin -= 0.1 * width
        xmax += 0.1 * width

        height = ymax - ymin
        ymin -= 0.1 * height
        ymax += 0.1 * height

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Get edges, cut off z-component.
        e = self.node_coords[self.edges['nodes']][:, :, :2]
        # Plot regular edges black, those with negative ce-ratio red.
        pos = self.ce_ratios >= 0
        line_segments0 = LineCollection(e[pos], color='k')
        ax.add_collection(line_segments0)
        #
        neg = numpy.logical_not(pos)
        line_segments1 = LineCollection(e[neg], color='r')
        ax.add_collection(line_segments1)

        if show_ce_ratios:
            # Connect all cell circumcenters with the edge midpoints
            if self.cell_circumcenters is None:
                X = self.node_coords[self.cells['nodes']]
                self.cell_circumcenters = \
                    self.compute_triangle_circumcenters(X)

            edge_midpoints = 0.5 * (
                self.node_coords[self.edges['nodes'][:, 0]] +
                self.node_coords[self.edges['nodes'][:, 1]]
                )

            # Plot connection of the circumcenter to the midpoint of all three
            # axes.
            a = numpy.stack([
                    self.cell_circumcenters[:, :2],
                    edge_midpoints[self.cells['edges'][:, 0], :2]
                    ], axis=1)
            b = numpy.stack([
                    self.cell_circumcenters[:, :2],
                    edge_midpoints[self.cells['edges'][:, 1], :2]
                    ], axis=1)
            c = numpy.stack([
                    self.cell_circumcenters[:, :2],
                    edge_midpoints[self.cells['edges'][:, 2], :2]
                    ], axis=1)

            line_segments = LineCollection(
                numpy.r_[a, b, c],
                color=[0.8, 0.8, 0.8]
                )
            ax.add_collection(line_segments)

        # plot centroids
        ax.plot(self.centroids[:, 0], self.centroids[:, 1], 'r.')

        return

    def show_vertex(self, node_id, show_ce_ratio=True):
        '''Plot the vicinity of a node and its ce_ratio.

        :param node_id: Node ID of the node to be shown.
        :type node_id: int

        :param show_ce_ratio: If true, shows the ce_ratio of the node, too.
        :type show_ce_ratio: bool, optional
        '''
        # Importing matplotlib takes a while, so don't do that at the header.
        import os
        if 'DISPLAY' not in os.environ:
            import matplotlib as mpl
            # headless mode, for remote executions (and travis)
            mpl.use('Agg')
        from matplotlib import pyplot as plt

        fig = plt.figure()
        ax = fig.gca()
        plt.axis('equal')

        # Find the edges that contain the vertex
        edge_ids = numpy.where((self.edges['nodes'] == node_id).any(axis=1))[0]
        # ... and plot them
        for node_ids in self.edges['nodes'][edge_ids]:
            x = self.node_coords[node_ids]
            ax.plot(x[:, 0], x[:, 1], 'k')

        # Highlight ce_ratios.
        if show_ce_ratio:
            if self.cell_circumcenters is None:
                X = self.node_coords[self.cells['nodes']]
                self.cell_circumcenters = \
                    self.compute_triangle_circumcenters(X)

            # Find the cells that contain the vertex
            cell_ids = numpy.where(
                (self.cells['nodes'] == node_id).any(axis=1)
                )[0]

            for cell_id in cell_ids:
                for edge_id in self.cells['edges'][cell_id]:
                    if node_id not in self.edges['nodes'][edge_id]:
                        continue
                    node_ids = self.edges['nodes'][edge_id]
                    edge_midpoint = 0.5 * (
                            self.node_coords[node_ids[0]] +
                            self.node_coords[node_ids[1]]
                            )
                    p = numpy.c_[
                        self.cell_circumcenters[cell_id],
                        edge_midpoint
                        ]
                    q = numpy.c_[
                        self.cell_circumcenters[cell_id],
                        edge_midpoint,
                        self.node_coords[node_id]
                        ]
                    ax.fill(q[0], q[1], color='0.5')
                    ax.plot(p[0], p[1], color='0.7')
        return
