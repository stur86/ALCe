# ALCe by Simone Sturniolo
# Copyright (C) 2018 - Science and Technology Facility Council

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Functions for use of the POWDER averaging algorithm for NMR spectra,
as described in:

D. W. Alderman, Mark S. Solum, and David M. Grant
Methods for analyzing spectroscopic line shapes. NMR solid powder patterns
[J. Chern. Phys. 84 (7), 1 April 1986]

Adapted for use with ALC (integration of whole spectrum instead of single
frequency)

"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def get_orient_set(N, mode='sphere'):
    """
    Generate and return the POWDER angles (in the form of direction cosines),
    weights, and triangles.

    | Args:
    |   N (int): number of divisions on each side of the octahedron used to
    |            generate angles. Higher numbers lead to greater precision.
    |   mode (str): whether the angles should be distributed over the whole
    |               'sphere', over an 'hemisphere' or only on an 'octant'.
    |               Default is 'sphere'.

    | Returns:
    |   points, weights, tris (np.ndarray): arrays containing respectively the
    |                                       direction cosines for each
    |                                       orientation, the weights, and the
    |                                       triangles (in form of triplets of
    |                                       indices of the first array)

    """

    # Use the POWDER algorithm
    # First, generate the positive octant, by row
    points = [np.arange(N-z+1) for z in range(N+1)]
    points = np.concatenate([zip(p, N-z-p, [z]*(N-z+1))
                             for z, p in enumerate(points)])*1.0/N

    def z_i(z):
        return int(z*N+1.5*z-z**2/2.0)

    tris = np.array([[x, x+1, x+(N-z+1)]
                     for z in range(N) for x in range(z_i(z), z_i(z+1)-1)] +
                    [[x, x+(N-z), x+(N-z+1)]
                     for z in range(N) for x in range(z_i(z)+1, z_i(z+1)-1)])

    # Point weights
    dists = np.linalg.norm(points, axis=1)
    points /= dists[:, None]
    weights = dists**-3.0

    # Repeat on as many octants as needed
    if mode == 'octant':
        ranges = [[1], [1], [1]]
    elif mode == 'hemisphere':
        ranges = [[1, -1], [1, -1], [1]]
    elif mode == 'sphere':
        ranges = [[1, -1], [1, -1], [1, -1]]
    else:
        raise ValueError("Invalid mode passed to powder_alg")

    octants = np.array(np.meshgrid(*ranges)).T.reshape((-1, 3))

    points = (points[None, :, :]*octants[:, None, :]).reshape((-1, 3))
    weights = np.tile(weights, len(octants))
    tris = np.concatenate([tris]*len(octants)) + \
        np.repeat(np.arange(len(octants))*((N+2)*(N+1))/2,
                  len(tris))[:, None]

    # Some points are duplicated though. Remove them.
    ncols = points.shape[1]
    dtype = points.dtype.descr * ncols
    struct = points.view(dtype)
    uniq, uniq_i, uniq_inv = np.unique(struct, return_index=True,
                                       return_inverse=True)
    points = uniq.view(points.dtype).reshape(-1, ncols)

    # Remap triangles
    weights = weights[uniq_i]
    tris = uniq_inv[tris.astype(int)]

    # Turn points into theta-phi pairs
    ct = points[:, 2]
    st = (1-ct**2)**0.5
    stden = np.where(st != 0, st, np.inf)
    cp = points[:, 0]/stden
    sp = points[:, 1]/stden

    orients = np.array([ct, st, cp, sp]).T

    return orients, weights, tris


def tri_avg(specs, weights, tris):

    specs = np.array(specs)

    trispecs = np.sort(specs[tris], axis=1)
    triweights = np.average(weights[tris], axis=1)

    # Average integrated value with linear interpolation
    B1 = trispecs[:, 0, :]
    B2 = trispecs[:, 1, :]
    B3 = trispecs[:, 2, :]

    avgSpec = ((1.0/3.0*(B2**3-B1**3)-0.5*(B2**2*B1-B1**3))/(B2-B1) +
               0.5*(B3**2-B2**2)+(-1.0/3.0*(B3**3-B2**3) +
                                  0.5*(B3**2*B2-B2**3))/(B3-B2))*2/(B3-B1)

    return np.sum(avgSpec*triweights[:, None], axis=0)/np.sum(triweights)
