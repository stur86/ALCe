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

"""Module containing utility functions"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from alcemuon.constants import MUON_TAU


def multikron(*m):
    # Recursive Kronecker product for multiple matrices
    if len(m) == 1:
        return m[0]
    elif len(m) == 2:
        return np.kron(*m)
    else:
        return np.kron(m[0], multikron(*m[1:]))


def basis_transform(op, evecs, reverse=False):
    # Operator basis transform. If evecs are eigenvectors of op, this will
    # diagonalise op. If reverse is set to True, it works the other way.
    if not reverse:
        return np.dot(evecs.conj().T, np.dot(op, evecs))
    else:
        return np.dot(evecs, np.dot(op, evecs.conj().T))


def make_rotation_matrix(ct, st, cp, sp):
    # Create rotation matrix given cosines and sines of direction angles
    # (ct = cos(theta), cp = cos(phi), st = sin(theta), sp = sin(phi))
    return np.array([[cp*ct,  sp,  cp*st],
                     [-sp*ct, cp, -sp*st],
                     [-st,     0,  ct]])


def decay_intop(rho0, evals, evecs, tau_times, Sz):
    # Compute decay with approximated integral operator
    Dz = basis_transform(Sz, evecs)
    rho = basis_transform(rho0, evecs)

    lDiff = evals[:, None]-evals[None, :]
    if tau_times is not None:
        t = MUON_TAU*tau_times
        exp_re = np.exp(-tau_times)
        exp_im = np.exp(1.0j*lDiff*t)
    else:
        exp_re = 0.0
        exp_im = 0.0

    Oz = (Dz/((1.0j*lDiff)-1.0/MUON_TAU)*(
        (exp_re*exp_im-1)/(
            MUON_TAU*(1-exp_re))))

    return np.real(np.dot(rho, Oz).trace())


def split_hamiltonian(H):
    # Produce eigenvalues/vectors for the two manifolds (up & down) of H
    msize = int(H.shape[0]/2)
    H2 = np.dot(H, H)
    H2_u, H2_d = (H2[:msize, :msize],
                  H2[msize:, msize:])
    evals2_u, evecs_u = np.linalg.eigh(H2_u)
    evals2_d, evecs_d = np.linalg.eigh(H2_d)

    return evals2_u**0.5, evecs_u, evals2_d**0.5, evecs_d
