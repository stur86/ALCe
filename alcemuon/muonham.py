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

"""Muon Hamiltonian class"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import numpy as np
import scipy.constants as cnst
from alcemuon.utils import (multikron, make_rotation_matrix, split_hamiltonian,
                            decay_intop)
from alcemuon.constants import spin_operators, magnetic_constant, QCONST


def _make_spin_arrays(spins, isotopes):
    _spins = spins
    if isotopes is None:
        isotopes = [None]*len(spins)
    else:
        if len(isotopes) != len(spins):
            raise ValueError('Invalid isotope list')
    _isos = isotopes
    _gammas = np.array([magnetic_constant(s, iso=isotopes[i])
                        for i, s in enumerate(spins)])
    _Is = np.array([magnetic_constant(s, iso=isotopes[i],
                                      value='I')
                    for i, s in enumerate(spins)])
    _Qs = np.array([magnetic_constant(s, iso=isotopes[i],
                                      value='Q')
                    for i, s in enumerate(spins)])
    _ops = [spin_operators(_Is[i])
            for i, s in enumerate(spins)]

    return _spins, _isos, _gammas, _Is, _Qs, _ops


class HamiltonianError(Exception):
    pass


class MuonHamiltonian(object):

    def __init__(self, other_spins=[], isotopes=[], use_electron=True,
                 unsafe=False):

        if len(isotopes) == 0 and len(other_spins) > 0:
            isotopes = [None]*len(other_spins)
        elif len(isotopes) != len(other_spins):
            raise ValueError('Invalid isotope list (should be as long as '
                             'spin list)')

        if use_electron:
            spins = ['e', 'mu'] + other_spins
            isotopes = [None]*2 + isotopes
        else:
            spins = ['mu'] + other_spins
            isotopes = [None] + isotopes

        self._has_e = use_electron

        _spins, _isos, _gammas, _Is, _Qs, _ops = _make_spin_arrays(spins,
                                                                   isotopes)

        self._spins = _spins
        self._isos = _isos
        self._gammas = _gammas
        self._Is = _Is
        self._Qs = _Qs
        self._spin_ops = _ops

        # First, projected size
        size = np.prod(2*_Is+1)
        if size > 1000 and not unsafe:
            raise RuntimeError('System is too big for safe simulation - '
                               'set unsafe=True to proceed anyway')

        # Build full operators
        self._full_ops = np.array([[multikron(*[o[n] if i == j else
                                                np.identity(
            int(2*_Is[j]+1))
            for j, o in
            enumerate(_ops)])
            for i in range(len(_spins))]
            for n in range(3)])

        # Build Zeeman Hamiltonian
        self._Hz = -np.sum(self._gammas[:, None, None]*self._full_ops[2],
                           axis=0)

        # Couplings
        self._ctens = {}

        # Initialise *unbuilt* Hamiltonian (need orientations first)
        self._orient = None
        self._Hc = None
        self._H = None

    @property
    def spins(self):
        return self._spins

    @property
    def spin_table(self):
        return dict(zip(self._spins, range(len(self._spins))))

    def get_spin_index(self, spinsym):
        spre = re.compile('([A-Za-z]{1,2})(:[0-9]+)?')
        m = spre.match(spinsym)
        if m is None:
            raise ValueError('Invalid spin symbol')

        s_el = m.groups()[0]
        try:
            s_i = int(m.groups()[1][1:])
        except TypeError:
            s_i = 0

        if s_el not in self._spins:
            raise ValueError('Spin not contained in system')

        inds = np.where(s_el == np.array(self._spins))[0]

        try:
            return inds[s_i]
        except IndexError:
            raise ValueError('Spin index too great for system')

    def add_hyperfine_coupling(self, A, i):
        # Add hyperfine coupling for spin i with tensor A

        if not self._has_e:
            raise HamiltonianError('Muon Hamiltonian with no electron can not'
                                   ' include hyperfine couplings')

        # Is it a spin symbol?
        try:
            s_i = self.get_spin_index(i)
            i = s_i
        except TypeError:
            pass

        A = np.array(A)

        if A.shape != (3, 3):
            raise ValueError('Invalid hyperfine tensor')

        self._ctens[(0, i)] = A + self._ctens.get((0, i), np.zeros((3, 3)))

    def add_quadrupolar_coupling(self, EFG, i):

        # Is it a spin symbol?
        try:
            s_i = self.get_spin_index(i)
            i = s_i
        except TypeError:
            pass

        EFG = np.array(EFG)

        if EFG.shape != (3, 3):
            raise ValueError('Invalid electric field gradient tensor')

        Q = QCONST*self._Qs[i]/(2*self._Is[i]*(2*self._Is[i]-1))*EFG

        self._ctens[(i, i)] = Q + self._ctens.get((i, i), np.zeros((3, 3)))

    def add_dipolar_coupling(self, r, i, j):
        # Add dipolar coupling between spins i and j connected by vector r
        # Distances expected in Angstroms

        # Is it a spin symbol?
        try:
            s_i = self.get_spin_index(i)
            i = s_i
        except TypeError:
            pass

        # Is it a spin symbol?
        try:
            s_j = self.get_spin_index(j)
            j = s_j
        except TypeError:
            pass

        r = np.array(r)

        if r.shape != (3,):
            raise ValueError('Invalid connecting vector')

        rnorm = np.linalg.norm(r)
        D = -(np.eye(3) - 3.0/rnorm**2.0*r[:, None]*r[None, :])
        dij = (- (cnst.mu_0*cnst.hbar*(self._gammas[i]*self._gammas[j]*1e12)) /
               (2*(rnorm*1e-10)**3))*1e-6 # MHz
        D *= dij

        self._ctens[(i, j)] = D + self._ctens.get((i, j), np.zeros((3, 3)))

    def _build_Hc(self, ct, st, cp, sp):

        # Build Hamiltonian with given orientation angles
        rotm = make_rotation_matrix(ct, st, cp, sp)
        rotc = {k: np.einsum('ji,jk,kl', rotm, c, rotm)
                for k, c in self._ctens.iteritems()}

        _Hc = self._Hz*0.0

        for (i, j), c in rotc.iteritems():
            _Hc += np.einsum('pjl,pq,qli',
                             self._full_ops[:, i, :, :],
                             c,
                             self._full_ops[:, j, :, :])

        return _Hc

    def ALC(self, B_range, orients, weights, tau_times=None,
            state={'mu': [1, 0]}, verbose=False, units='MHz',
            unsafe=False, split_e=False):
            # Create an ALC spectrum

        if not self._has_e:
            # No point in splitting anything...
            split_e = False

        unitconv = {
            'T': 1.0,
            'MHz': 0.5/magnetic_constant('mu')
        }

        B_range = np.array(B_range)
        orients = np.array(orients)
        weights = np.array(weights)

        if len(orients) != len(weights):
            raise ValueError('Inconsistent orientations and weights arrays')

        # Now do the average
        specs = np.zeros((len(orients), len(B_range)))

        # Build psi0 from states
        psi0 = [state.get(s, np.ones(int(2*self._Is[i]+1)))
                for i, s in enumerate(self._spins)]
        if split_e:
            psi0 = psi0[1:]
        psi0 = multikron(*psi0)
        psi0 /= np.linalg.norm(psi0)
        rho0 = psi0[:, None]*psi0[None, :].conj()  # Density matrix

        if split_e:
            muonSz = multikron(*[sop[2] for sop in self._spin_ops[1:]])
        else:
            muonSz = self._full_ops[2, 0+self._has_e]

        for i, ((ct, st, cp, sp), w) in enumerate(zip(orients, weights)):
            if verbose:
                sys.stdout.write(
                    '\rOrientational average: '
                    '{0: 6.1f}%'.format((i+1.0) * 100.0/len(weights)))
            # Get the hyperfine component
            _Hc = self._build_Hc(ct, st, cp, sp)
            for j, B in enumerate(B_range):
                # Total Hamiltonian?
                H_tot = self._Hz*B*unitconv[units]+_Hc
                manifolds = []
                if split_e:
                    eu, evu, ed, evd = split_hamiltonian(H_tot)
                    manifolds = [[eu, evu], [ed, evd]]
                else:
                    manifolds = [np.linalg.eigh(H_tot)]

                for (evals, evecs) in manifolds:
                    specs[i, j] += decay_intop(rho0, evals, evecs,
                                               tau_times, muonSz)
                specs[i, j] /= len(manifolds)

        # Orientational average
        full_spec = np.sum(specs*weights[:, None], axis=0)/np.sum(weights)

        return full_spec
