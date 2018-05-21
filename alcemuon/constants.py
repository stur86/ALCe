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

"""Module containing important constants"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy import constants as cnst
try:
    from soprano.properties.nmr.utils import _get_isotope_data
except ImportError:
    _get_isotope_data = None

# Basic NMR data
# All gammas are in MHz/T
_nmr_basic = {
    'e': {
        'I': 0.5,
        'Q': 0,
        'gamma': -cnst.physical_constants['electron gyromag. '
                                          'ratio over 2 pi'][0],
    },
    'mu': {
        'I': 0.5,
        'Q': 0,
        'gamma': (cnst.physical_constants['electron gyromag. '
                                          'ratio over 2 pi'][0] /
                  cnst.physical_constants['electron-muon mag. '
                                          'mom. ratio'][0])
    },
    'H': {  # Data only for 1H, proton
        'I': 0.5,
        'Q': 0,
        'gamma': cnst.physical_constants['proton gyromag. ratio over 2 pi'][0]
    }
}


def magnetic_constant(elem, value='gamma', iso=None):

    if elem in ('e', 'mu') or (elem == 'H' and iso in (None, 1)):
        return _nmr_basic[elem][value]
    elif _get_isotope_data is not None:
        val = _get_isotope_data([elem], value, isotope_list=[iso])[0]
        return val / (2*np.pi*1e6 if value == 'gamma' else 1.0)
    else:
        raise RuntimeError('A valid installation of Soprano is necessary '
                           'for magnetic data of elements other than 1H')


# Muon decay rate, microseconds
MUON_TAU = 2.196
# EFG to MHz constant for Quadrupole couplings
# (the total quadrupole coupling in MHz is QCONST*Q*Vzz)
QCONST = cnst.physical_constants['atomic unit of electric field '
                                 'gradient'][0]*cnst.e*1e-37/cnst.h

# Spin one half operators (for convenience)
_spin_half_ops = np.array([
    [[0, 1],
     [1, 0]],
    [[0, -1.0j],
     [1.0j, 0]],
    [[1, 0],
     [0, -1]]
])*0.5


def spin_operators(I=0.5):
    # Return a numpy array containing the three operators Sx, Sy, Sz for a
    # given spin of magnitude I, of size (3, 2*I+1, 2*I+1)

    if I % 0.5 or I < 0.5:
        raise ValueError('{0} is not a valid spin value'.format(I))

    if I == 0.5:
        return _spin_half_ops.copy()

    moms = np.linspace(I, -I, int(2*I+1))

    Sz = np.diag(moms)

    Sp = np.diag(0.5*(np.cumsum(2*moms)[:-1]**0.5), k=1)
    Sm = Sp.T

    Sx = Sp+Sm
    Sy = 1.0j*(-Sp+Sm)

    return np.array([Sx, Sy, Sz])
