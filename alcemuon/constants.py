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
        'gamma': cnst.physical_constants['electron gyromag. '
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
