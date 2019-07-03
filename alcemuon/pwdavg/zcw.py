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

"""Module containing functions to generate ZCW powder averaging
orientations"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def get_orient_set(lower_N, mode='sphere'):

    # Find the correct g
    zcw_g = [8, 13]

    try:
        zcw_c = {
            'sphere': (1, 2, 1),
            'hemisphere': (-1, 1, 1),
            'octant': (2, 1, 8)}[mode]
    except KeyError:
        raise ValueError("Invalid mode value passed to ZCWgen")

    # Starting M & N
    zcw_M = 2
    zcw_N = 21
    while zcw_N < lower_N:
        zcw_g.append(zcw_N)
        zcw_M += 1
        zcw_N = zcw_g[-1]+zcw_g[-2]

    # If it's over
    zcw_Nf = 1.0*zcw_N
    zcw_g = zcw_g[-1]

    n = np.arange(0, zcw_N)

    phi = 2.0*np.pi/zcw_c[2]*np.mod(n*zcw_g/zcw_Nf, 1.0)
    cp = np.cos(phi)
    sp = np.sin(phi)
    ct = zcw_c[0]*(zcw_c[1]*np.mod(n/zcw_Nf, 1.0)-1.0)
    st = (1.0-ct**2)**0.5

    return np.array(list(zip(ct, st, cp, sp))), np.ones(len(phi))/len(phi)
