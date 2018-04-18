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

"""Module containing functions to generate Lebedev powder averaging
orientations"""

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import pkgutil
import numpy as np

# Load all orientation sets

def _load_data():

    data = pkgutil.get_data('alcemuon',
                                 'data/OurLebedevSets.dat').decode('utf-8')
    # Split and sanitize
    data = [l for l in data.split('\n') if l.strip() != '']

    headre = re.compile('\s*M\s*=\s*([0-9]+)\s+rmax\s*=\s*([0-9.e+-]+)')
    group = None

    OurLebSets = {}

    for l in data:
        m = headre.match(l)
        if m is not None:
            if group is not None:
                OurLebSets[group][0] = np.array(OurLebSets[group][0])
                OurLebSets[group][1] = np.array(OurLebSets[group][1])
            group, rmax = m.groups()
            group = int(group)
            OurLebSets[group] = [[], []]
        elif group is not None:
            _, th, phi, w = [float(x) for x in l.split()]
            th *= np.pi/180.0
            phi *= np.pi/180.0
            OurLebSets[group][0].append([np.cos(th), np.sin(th),
                                         np.cos(phi), np.sin(phi)])
            OurLebSets[group][1].append(w)

    return OurLebSets

_OurLebSets = _load_data()

def get_orient_set(lower_N):
    """
    Returns the Lebedev orientation set with the number of angles closest to
    N (by excess).
    
    The sets are taken from the Southampton University webpage, specifically
    Malcolm Levitt's group:

    http://www.southampton.ac.uk/~mhl/resources/Orientations/index.html

    and have been created by Marina Carravetta.
    """
    
    keys = _OurLebSets.keys()
    diffs = np.array(keys) - lower_N
    return _OurLebSets[sorted(diffs[np.where(diffs >= 0)] + lower_N)[0]]