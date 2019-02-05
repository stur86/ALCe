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

# Python 2-to-3 compatibility code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import numpy as np
import argparse as ap
from collections import OrderedDict

try:
    from ase import io
except ImportError:
    print('The Atomic Simulation Environment (ASE) must be installed to run this script')
    sys.exit()

# Function to read hyperfine tensors from magres


def read_magres_hfine(mfile):
    hfine_data = OrderedDict()
    magdata = open(mfile, 'r').readlines()
    for i, l in enumerate(magdata):
        if 'Coordinates' in l:
            sym = '_'.join(l.split()[0:2])
            tens = magdata[i+4:i+7]
            tens = [[float(v) for v in r.split()] for r in tens]
            hfine_data[sym] = np.array(tens)

    return hfine_data

# Main function


def main():

    parser = ap.ArgumentParser(
        description='Produce an ALC muon spectrum from one or multiple magres files')
    parser.add_argument('files', type=str, nargs='+',
                        help='Input magres files')
    parser.add_argument('-mu', type=str, default=None,
                        help='Symbol of the muon site (default: last atom)')
    # Boolean arguments
    parser.add_argument('-ipso', type=str, action='store_true',
                        help='Use the closest hydrogen to the muon as well (ipso hydrogen)')
    parser.add_argument('-ebranch', type=str, action='store_true', help='Use the decomposition in up/down branches '
                        '(speeds up calculation but loses detail near zero field)')

    args = parser.parse_args()

    print(args.mu)

    for f in args.files:

        hfine = read_magres_hfine(f)
        syms, vals = zip(*hfine.items())
