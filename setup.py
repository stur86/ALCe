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
# The next line is removed because it causes issues in interpreting
# the package_data line, unfortunately
# from __future__ import unicode_literals

from setuptools import setup, find_packages
from alcemuon import __version__

long_description = """
ALCe is a Python library developed and maintained by Simone Sturniolo with the
purpose of calculating Avoided Level Crossing spectra in muon spectroscopy.

It makes use of the 'integral operator' algorithm described in this paper:
https://arxiv.org/abs/1704.02785
"""

if __name__ == '__main__':
    setup(name='ALCe',
          version=__version__,
          description='A Python library to compute ALC muon spectra',
          long_description=long_description,
          author='Simone Sturniolo',
          author_email='simone.sturniolo@stfc.ac.uk',
          license='MIT',
          classifiers=[
              # How mature is this project? Common values are
              #   3 - Alpha
              #   4 - Beta
              #   5 - Production/Stable
              'Development Status :: 3 - Alpha',

              # Indicate who your project is intended for
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Chemistry',
              'Topic :: Scientific/Engineering :: Physics',

              # Pick your license as you wish (should match "license" above)
              'License :: OSI Approved :: MIT License',

              # Specify the Python versions you support here. In particular,
              # ensure that you indicate whether you support Python 2,
              # Python 3 or both.
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
          ],
          keywords='crystallography ccpnc computational muon spectroscopy',
          packages=find_packages(),
          # For data files. Example: 'soprano': ['data/*.json']
          package_data={'alcemuon': ['data/*.json', 'data/*.dat']},
          entry_points={
              'console_scripts': ['magres2alc = '
                                  'alcemuon.scripts.magres2alc:main']
          },          
          # Requirements
          install_requires=[
              'numpy',
              'scipy'
          ],
          python_requires='>=2.7'
          )
