# sktime-shapelet-forest

This repo is an adaptation of [Shapelet Forest (SF)](https://github.com/isakkarlsson/wildboar), 
which is a Python module for temporal machine learning and fast
distance computations built on top of
[Scikit-Learn](http://scikit-learn.org) and [Numpy](http://numpy.org)
distributed under the GNU General Public License Version 3.

The original repo is currently maintained by Isak Karlsson, while this one is maintained by David Guijo-Rubio.

## Installation

### Dependencies

SF requires:

 * Python (>= 3.4)
 * NumPy (>= 1.8.2)
 * SciPy (>= 0.13.3)
 * sktime (>= 0.3.0)
 
Some parts of SF is implemented using Cython. Hence, compilation
requires:

 * Cython (>= 0.28)

### Compilation

If you already have a working installation of NumPy, SciPy and Cython,
compiling and installing SF is as simple as:

    python setup.py install
	
To install the requirements, use:

    pip install -r requirements.txt
	
## Development

Contributions are welcome. Pull requests are encouraged to be
formatted according to
[PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using
[yapf](https://github.com/google/yapf).

## Usage

See the examples folder, containing two examples:
 * [Univariate time series]()
 * [Multivariate time series]()
	
## Citation

If you use SF in a scientific publication, please cite 
the original paper: Karlsson, I., Papapetrou, P. Boström, H.,
*Generalized Random Shapelet Forests*. In the Data Mining and
Knowledge Discovery Journal (DAMI), 2016

## Contributors

Former and current active contributors are as follows.

Project management: Jason Lines (@jasonlines), Franz Király (@fkiraly)

Design: Anthony Bagnall(@TonyBagnall), Sajaysurya Ganesh (@sajaysurya), Jason Lines (@jasonlines), Viktor Kazakov (@viktorkaz), Franz Király (@fkiraly), Markus Löning (@mloning)

Coding: Sajaysurya Ganesh (@sajaysurya), Bagnall(@TonyBagnall), Jason Lines (@jasonlines), George Oastler (@goastler), Viktor Kazakov (@viktorkaz), Markus Löning (@mloning)

We are actively looking for contributors. Please contact @fkiraly or @jasonlines for volunteering or information on paid opportunities, or simply raise an issue in the tracker.