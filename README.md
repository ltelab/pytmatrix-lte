# PyTMatrix-LTE

A Python library for computing the scattering properties of homogeneous nonspherical scatterers with the _T_-Matrix method.\
Uses the [T-Matrix code by M. I. Mishchenko and L. D. Travis](http://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html).

This is repository adapted the original PyTMatrix code by Jussi Leinonen, which can be found [here](https://github.com/jleinonen/pytmatrix), to run with newer versions of python >=3.11 and numpy >=2. The original code was last updated in 2018, and the installation of the original code was buggy for `python>3.6`, fully deprecated for `python>3.12` and `numpy>2`.

The code adaptations include:

- Migration to pyproject.toml and meson build system
- Update of scipy functions to new names
- Optimization and vectorization of some computations

## Usage

See the [usage instructions](https://github.com/jleinonen/pytmatrix/wiki) in the original wiki.

## Installation for users

The installation instructions in the original pytmatrix library are outdated and **do not** work for recent python versions (`python>=3.11`).

The instructions below describe how to install the **LTE-maintained fork of pyTMatrix**, which is compatible with modern Python interpreters.

### 1. Install dependencies

Make sure you have the GNU Fortran Compiler (`gfortran`) and the Meson build system installed. You can install them via conda:

```bash
conda install -c conda-forge gfortran meson meson-python ninja
```

### 2. Install pytmatrix-lte

With the dependencies installed, you can install the package from the GitHub repository:

```bash
pip install git+https://github.com/ltelab/pytmatrix-lte.git
```

If you use a free-threaded CPython build (`python3.13t`/`python3.14t`), run with
the GIL enabled for safety (`PYTHON_GIL=1` or `python -X gil=1`).

### 4. Run tests

To confirm that everything was installed correctly, run the test suite from the command line:

```bash
python -c "import pytmatrix; pytmatrix.run_tests()"
```

or in python:

```python
import pytmatrix

pytmatrix.run_tests()
```

The software should now be installed and ready to use.

## Installation for developers

If you want to install the package for development, you can clone the repository and install it in editable mode:

```bash
micromamba install gfortran meson meson-python ninja numpy scipy pytest
pip uninstall -y pytmatrix
pip uninstall -y pytmatrix # just to be sure
pip install -e . --no-build-isolation --force-reinstall
pytest
```
