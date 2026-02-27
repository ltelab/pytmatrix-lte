# PyTMatrix-LTE

A Python library for computing the scattering properties of homogeneous nonspherical scatterers with the _T_-Matrix method.\
Uses the [T-Matrix code by M. I. Mishchenko and L. D. Travis](http://www.giss.nasa.gov/staff/mmishchenko/t_matrix.html).

This is repository adapted the original PyTMatrix code by Jussi Leinonen, which can be found [here](https://github.com/jleinonen/pytmatrix), to run with newer versions of python:
installation of the original code was buggy for `python>3.6`, fully deprecated for `python>3.12` and `numpy>2`.

The code adaptations include:
- Migration from distutils to setuptools (setup.py rewritten)
- Migration of certain scipy functions to new names
- Optimization and vectorization of some computations

## Installation

The installation instructions in the original pytmatrix library are outdated and **do not** work for recent python versions (`python>3.6`).

The instructions below describe how to install the **LTE-maintained fork of pyTMatrix**, which is compatible with modern Python interpreters.

**WARNING**:  Installing pyTMatrix directly via `pip install git+https://github.com/ltelab/pytmatrix-lte.git` does not work at this time. We welcome contributions to enable this type of installation !

### 1. Install dependencies

Make sure you have the GNU Fortran Compiler (`gfortran`) and the Meson build system installed. You can install them via conda:

```bash
conda install -c conda-forge gfortran meson
```

### 2. Clone the repository

Fork and clone the LTE-maintained pyTMatrix repository:

```bash
git clone https://github.com/<your-account>/pytmatrix-lte.git
```

### 3. Install the package

Navigate into the cloned repository and install the package in editable mode:

```bash
cd pytmatrix-lte
pip install -e .
```

### 4. Run tests

To confirm that everything was installed correctly, run the built-in test suite:

```bash
pytest pytmatrix/tests
```

or from Python:

```python
import pytmatrix
pytmatrix.run_tests()
```

The software should now be installed and ready to use.

## Build artifacts

To create source and wheel distributions for publishing:

```bash
python -m build
```

If you build without isolation, ensure `ninja` and `patchelf` are available on `PATH`:

```bash
python -m build --no-isolation --skip-dependency-check
```

## Upload to (Test)PyPI

PyPI/TestPyPI reject wheels tagged `linux_x86_64`.
You must upload either:

- an `sdist` (`.tar.gz`) only, or
- a repaired Linux wheel with a `manylinux_*` tag.

Example:

```bash
# Upload source distribution only
twine upload -r testpypi dist/*.tar.gz

# Or repair wheel tag first (Linux)
python -m pip install auditwheel
auditwheel repair dist/*.whl -w dist/repaired
twine upload -r testpypi dist/repaired/*.whl dist/*.tar.gz
```

## Usage

See the [usage instructions](https://github.com/jleinonen/pytmatrix/wiki) in the original wiki.
 
