# PyTMatrix-LTE

`pytmatrix-lte` is a Python package to simulate how particles scatter electromagnetic waves (for example in radar applications).

This project is a maintained update of the original [jleinonen/pytmatrix](https://github.com/jleinonen/pytmatrix) so it works with modern Python versions.

## Quick Install

You need:

- Python 3.11 or newer
- `gfortran` installed on your computer

Install `gfortran` (example with conda):

```bash
conda install -c conda-forge gfortran
```

Install the package:

```bash
pip install git+https://github.com/ltelab/pytmatrix-lte.git
```

Check that it works:

```bash
python -c "import pytmatrix; print(pytmatrix.__version__)"
python -c "import pytmatrix; raise SystemExit(pytmatrix.run_tests())"
```

## Usage

See the usage wiki:
[PyTMatrix Wiki](https://github.com/jleinonen/pytmatrix/wiki)

## For Developers

If you want to contribute and edit the code locally:

```bash
micromamba install -c conda-forge gfortran meson meson-python ninja numpy scipy pytest
pip install -e . --no-build-isolation --force-reinstall
pytest
```

## Note for Python 3.13t/3.14t

If you use free-threaded Python, run commands with GIL enabled:

```bash
PYTHON_GIL=1 python -m pytest
```
