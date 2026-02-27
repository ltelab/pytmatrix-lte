# Changelog

## Version 0.0.4 - Meson Era

### Enhancements

- pytmatrix can be installed for python versions >= 3.11 using meson.
- Set up CI pipelines for testing and code quality checks.
- Set up precommit hooks for code formatting and linting.
- Improve documentation and add docstrings to all public functions and classes.
- Refactored tests to use pytest

### Changes

- Remove deprecated TMatrixPSD class and related code.
- Remove deprecated alias axi, lam, eps, rat, np and scatter from Scatterer class.
- Speed up computations in various places with vectorized operations
