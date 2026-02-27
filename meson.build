# NumPy’s Meson guidance for f2py is: generate wrappers with custom_target(), then build the extension with py.extension_module()
# py.extension_module() place the compiled .so/.pyd inside pytmatrix/fortran_tm
# meson-python supports editable installs by building into a build/<abi>/ directory and then exposing it to Python

project(
  'pytmatrix',
  ['c', 'fortran'],
  version: '0.4.2',
  meson_version: '>=0.64.0',
  default_options: ['warning_level=2'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

# Keep warning suppression scoped to the wrapped legacy backend only.
backend_c_args = []
backend_fortran_args = []

if meson.get_compiler('c').get_id() == 'gcc'
  # f2py-generated C wrappers trigger this frequently.
  backend_c_args += [
    '-Wno-cast-function-type',
    '-Wno-missing-field-initializers',
    '-Wno-unused-parameter',
    '-Wno-unused-function',
    '-Wno-misleading-indentation',
    '-Wno-incompatible-pointer-types',
  ]
endif

if meson.get_compiler('fortran').get_id() == 'gcc'
  # The bundled legacy Fortran/LAPACK-style code is warning-heavy but stable.
  backend_fortran_args += [
    '-std=legacy',
    '-Wno-unused-parameter',
    '-Wno-unused-variable',
    '-Wno-unused-dummy-argument',
    '-Wno-unused-label',
    '-Wno-conversion',
    '-Wno-compare-reals',
    '-Wno-function-elimination',
    '-Wno-maybe-uninitialized',
    '-Wno-surprising',
  ]
endif

# Ask the active Python for NumPy include dirs (works in venvs too).
incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

incdir_f2py = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# ---- f2py wrapper generation (no compilation here) ----
# Inputs: your .pyf + Fortran sources.
# Outputs: the generated C wrapper + f2py wrapper Fortran.
f2py_gen = custom_target(
  'pytmatrix_f2py_gen',
  input: [
    'pytmatrix/fortran_tm/pytmatrix.pyf',
    'pytmatrix/fortran_tm/ampld.lp.f',
    'pytmatrix/fortran_tm/lpd.f',
  ],
  output: [
    'pytmatrixmodule.c',
    'pytmatrix-f2pywrappers.f',
  ],
  command: [
    py, '-m', 'numpy.f2py',
    '@INPUT@',
    '-m', 'pytmatrix',
    # If your Fortran symbols are uppercase / need normalization:
    '--lower',
  ],
)

# ---- compile the extension module with Meson ----
# This will install into site-packages/pytmatrix/fortran_tm/
py.extension_module(
  'pytmatrix',
  [
    'pytmatrix/fortran_tm/ampld.lp.f',
    'pytmatrix/fortran_tm/lpd.f',
    f2py_gen,
    incdir_f2py / 'fortranobject.c',
  ],
  include_directories: inc_np,
  dependencies: py_dep,
  c_args: backend_c_args,
  fortran_args: backend_fortran_args,
  install: true,
  subdir: 'pytmatrix/fortran_tm',
)

# ---- install pure Python sources ----
# Keep subpackages in separate install targets to avoid __init__.py collisions.
py.install_sources(
  [
    'pytmatrix/__init__.py',
    'pytmatrix/orientation.py',
    'pytmatrix/psd.py',
    'pytmatrix/radar.py',
    'pytmatrix/refractive.py',
    'pytmatrix/scatter.py',
    'pytmatrix/tmatrix.py',
    'pytmatrix/tmatrix_aux.py',
    'pytmatrix/quadrature.py',
  ],
  subdir: 'pytmatrix',
)

py.install_sources(
  [
    'pytmatrix/tests/__init__.py',
    'pytmatrix/tests/test_tmatrix.py',
  ],
  subdir: 'pytmatrix/tests',
)

py.install_sources(
  [
    'pytmatrix/fortran_tm/__init__.py',
  ],
  subdir: 'pytmatrix/fortran_tm',
)

# ---- install data files ----
install_data(
  'pytmatrix/ice_refr.dat',
  install_dir: py.get_install_dir() / 'pytmatrix',
)

install_data(
  'pytmatrix/fortran_tm/ampld.par.f',
  install_dir: py.get_install_dir() / 'pytmatrix' / 'fortran_tm',
)
