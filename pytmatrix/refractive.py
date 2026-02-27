# Copyright (C) 2009-2015 Jussi Leinonen, Finnish Meteorological Institute,
# California Institute of Technology

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Refractive-index mixing and lookup utilities."""

from os import path

import numpy as np
from scipy import interpolate

from pytmatrix.tmatrix_aux import wl_C, wl_Ka, wl_Ku, wl_S, wl_W, wl_X

# Water refractive indices for different bands at 0 C
m_w_0C = {
    wl_S: complex(9.075, 1.253),
    wl_C: complex(8.328, 2.217),
    wl_X: complex(7.351, 2.785),
    wl_Ku: complex(6.265, 2.993),
    wl_Ka: complex(4.040, 2.388),
    wl_W: complex(2.880, 1.335),
}


# Water refractive indices for different bands at 10 C
m_w_10C = {
    wl_S: complex(9.019, 0.887),
    wl_C: complex(8.601, 1.687),
    wl_X: complex(7.942, 2.332),
    wl_Ku: complex(7.042, 2.777),
    wl_Ka: complex(4.638, 2.672),
    wl_W: complex(3.117, 1.665),
}


# Water refractive indices for different bands at 20 C
m_w_20C = {
    wl_S: complex(8.876, 0.653),
    wl_C: complex(8.633, 1.289),
    wl_X: complex(8.208, 1.886),
    wl_Ku: complex(7.537, 2.424),
    wl_Ka: complex(5.206, 2.801),
    wl_W: complex(3.382, 1.941),
}


# Ice density in g/cm^3
ice_density = 0.9167


def mg_refractive(m, mix):
    """Maxwell-Garnett EMA for the refractive index.

    Parameters
    ----------
    m : tuple of complex
        Complex refractive indices of each medium.
    mix : tuple of float
        Volume fractions of each medium. Must satisfy ``len(mix) == len(m)``.
        Fractions are normalized by ``sum(mix)``.

    Returns
    -------
    complex
        Maxwell-Garnett approximation for the effective-medium refractive
        index.

    Notes
    -----
    If ``len(m) == 2``, the first element is treated as matrix and the second
    as inclusion. For ``len(m) > 2``, components are mixed recursively.
    """
    if len(m) == 2:
        cF = float(mix[1]) / (mix[0] + mix[1]) * (m[1] ** 2 - m[0] ** 2) / (m[1] ** 2 + 2 * m[0] ** 2)
        er = m[0] ** 2 * (1.0 + 2.0 * cF) / (1.0 - cF)
        m = np.sqrt(er)
    else:
        m_last = mg_refractive(m[-2:], mix[-2:])
        mix_last = mix[-2] + mix[-1]
        m = mg_refractive((*m[:-2], m_last), (*mix[:-2], mix_last))
    return m


def bruggeman_refractive(m, mix):
    """Bruggeman EMA for the refractive index.

    Parameters
    ----------
    m : tuple of complex
        Two complex refractive indices.
    mix : tuple of float
        Two volume fractions.

    Returns
    -------
    complex
        Bruggeman effective-medium refractive index.

    Notes
    -----
    Unlike :func:`mg_refractive`, this routine only supports two components.
    """
    f1 = mix[0] / sum(mix)
    f2 = mix[1] / sum(mix)
    e1 = m[0] ** 2
    e2 = m[1] ** 2
    a = -2 * (f1 + f2)
    b = 2 * f1 * e1 - f1 * e2 + 2 * f2 * e2 - f2 * e1
    c = (f1 + f2) * e1 * e2
    e_eff = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    return np.sqrt(e_eff)


def ice_refractive(file):
    """Create an interpolator for refractive index of ice.

    Parameters
    ----------
    file : str or path-like
        Path to refractive-index lookup table (for example
        ``ice_refr.dat`` from
        ``http://www.atmos.washington.edu/ice_optical_constants/``).

    Returns
    -------
    callable
        Function ``ref(wl, snow_density)`` taking wavelength in millimeters
        and snow density in g/cm^3.
    """
    D = np.loadtxt(file)

    log_wl = np.log10(D[:, 0] / 1000)
    re = D[:, 1]
    log_im = np.log10(D[:, 2])

    iobj_re = interpolate.interp1d(log_wl, re)
    iobj_log_im = interpolate.interp1d(log_wl, log_im)

    def ref(wl, snow_density):
        lwl = np.log10(wl)
        try:
            len(lwl)
        except TypeError:
            mi_sqr = complex(iobj_re(lwl), 10 ** iobj_log_im(lwl)) ** 2
        else:
            mi_sqr = (
                np.array(
                    [complex(a, b) for (a, b) in zip(iobj_re(lwl), 10 ** iobj_log_im(lwl), strict=False)],
                )
                ** 2
            )

        c = (mi_sqr - 1) / (mi_sqr + 2) * snow_density / ice_density
        return np.sqrt((1 + 2 * c) / (1 - c))

    return ref


module_path = path.split(path.abspath(__file__))[0]

mi = ice_refractive(module_path + "/ice_refr.dat")
