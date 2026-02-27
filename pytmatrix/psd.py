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
"""Particle size distribution models and integration helpers."""

import sys
from datetime import datetime

try:
    import cPickle as pickle
except ImportError:
    import pickle
import warnings

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import gamma

from pytmatrix import scatter, tmatrix_aux


class PSD:
    """Base class for particle size distribution callables."""

    def __call__(self, D):
        """Evaluate the PSD.

        Parameters
        ----------
        D : float or array-like
            Particle diameter in millimeters.

        Returns
        -------
        float or numpy.ndarray
            PSD values with the same shape as ``D``.
        """
        if np.shape(D) == ():
            return 0.0
        return np.zeros_like(D)

    def __eq__(self, other):
        """Compare two PSD objects.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when objects are considered equivalent.
        """
        return False


class ExponentialPSD(PSD):
    """Exponential particle size distribution (PSD).

    Callable class implementing:

    ``N(D) = N0 * exp(-Lambda * D)``.

    Attributes
    ----------
    N0 : float
        Intercept parameter.
    Lambda : float
        Inverse scale parameter.
    D_max : float
        Maximum diameter considered by :meth:`__call__`.
    """

    def __init__(self, N0=1.0, Lambda=1.0, D_max=None):
        """Initialize an exponential PSD.

        Parameters
        ----------
        N0 : float, default=1.0
            Intercept parameter.
        Lambda : float, default=1.0
            Inverse scale parameter.
        D_max : float or None, default=None
            Maximum diameter. If ``None``, ``11/Lambda`` is used.
        """
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0 / Lambda if D_max is None else D_max

    def __call__(self, D):
        """Evaluate the exponential PSD.

        Parameters
        ----------
        D : float or array-like
            Particle diameter in millimeters.

        Returns
        -------
        float or numpy.ndarray
            PSD values, set to zero for ``D > D_max``.
        """
        psd = self.N0 * np.exp(-self.Lambda * D)
        if np.shape(D) == ():
            if self.D_max < D:
                return 0.0
        else:
            psd[self.D_max < D] = 0.0
        return psd

    def __eq__(self, other):
        """Compare two exponential PSD parameterizations.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when both objects are equivalent.
        """
        try:
            return (
                isinstance(other, ExponentialPSD)
                and (self.N0 == other.N0)
                and (self.Lambda == other.Lambda)
                and (self.D_max == other.D_max)
            )
        except AttributeError:
            return False


class UnnormalizedGammaPSD(ExponentialPSD):
    """Gamma particle size distribution (PSD).

    Callable class implementing:

    ``N(D) = N0 * D**mu * exp(-Lambda * D)``.

    Attributes
    ----------
    N0 : float
        Intercept parameter.
    Lambda : float
        Inverse scale parameter.
    mu : float
        Shape parameter.
    D_max : float
        Maximum diameter considered by :meth:`__call__`.
    """

    def __init__(self, N0=1.0, Lambda=1.0, mu=0.0, D_max=None):
        """Initialize an unnormalized gamma PSD.

        Parameters
        ----------
        N0 : float, default=1.0
            Intercept parameter.
        Lambda : float, default=1.0
            Inverse scale parameter.
        mu : float, default=0.0
            Shape parameter.
        D_max : float or None, default=None
            Maximum diameter. If ``None``, ``11/Lambda`` is used.
        """
        super().__init__(N0=N0, Lambda=Lambda, D_max=D_max)
        self.mu = mu

    def __call__(self, D):
        """Evaluate the unnormalized gamma PSD.

        Parameters
        ----------
        D : float or array-like
            Particle diameter in millimeters.

        Returns
        -------
        float or numpy.ndarray
            PSD values, set to zero for ``D > D_max`` and ``D == 0``.
        """
        # For large mu, this is better numerically than multiplying by D**mu
        psd = self.N0 * np.exp(self.mu * np.log(D) - self.Lambda * D)
        if np.shape(D) == ():
            if (self.D_max < D) or (D == 0):
                return 0.0
        else:
            psd[(self.D_max < D) | (D == 0)] = 0.0
        return psd

    def __eq__(self, other):
        """Compare two unnormalized gamma PSD parameterizations.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when both objects are equivalent.
        """
        try:
            return super().__eq__(other) and self.mu == other.mu
        except AttributeError:
            return False


class GammaPSD(PSD):
    """Normalized gamma particle size distribution (PSD).

    Callable class implementing:

    ``N(D) = Nw * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)``,
    with
    ``f(mu) = 6/(3.67**4) * (3.67+mu)**(mu+4)/Gamma(mu+4)``.

    Attributes
    ----------
    D0 : float
        Median volume diameter.
    Nw : float
        Intercept parameter.
    mu : float
        Shape parameter.
    D_max : float
        Maximum diameter considered by :meth:`__call__`.
    """

    def __init__(self, D0=1.0, Nw=1.0, mu=0.0, D_max=None):
        """Initialize a normalized gamma PSD.

        Parameters
        ----------
        D0 : float, default=1.0
            Median volume diameter.
        Nw : float, default=1.0
            Intercept parameter.
        mu : float, default=0.0
            Shape parameter.
        D_max : float or None, default=None
            Maximum diameter. If ``None``, ``3*D0`` is used.
        """
        self.D0 = float(D0)
        self.mu = float(mu)
        self.D_max = 3.0 * D0 if D_max is None else D_max
        self.Nw = float(Nw)
        self.nf = Nw * 6.0 / 3.67**4 * (3.67 + mu) ** (mu + 4) / gamma(mu + 4)

    def __call__(self, D):
        """Evaluate the normalized gamma PSD.

        Parameters
        ----------
        D : float or array-like
            Particle diameter in millimeters.

        Returns
        -------
        float or numpy.ndarray
            PSD values, set to zero for ``D > D_max`` and ``D == 0``.
        """
        d = D / self.D0
        psd = self.nf * np.exp(self.mu * np.log(d) - (3.67 + self.mu) * d)
        if np.shape(D) == ():
            if (self.D_max < D) or (D == 0.0):
                return 0.0
        else:
            psd[(self.D_max < D) | (D == 0.0)] = 0.0
        return psd

    def __eq__(self, other):
        """Compare two normalized gamma PSD parameterizations.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when both objects are equivalent.
        """
        try:
            return (
                isinstance(other, GammaPSD)
                and (self.D0 == other.D0)
                and (self.Nw == other.Nw)
                and (self.mu == other.mu)
                and (self.D_max == other.D_max)
            )
        except AttributeError:
            return False


class BinnedPSD(PSD):
    """Binned gamma particle size distribution (PSD).

    Callable class to provide a binned PSD with the given bin edges and PSD
    values.

    Notes
    -----
    Constructor inputs should contain ``n + 1`` bin edges and ``n`` bin
    values.
    """

    def __init__(self, bin_edges, bin_psd):
        """Initialize a binned PSD.

        Parameters
        ----------
        bin_edges : array-like
            Bin-edge diameters of length ``n + 1``.
        bin_psd : array-like
            Constant PSD values for each bin of length ``n``.
        """
        if len(bin_edges) != len(bin_psd) + 1:
            raise ValueError("There must be n+1 bin edges for n bins.")

        self.bin_edges = bin_edges
        self.bin_psd = bin_psd

    def psd_for_D(self, D):
        """Return the PSD value for a scalar diameter.

        Parameters
        ----------
        D : float
            Particle diameter in millimeters.

        Returns
        -------
        float
            Bin value associated with ``D``; returns ``0.0`` outside the bins.
        """
        if not (self.bin_edges[0] < D <= self.bin_edges[-1]):
            return 0.0

        # binary search for the right bin
        start = 0
        end = len(self.bin_edges)
        while end - start > 1:
            half = (start + end) // 2
            if self.bin_edges[start] < D <= self.bin_edges[half]:
                end = half
            else:
                start = half

        return self.bin_psd[start]

    def __call__(self, D):
        """Evaluate the binned PSD.

        Parameters
        ----------
        D : float or array-like
            Particle diameter in millimeters.

        Returns
        -------
        float or numpy.ndarray
            PSD values corresponding to each diameter.
        """
        if np.shape(D) == ():  # D is a scalar
            return self.psd_for_D(D)
        return np.array([self.psd_for_D(d) for d in D])

    def __eq__(self, other):
        """Compare two binned PSD definitions.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` when both bin edges and bin values are identical.
        """
        if other is None:
            return False
        return (
            len(self.bin_edges) == len(other.bin_edges)
            and (self.bin_edges == other.bin_edges).all()
            and (self.bin_psd == other.bin_psd).all()
        )


class PSDIntegrator:
    """A class used to perform computations over PSDs.

    This class can be used to integrate scattering properties over particle
    size distributions.

    Initialize an instance of the class and set the attributes as described
    below. Call init_scatter_table to compute the lookup table for scattering
    values at different scatterer geometries. Set the class instance as the
    psd_integrator attribute of a Scatterer object to enable PSD averaging for
    that object.

    After a call to init_scatter_table, the scattering properties can be
    retrieved multiple times without re-initializing. However, the geometry of
    the Scatterer instance must be set to one of those specified in the
    "geometries" attribute.

    Attributes
    ----------
    num_points : int
        Number of diameters used to sample PSD and scattering properties.
    m_func : callable or None
        Optional refractive-index function of diameter.
    axis_ratio_func : callable or None
        Optional axis-ratio function of diameter.
    D_max : float or None
        Maximum scatterer diameter used in table initialization.
    geometries : tuple
        Geometry tuples ``(thet0, thet, phi0, phi, alpha, beta)`` to
        precompute.
    """

    attrs = {"num_points", "m_func", "axis_ratio_func", "D_max", "geometries"}

    def __init__(self, **kwargs):
        """Initialize a PSD integrator.

        Parameters
        ----------
        **kwargs
            Optional overrides for class attributes in ``PSDIntegrator.attrs``.
        """
        self.num_points = 1024
        self.m_func = None
        self.axis_ratio_func = None
        self.D_max = None
        self.geometries = (tmatrix_aux.geom_horiz_back,)

        for k in kwargs:
            if k in self.__class__.attrs:
                self.__dict__[k] = kwargs[k]

        self._S_table = None
        self._Z_table = None
        self._angular_table = None
        self._previous_psd = None

    def __call__(self, psd, geometry):
        """Integrate scattering matrices for a PSD at one geometry.

        Parameters
        ----------
        psd : PSD
            Particle size distribution instance.
        geometry : tuple
            Geometry tuple ``(thet0, thet, phi0, phi, alpha, beta)``.

        Returns
        -------
        tuple
            ``(S, Z)`` amplitude and phase matrices.
        """
        return self.get_SZ(psd, geometry)

    def get_SZ(self, psd, geometry):
        """Compute scattering matrices for a PSD and geometry.

        Parameters
        ----------
        psd : PSD
            Particle size distribution instance.
        geometry : tuple
            Geometry tuple ``(thet0, thet, phi0, phi, alpha, beta)``.

        Returns
        -------
        tuple
            Amplitude and phase matrices ``(S, Z)``.
        """
        if (self._S_table is None) or (self._Z_table is None):
            raise AttributeError("Initialize or load the scattering table first.")

        if (not isinstance(psd, PSD)) or self._previous_psd != psd:
            self._S_dict = {}
            self._Z_dict = {}
            psd_w = psd(self._psd_D)

            for geom in self.geometries:
                self._S_dict[geom] = trapezoid(self._S_table[geom] * psd_w, self._psd_D)
                self._Z_dict[geom] = trapezoid(self._Z_table[geom] * psd_w, self._psd_D)

            self._previous_psd = psd

        return (self._S_dict[geometry], self._Z_dict[geometry])

    def get_angular_integrated(self, psd, geometry, property_name, h_pol=True):
        """Integrate an angular scattering property over a PSD.

        Parameters
        ----------
        psd : PSD
            Particle size distribution instance.
        geometry : tuple
            Geometry tuple ``(thet0, thet, phi0, phi, alpha, beta)``.
        property_name : {"sca_xsect", "ext_xsect", "asym"}
            Name of the angular quantity to integrate.
        h_pol : bool, default=True
            Polarization selector. Horizontal when ``True``, vertical when
            ``False``.

        Returns
        -------
        float
            PSD-integrated value of the requested quantity.
        """
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated " + "quantities first.",
            )

        pol = "h_pol" if h_pol else "v_pol"
        psd_w = psd(self._psd_D)

        def sca_xsect(geom):
            return trapezoid(
                self._angular_table["sca_xsect"][pol][geom] * psd_w,
                self._psd_D,
            )

        if property_name == "sca_xsect":
            sca_prop = sca_xsect(geometry)
        elif property_name == "ext_xsect":
            sca_prop = trapezoid(
                self._angular_table["ext_xsect"][pol][geometry] * psd_w,
                self._psd_D,
            )
        elif property_name == "asym":
            sca_xsect_int = sca_xsect(geometry)
            if sca_xsect_int > 0:
                sca_prop = trapezoid(
                    self._angular_table["asym"][pol][geometry]
                    * self._angular_table["sca_xsect"][pol][geometry]
                    * psd_w,
                    self._psd_D,
                )
                sca_prop /= sca_xsect_int
            else:
                sca_prop = 0.0

        return sca_prop

    def init_scatter_table(self, tm, angular_integration=False, verbose=False):
        """Initialize the scattering lookup tables.

        Initialize the scattering lookup tables for the different geometries.
        Before calling this, the following attributes must be set:
           num_points, m_func, axis_ratio_func, D_max, geometries
        and additionally, all the desired attributes of the Scatterer class
        (e.g. wavelength, aspect ratio).

        Parameters
        ----------
        tm : pytmatrix.tmatrix.Scatterer
            Scatterer instance used to tabulate single-particle properties.
        angular_integration : bool, default=False
            If ``True``, also tabulate angular-integrated quantities
            (scattering/extinction cross section and asymmetry parameter).
        verbose : bool, default=False
            If ``True``, print progress messages during table generation.
        """
        self._psd_D = np.linspace(
            self.D_max / self.num_points,
            self.D_max,
            self.num_points,
        )

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None
        self._m_table = np.empty(self.num_points, dtype=complex)
        if angular_integration:
            self._angular_table = {
                "sca_xsect": {"h_pol": {}, "v_pol": {}},
                "ext_xsect": {"h_pol": {}, "v_pol": {}},
                "asym": {"h_pol": {}, "v_pol": {}},
            }
        else:
            self._angular_table = None

        old_m, old_axis_ratio, old_radius, old_geom, old_psd_integrator = (
            tm.m,
            tm.axis_ratio,
            tm.radius,
            tm.get_geometry(),
            tm.psd_integrator,
        )

        try:
            # temporarily disable PSD integration to avoid recursion
            tm.psd_integrator = None

            for geom in self.geometries:
                self._S_table[geom] = np.empty((2, 2, self.num_points), dtype=complex)
                self._Z_table[geom] = np.empty((4, 4, self.num_points))

                if angular_integration:
                    for int_var in ["sca_xsect", "ext_xsect", "asym"]:
                        for pol in ["h_pol", "v_pol"]:
                            self._angular_table[int_var][pol][geom] = np.empty(
                                self.num_points,
                            )

            for i, D in enumerate(self._psd_D):
                if verbose:
                    print(f"Computing point {i} at D={D}...")
                if self.m_func is not None:
                    tm.m = self.m_func(D)
                if self.axis_ratio_func is not None:
                    tm.axis_ratio = self.axis_ratio_func(D)
                self._m_table[i] = tm.m
                tm.radius = D / 2.0
                for geom in self.geometries:
                    tm.set_geometry(geom)
                    S, Z = tm.get_SZ_orient()
                    self._S_table[geom][:, :, i] = S
                    self._Z_table[geom][:, :, i] = Z

                    if angular_integration:
                        for pol in ["h_pol", "v_pol"]:
                            h_pol = pol == "h_pol"
                            self._angular_table["sca_xsect"][pol][geom][i] = scatter.sca_xsect(tm, h_pol=h_pol)
                            self._angular_table["ext_xsect"][pol][geom][i] = scatter.ext_xsect(tm, h_pol=h_pol)
                            self._angular_table["asym"][pol][geom][i] = scatter.asym(
                                tm,
                                h_pol=h_pol,
                            )
        finally:
            # restore old values
            tm.m, tm.axis_ratio, tm.radius, tm.psd_integrator = (
                old_m,
                old_axis_ratio,
                old_radius,
                old_psd_integrator,
            )
            tm.set_geometry(old_geom)

    def save_scatter_table(self, fn, description=""):
        """Save the scattering lookup tables.

        Save the state of the scattering lookup tables to a file.
        This can be loaded later with load_scatter_table.

        Other variables will not be saved, but this does not matter because
        the results of the computations are based only on the contents
        of the table.

        Parameters
        ----------
        fn : str or path-like
            Output filename for the serialized lookup table.
        description : str, default=""
            Free-text description stored with the table metadata.
        """
        data = {
            "description": description,
            "time": datetime.now(),
            "psd_scatter": (
                self.num_points,
                self.D_max,
                self._psd_D,
                self._S_table,
                self._Z_table,
                self._angular_table,
                self._m_table,
                self.geometries,
            ),
            "python_version": sys.version[0:7],
            "numpy_version": np.__version__,
            "tmatrix_version": tmatrix_aux.VERSION,
        }
        with open(fn, "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_scatter_table(self, fn):
        """Load the scattering lookup tables.

        Load the scattering lookup tables saved with save_scatter_table.

        Parameters
        ----------
        fn : str or path-like
            Input filename created by :meth:`save_scatter_table`.

        Returns
        -------
        tuple
            Pair ``(time, description)`` read from table metadata.
        """
        with open(fn, "rb") as f:
            data = pickle.load(f)

        if ("tmatrix_version" not in data) or (data["tmatrix_version"] != tmatrix_aux.VERSION):
            warnings.warn("Loading data saved with another version.", Warning, stacklevel=2)

        (
            self.num_points,
            self.D_max,
            self._psd_D,
            self._S_table,
            self._Z_table,
            self._angular_table,
            self._m_table,
            self.geometries,
        ) = data["psd_scatter"]
        return (data["time"], data["description"])
