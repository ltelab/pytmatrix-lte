"""
Copyright (C) 2009-2015 Jussi Leinonen, Finnish Meteorological Institute,
California Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np


def discrete_gautschi(z, w, n_iter):
    """Compute recurrence coefficients for orthogonal polynomials using Gautschi's method.

    Parameters
    ----------
    z : ndarray of shape (n,)
        Support points (e.g., abscissas of the discrete measure).
    w : ndarray of shape (n,)
        Non-negative weights associated with the support points.
    n_iter : int
        Number of recurrence coefficients (polynomial degrees) to compute.

    Returns
    -------
    a : ndarray of shape (n_iter,)
        Diagonal recurrence coefficients (also called alpha).
    b : ndarray of shape (n_iter - 1,)
        Subdiagonal recurrence coefficients (also called beta).

    Notes
    -----
    This function implements the discrete version of the three-term recurrence
    relation used to generate orthogonal polynomials with respect to a discrete
    inner product:

        ⟨f, g⟩ = ∑ w_i f(z_i) g(z_i)

    using the algorithm of Gautschi (1982). The polynomials are not returned,
    only the recurrence coefficients `a` and `b` such that:

        p_{j+1}(z) = (z - a_j) * p_j(z) - b_{j-1} * p_{j-1}(z)

    with p_0(z) = 1 (normalized).

    """
    # Initial polynomial p_0 (normalized constant)
    p = np.ones(z.shape)
    p /= np.sqrt(np.dot(p, p))
    p_prev = np.zeros(z.shape)
    p_prev_norm = 1.0  # Dummy init for first iteration (b[0] set to 0)
    wz = z * w
    a = np.empty(n_iter, dtype=np.float64)
    b = np.empty(n_iter - 1, dtype=np.float64)

    for j in range(n_iter):
        p_norm = np.dot(w * p, p)
        # Diagonal coefficient
        a[j] = np.dot(wz * p, p) / p_norm
        # Subdiagonal coefficient (skip b[0], only fill from b[0] onward)
        if j > 0:
            b[j - 1] = p_norm / p_prev_norm
        # 3-term recurrence
        p_new = (z - a[j]) * p - (b[j - 1] if j > 0 else 0.0) * p_prev
        # Update variables
        (p_prev, p_prev_norm) = (p, p_norm)
        p = p_new

    return a, b


def get_points_and_weights(
    w_func=lambda x: np.ones(x.shape), left=-1.0, right=1.0, num_points=5, n=4096
):
    r"""Quadratude points and weights for a weighting function.

    Points and weights for approximating the integral
        I = \int_left^right f(x) w(x) dx
    given the weighting function w(x) using the approximation
        I ~ w_i f(x_i)

    Args:
        w_func: The weighting function w(x). Must be a function that takes
            one argument and is valid over the open interval (left, right).
        left: The left boundary of the interval
        right: The left boundary of the interval
        num_points: number of integration points to return
        n: the number of points to evaluate w_func at.

    Returns:
        A tuple (points, weights) where points is a sorted array of the
        points x_i and weights gives the corresponding weights w_i.
    """

    dx = (float(right) - left) / n
    z = np.hstack(np.linspace(left + 0.5 * dx, right - 0.5 * dx, n))
    w = dx * w_func(z)

    (a, b) = discrete_gautschi(z, w, num_points)
    alpha = a
    beta = np.sqrt(b)

    J = np.diag(alpha)
    J += np.diag(beta, k=-1)
    J += np.diag(beta, k=1)

    (points, v) = np.linalg.eigh(J)
    ind = points.argsort()
    points = points[ind]
    weights = v[0, :] ** 2 * w.sum()
    weights = weights[ind]

    return (points, weights)
