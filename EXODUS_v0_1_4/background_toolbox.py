# background_toolbox.py for EXODUS v0.1.4
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Bernhard Massani
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE
# file or <https://www.gnu.org/licenses/> for details.
#
# ----------------------------------------------------------------------------
#
# Background determination and subtraction for 1D diffraction patterns
# (no Qt):
#
#   * polynomial background through the pattern with the peaks cut out
#     (peaks found with scipy.signal.find_peaks),
#   * asymmetric least squares (ALS) baseline [1].
#
# All functions work on patterns already in memory (see
# data_toolbox.DataManager); compute_background() is the single entry point
# used by the GUI and the cache.
#
# References
# ----------
# [1] P. H. C. Eilers, H. F. M. Boelens, "Baseline correction with
#     asymmetric least squares smoothing", Leiden University Medical Centre
#     report (2005).
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import bisect

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.signal import find_peaks



# ============================================================================
#                           POLYNOMIAL BACKGROUND
# ============================================================================
def background_fit_poly(twoTheta, valueInt, errorInt,
                         peakSearchAuto=True, prominence=0.10, height=0.08,
                         order=6, excludePeakList=[], peakwidth=0.35):
    """Array-in / array-out variant of backgroundFit. Identical logic,
    but operates on pre-loaded arrays (e.g. from DataManager.get_cropped)
    instead of a file path. backgroundFit() is now a thin wrapper."""
    twoTheta = np.array(twoTheta)
    valueInt = np.array(valueInt)
    errorInt = np.array(errorInt)

    twoTheta_unaltered = np.array(twoTheta)
    valueInt_unaltered = np.array(valueInt)
    errorInt_unaltered = np.array(errorInt)

    peaks, _ = find_peaks(valueInt, height=max(valueInt) * height,
                          threshold=None, distance=None, prominence=prominence,
                          width=None, wlen=None, rel_height=0.5, plateau_size=None)

    if excludePeakList:
        peaks = excludePeakList

    peak_list = []
    twoTheta_removed = twoTheta
    valueInt_removed = valueInt
    for i, peak in enumerate(twoTheta[peaks]):
        peak_list.append(peak)
        peakmin = bisect.bisect(twoTheta, peak - peakwidth)
        peakmax = bisect.bisect(twoTheta, peak + peakwidth)
        print(f'Peak {i} ({peak} deg) removed in the range: '
              f'{round(peak - peakwidth, 1)} - {round(peak + peakwidth, 1)}')
        twoTheta[peakmin:peakmax] = np.nan
        valueInt[peakmin:peakmax] = np.nan
        twoTheta_removed = twoTheta[np.isfinite(twoTheta)]
        valueInt_removed = valueInt[np.isfinite(valueInt)]

    con_fit = np.polyfit(twoTheta_removed, valueInt_removed, order)
    polyFit = np.poly1d(con_fit)

    return (twoTheta_unaltered, valueInt_unaltered, errorInt_unaltered,
            polyFit(twoTheta_unaltered), peak_list)


# ============================================================================
#                               ALS BACKGROUND
# ============================================================================
def background_fit_als(twoTheta, valueInt, errorInt,
                             lam=1e5, p=0.01, niter=10):
    """Array-in / array-out variant of backgroundFit_ALS. Identical maths,
    but no disk I/O - call this when you already have the cropped pattern
    in memory (e.g. via DataManager.get_cropped). The original disk-based
    backgroundFit_ALS is now a thin wrapper around this."""
    twoTheta = np.asarray(twoTheta)
    valueInt = np.asarray(valueInt)
    errorInt = np.asarray(errorInt)

    L = len(valueInt)
    if L < 3:
        # Not enough points for the second-derivative matrix
        return twoTheta, valueInt, errorInt, np.zeros_like(valueInt), valueInt.copy()

    # Second derivative matrix.
    # - dtype=float silences the int->float FutureWarning in scipy.sparse.diags.
    # - .tocsc() up-front so D.T @ D, the additions below, and the spsolve
    #   call all happen in CSC format and we don't get a SparseEfficiencyWarning.
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), dtype=float).tocsc()
    D = lam * (D.T @ D)

    # Initialize weights
    w = np.ones(L)
    for _ in range(int(niter)):
        # W is built as CSC each iteration (cheap; just a diagonal),
        # so Z = W + D stays CSC and spsolve doesn't warn.
        W = sp.diags(w, 0, format='csc')
        Z = (W + D).tocsc()
        z = splinalg.spsolve(Z, w * valueInt)
        # Update weights (asymmetric)
        w = p * (valueInt > z) + (1 - p) * (valueInt < z)

    BGInt = z
    valueIntBGsub = valueInt - BGInt
    return twoTheta, valueInt, errorInt, BGInt, valueIntBGsub


# ============================================================================
#                         SUBTRACTION AND DISPATCH
# ============================================================================
def subtract_background(twoTheta, valueInt, valueBG):
    valueIntBGsub = valueInt - valueBG
    twoTheta = np.array(twoTheta)
    valueIntBGsub = np.array(valueIntBGsub)

    return twoTheta, valueInt, valueBG, valueIntBGsub


def compute_background(twoTheta, valueInt, errorInt, method, params):
    """Background of one cropped pattern with the GUI settings.

    method : 'ALS' or 'POLY'
    params : dict with 'lam' (ALS lambda in units of 1e6), 'p', 'niter'
             (ALS) and 'prominence', 'height', 'order', 'peakwidth' (POLY)
    Returns (twoTheta, valueInt, valueBG, valueIntBGsub).
    """
    if method == 'ALS':
        twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = \
            background_fit_als(twoTheta, valueInt, errorInt,
                                     lam=params['lam'] * 1e6,
                                     p=params['p'],
                                     niter=params['niter'])
        return twoTheta, valueInt, valueBG, valueIntBGsub
    if method == 'POLY':
        twoTheta, valueInt, errorInt, valueBG, _peaks = \
            background_fit_poly(twoTheta, valueInt, errorInt,
                                 prominence=params['prominence'],
                                 height=params['height'],
                                 order=params['order'],
                                 excludePeakList=[],
                                 peakwidth=params['peakwidth'])
        return subtract_background(twoTheta, valueInt, valueBG)
    raise ValueError(f"Unknown background method: {method}")
