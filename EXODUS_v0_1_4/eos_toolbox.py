# eos_toolbox.py for EXODUS v0.1.4
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
# All isothermal equations of state (EoS) used by EXODUS (pure numpy, no Qt):
#
#   1. EoS MODELS     P(V) for the models offered by the "EoS Fit" pop-up,
#                     the K0'' implied by each truncated model, and
#                     eos_pressure() as the common entry point.
#   2. BM3 HELPERS    The 3rd-order Birch-Murnaghan EoS used by the main
#                     window: pressure (and its error) of a fitted cell from
#                     the JCPDS EoS constants, and unit cells at a target
#                     pressure for seeding the Pawley fit.
#
# Models   (f_E = [(V0/V)^(2/3) - 1]/2,  f_N = ln(V0/V)/3,  X = (V/V0)^(1/3))
# ------
# Murnaghan        P = K0/K' [(V0/V)^K' - 1]                       [1]
# Birch-Murnaghan  2nd: P = 3K0 f_E (1+2f_E)^(5/2)       (K' = 4)  [2, 3]
#                  3rd: ... [1 + 3/2 (K'-4) f_E]
#                  4th: ... [1 + 3/2 (K'-4) f_E
#                            + 3/2 (K0 K'' + (K'-4)(K'-3) + 35/9) f_E^2]
# Vinet            P = 3K0 (1-X) X^-2 exp[3/2 (K'-1)(1-X)]         [4]
# Natural strain   P = 3K0 (V0/V) f_N [1 + 3/2 (K'-2) f_N]         [5]
# (Poirier-Tarantola, 3rd order)
#
# Implied K0'': truncating an EoS at a given order fixes the next pressure
# derivative. BM2 implies K' = 4; BM2, BM3, Vinet, natural strain and
# Murnaghan each imply a K0'' (see implied_kpp) [3, 6].
#
# References
# ----------
# [1] F. D. Murnaghan, Proc. Natl. Acad. Sci. USA 30, 244-247 (1944).
# [2] F. Birch, Phys. Rev. 71, 809-824 (1947).
# [3] R. J. Angel, Rev. Mineral. Geochem. 41, 35-59 (2000).
# [4] P. Vinet, J. Ferrante, J. H. Rose, J. R. Smith,
#     J. Geophys. Res. 92, 9319-9325 (1987).
# [5] J.-P. Poirier, A. Tarantola, Phys. Earth Planet. Inter. 109, 1-8 (1998).
# [6] O. L. Anderson, Equations of State of Solids for Geophysics and Ceramic
#     Science, Oxford University Press (1995).
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import numpy as np
from scipy.optimize import brentq

import crystallography_toolbox as cryst


# =============================================================================
#                               EoS MODELS
# =============================================================================
def _eulerian(V, V0):
    """Eulerian finite strain f_E = [(V0/V)^(2/3) - 1] / 2."""
    return ((V0 / V) ** (2.0 / 3.0) - 1.0) / 2.0


def p_murnaghan(V, V0, K0, Kp, Kpp=None):
    """Murnaghan EoS [1]."""
    Kp = Kp if abs(Kp) > 1e-12 else 1e-12
    return K0 / Kp * ((V0 / V) ** Kp - 1.0)


def p_bm2(V, V0, K0, Kp=4.0, Kpp=None):
    """2nd-order Birch-Murnaghan EoS [2] (K' = 4 by construction)."""
    f = _eulerian(V, V0)
    return 3.0 * K0 * f * (1.0 + 2.0 * f) ** 2.5


def p_bm3(V, V0, K0, Kp, Kpp=None):
    """3rd-order Birch-Murnaghan EoS [2]."""
    f = _eulerian(V, V0)
    return 3.0 * K0 * f * (1.0 + 2.0 * f) ** 2.5 * (1.0 + 1.5 * (Kp - 4.0) * f)


def p_bm4(V, V0, K0, Kp, Kpp):
    """4th-order Birch-Murnaghan EoS [3]; reduces to BM3 for the implied K''."""
    f = _eulerian(V, V0)
    c2 = 1.5 * (K0 * Kpp + (Kp - 4.0) * (Kp - 3.0) + 35.0 / 9.0)
    return 3.0 * K0 * f * (1.0 + 2.0 * f) ** 2.5 * (
        1.0 + 1.5 * (Kp - 4.0) * f + c2 * f ** 2)


def p_vinet(V, V0, K0, Kp, Kpp=None):
    """Vinet EoS [4]."""
    X = (V / V0) ** (1.0 / 3.0)
    return 3.0 * K0 * (1.0 - X) / X ** 2 * np.exp(1.5 * (Kp - 1.0) * (1.0 - X))


def p_natural_strain(V, V0, K0, Kp, Kpp=None):
    """3rd-order natural-strain (Poirier-Tarantola) EoS [5]."""
    fN = np.log(V0 / V) / 3.0
    return 3.0 * K0 * (V0 / V) * fN * (1.0 + 1.5 * (Kp - 2.0) * fN)


# name -> (function, K' refined?, K'' refined?)
EOS_MODELS = {
    'BM3 (Birch-Murnaghan 3rd)':         (p_bm3, True, False),
    'BM2 (Birch-Murnaghan 2nd)':         (p_bm2, False, False),
    'BM4 (Birch-Murnaghan 4th)':         (p_bm4, True, True),
    'Vinet':                             (p_vinet, True, False),
    'Murnaghan':                         (p_murnaghan, True, False),
    'Natural strain (Poirier-Tarantola)': (p_natural_strain, True, False),
}
DEFAULT_MODEL = 'BM3 (Birch-Murnaghan 3rd)'



def implied_kpp(model, K0, Kp):
    """K0'' (1/GPa) implied by a truncated EoS [3, 6].

    BM2       K'' = -35 / (9 K0)                       (with K' = 4)
    BM3       K'' = -[(3 - K')(4 - K') + 35/9] / K0
    Vinet     K'' = -[(K'/2)^2 + K'/2 - 19/36] / K0
    Natural   K'' = -[1 + (K' - 2) + (K' - 2)^2] / K0
    Murnaghan K'' = 0
    Returns NaN for BM4 (K'' is refined there) and for K0 = 0.
    """
    if not K0:
        return float('nan')
    if model.startswith('BM2'):
        return -35.0 / (9.0 * K0)
    if model.startswith('BM3'):
        return -((3.0 - Kp) * (4.0 - Kp) + 35.0 / 9.0) / K0
    if model.startswith('Vinet'):
        return -((Kp / 2.0) ** 2 + Kp / 2.0 - 19.0 / 36.0) / K0
    if model.startswith('Natural'):
        return -(1.0 + (Kp - 2.0) + (Kp - 2.0) ** 2) / K0
    if model.startswith('Murnaghan'):
        return 0.0
    return float('nan')


def eos_pressure(model, V, V0, K0, Kp=4.0, Kpp=0.0):
    """P(V) in GPa for one of the EOS_MODELS (V may be an array)."""
    fn, _uk, _ukk = EOS_MODELS[model]
    V = np.asarray(V, dtype=float)
    with np.errstate(all='ignore'):
        return fn(V, V0, K0, Kp, Kpp)


# =============================================================================
#                     BM3 HELPERS USED BY THE MAIN WINDOW
# =============================================================================
def bm3_pressure(V, V0, K0, K0P):
    """
    Pressure (GPa) from the 3rd-order Birch-Murnaghan EoS [2] for a volume
    V (A^3), written in volume ratios; identical to p_bm3().
    """
    V = float(V)
    V0 = float(V0)
    P = (3/2) * K0 * ((V0 / V)**(7/3) - (V0 / V)**(5/3)) * \
        (1 + (3/4) * (K0P - 4) * ((V0 / V)**(2/3) - 1))
    return P


def bm3_pressure_and_error(V, dV, V0, K0, K0P):
    """(P, sigma_P) in GPa from the BM3 EoS. sigma_P only contains the
    propagated volume error, sigma_P = |dP/dV| sigma_V; the EoS parameters
    are treated as exact."""
    if V is None or not np.isfinite(V) or V <= 0:
        return float('nan'), float('nan')
    try:
        P = float(bm3_pressure(V, V0, K0, K0P))
    except (ZeroDivisionError, ValueError, TypeError):
        return float('nan'), float('nan')
    if dV is None or not np.isfinite(dV):
        return P, float('nan')
    h = max(abs(V) * 1e-6, 1e-8)
    dPdV = (bm3_pressure(V + h, V0, K0, K0P) - bm3_pressure(V - h, V0, K0, K0P)) / (2 * h)
    return P, float(abs(dPdV) * dV)


def find_volume_at_pressure(P, unitCell, K0, K0P, maxIter=50):
    """
    Volume (A^3, float) at pressure P from the BM3 EoS, with the volume of
    `unitCell` as V0. The bracket [0.5 V0, 1.5 V0] is widened until the
    pressure difference changes sign, then solved with Brent's method.
    Returns None if no solution is found or the cell is invalid.
    """
    V0 = cryst.unit_cell_volume(unitCell)
    if V0 is None:
        return None

    def volume_difference(V):
        return bm3_pressure(V, V0, K0, K0P) - P

    V_low = V0 * 0.5
    V_high = V0 * 1.5

    for i in range(maxIter):
        f_low = volume_difference(V_low)
        f_high = volume_difference(V_high)

        if f_low * f_high < 0:
            # Sign change -> the root is bracketed
            try:
                return brentq(volume_difference, V_low, V_high)
            except Exception as e:
                print(f"Brentq failed: {e}")
                return None

        # No sign change: widen the bracket
        V_low *= 0.9
        V_high *= 1.1

    print("ERROR: Could not find sign change after expanding bounds.")
    return None


def find_uc_at_p(unitCell0, P, V0, K0, K0P):
    """
    Scale a 0 GPa unit cell to pressure P using the BM3 EoS.

    V(P) is solved with the cell volume of `unitCell0` as reference and
    the cell is scaled by (V(P)/V0)^(1/3), where V0 is the value passed in
    (the JCPDS V0, which load_jcpds sets to the same cell volume).

    LIMITATION (isotropic compression assumption)
    ---------------------------------------------
    a, b and c are all multiplied by the same factor and the angles are
    kept. This is exact for CUBIC and a reasonable approximation for cells
    with similar axial compressibilities. For lower symmetries it is only
    an *initial guess*; the Pawley refinement then moves every lattice
    parameter independently.
    """
    VP = find_volume_at_pressure(P, unitCell0, K0, K0P)

    if VP is None:
        print("Volume at pressure could not be determined.")
        return None

    scale = (VP / V0) ** (1/3)
    return [
        unitCell0[0] * scale,
        unitCell0[1] * scale,
        unitCell0[2] * scale,
        unitCell0[3],
        unitCell0[4],
        unitCell0[5]
    ]


def scale_uc_at_p(unitCellFit, PTarget, unitCell0, K0, K0P):
    """
    Scale a fitted unit cell to the BM3 volume at pressure PTarget.

    Parameters:
    - unitCellFit: list of 6 lattice parameters of the fitted cell
    - PTarget: desired target pressure (GPa)
    - unitCell0: 0 GPa (JCPDS) unit cell, defines V0
    - K0, K0P: BM3 bulk modulus and its pressure derivative

    Returns:
    - the scaled unit cell, or None if a volume cannot be determined

    LIMITATION (isotropic compression assumption)
    ---------------------------------------------
    Same as find_uc_at_p: a, b, c are scaled by the same cube-root factor
    and the angles are kept. Exact for CUBIC, an approximation otherwise.
    Intended as the initial guess for the next frame in a sequential
    ('pressureWalk') fit.
    """
    V_fit = cryst.unit_cell_volume(unitCellFit)
    V0 = cryst.unit_cell_volume(unitCell0)
    if V_fit is None or V0 is None:
        print("Invalid unit cell: volume could not be computed.")
        return None

    # BM3 volume at the target pressure
    V_P = find_volume_at_pressure(PTarget, unitCell0, K0, K0P)
    if V_P is None:
        print("Volume at target pressure could not be determined.")
        return None

    scale = (V_P / V_fit) ** (1/3)

    return [unitCellFit[0] * scale,
            unitCellFit[1] * scale,
            unitCellFit[2] * scale,
            unitCellFit[3],
            unitCellFit[4],
            unitCellFit[5]]


def safe_uc_at_p(uc0, P, K0, K0P):
    """EoS-scaled unit cell at pressure P, falling back to uc0 if the EoS
    can't be evaluated (e.g. K0 = 0 or P outside the solvable range)."""
    uc0 = [float(x) for x in uc0]
    try:
        if abs(float(P)) < 1e-12 or K0 is None or float(K0) <= 0:
            return np.array(uc0, dtype=float)
        V0 = cryst.unit_cell_volume(uc0)
        if V0 is None:
            return np.array(uc0, dtype=float)
        uc = find_uc_at_p(uc0, float(P), V0, float(K0), float(K0P))
        if uc is None or not np.all(np.isfinite(uc)):
            return np.array(uc0, dtype=float)
        return np.array(uc, dtype=float)
    except Exception:
        return np.array(uc0, dtype=float)
