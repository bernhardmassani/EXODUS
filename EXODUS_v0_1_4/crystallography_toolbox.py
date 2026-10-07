# crystallography_toolbox.py for EXODUS v0.1.4
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
# Unit-cell geometry shared by every other module (no Qt, no fitting):
#
#   * crystal-system names, symmetry ties and the independent lattice
#     parameters of each system,
#   * unit-cell volume,
#   * d-spacings via the reciprocal metric tensor (one formula for all seven
#     crystal systems), 2theta <-> d conversion and reflection lists.
#
# Conventions: lengths in Angstrom, angles in degrees, unit cells as
# [a, b, c, alpha, beta, gamma]. Rhombohedral cells are used in the
# rhombohedral setting (a = b = c, alpha = beta = gamma).
#
# d-spacings: 1/d^2 = h^T G* h with G* = G^-1 the reciprocal metric tensor,
# G_ij = a_i . a_j (e.g. Giacovazzo et al., Fundamentals of Crystallography,
# 3rd ed., Oxford University Press, 2011, ch. 2).
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import math

import numpy as np


# ============================================================================
#                                CONSTANTS
# ============================================================================
RAD_TO_DEG = 180.0 / np.pi
DEG_TO_RAD = np.pi / 180.0

# Canonical upper-case names used everywhere inside EXODUS.
CRYSTAL_SYSTEMS = ('CUBIC', 'TETRAGONAL', 'HEXAGONAL', 'RHOMBOHEDRAL',
                   'ORTHORHOMBIC', 'MONOCLINIC', 'TRICLINIC')

# Accepted alternative spellings. 'ORTHOROMBIC' is the spelling written by
# EXODUS <= 0.1.3, so those JCPDS files keep loading.
_CRYSTAL_SYSTEM_ALIASES = {
    'ORTHOROMBIC':  'ORTHORHOMBIC',
    'ORTHORHOMIC':  'ORTHORHOMBIC',
    'ORTHO':        'ORTHORHOMBIC',
}

# Lattice-parameter names in unit-cell order (also the lmfit name stems).
LATTICE_NAMES = ('a', 'b', 'c', 'alp', 'bet', 'gam')

# Independent (refinable) lattice parameters per crystal system, in the order
# a b c alpha beta gamma. Every parameter that is not free is either tied to
# another one (TIED_LATTICE) or fixed at its symmetry value.
FREE_LATTICE = {
    'CUBIC':         (True,  False, False, False, False, False),
    'TETRAGONAL':    (True,  False, True,  False, False, False),
    'HEXAGONAL':     (True,  False, True,  False, False, False),
    'RHOMBOHEDRAL':  (True,  False, False, True,  False, False),
    'ORTHORHOMBIC':  (True,  True,  True,  False, False, False),
    'MONOCLINIC':    (True,  True,  True,  False, True,  False),
    'TRICLINIC':     (True,  True,  True,  True,  True,  True),
}

# Symmetry ties: parameter -> parameter it equals.
TIED_LATTICE = {
    'CUBIC':         {'b': 'a', 'c': 'a'},
    'TETRAGONAL':    {'b': 'a'},
    'HEXAGONAL':     {'b': 'a'},
    'RHOMBOHEDRAL':  {'b': 'a', 'c': 'a', 'bet': 'alp', 'gam': 'alp'},
    'ORTHORHOMBIC':  {},
    'MONOCLINIC':    {},
    'TRICLINIC':     {},
}


# ============================================================================
#                          CRYSTAL-SYSTEM HELPERS
# ============================================================================
def normalise_crystal_system(name):
    """Return the canonical upper-case crystal-system name for `name`.

    Strips whitespace, upper-cases, and maps known aliases / legacy
    misspellings (e.g. 'ORTHOROMBIC') onto the canonical spelling.
    Unknown names are returned upper-cased and stripped so the caller can
    still report them in an error message.
    """
    s = str(name).strip().upper()
    return _CRYSTAL_SYSTEM_ALIASES.get(s, s)


def free_lattice_flags(crystalSystem):
    """Tuple of 6 bools: which lattice parameters are independent."""
    return FREE_LATTICE.get(normalise_crystal_system(crystalSystem),
                            (True,) * 6)


def default_angles(crystalSystem):
    """Default (alpha, beta, gamma) for a crystal system.

    Used when a JCPDS file omits the angles (e.g. the abbreviated canonical
    format of cubic phases). Monoclinic and triclinic files are expected to
    give their angles explicitly.
    """
    if normalise_crystal_system(crystalSystem or '') == 'HEXAGONAL':
        return 90.0, 90.0, 120.0
    return 90.0, 90.0, 90.0


def fill_unit_cell(a, b, c, alpha, beta, gamma, crystalSystem):
    """Fill in missing (None) lattice parameters from the symmetry.

    Abbreviated JCPDS files often only give A (cubic) or A and C
    (hexagonal / tetragonal). This expands them to a full six-parameter
    cell so the rest of the code always sees the same shape.
    """
    sym = normalise_crystal_system(crystalSystem or '')
    if sym == 'CUBIC':
        b = a if b is None else b
        c = a if c is None else c
    elif sym in ('HEXAGONAL', 'TETRAGONAL'):
        b = a if b is None else b          # c must be given for these
    elif sym == 'RHOMBOHEDRAL':
        b = a if b is None else b
        c = a if c is None else c
        if alpha is not None:
            beta = alpha if beta is None else beta
            gamma = alpha if gamma is None else gamma
    # ORTHORHOMBIC, MONOCLINIC, TRICLINIC: a, b and c must all be given.

    if alpha is None or beta is None or gamma is None:
        da, db, dg = default_angles(sym)
        alpha = da if alpha is None else alpha
        beta = db if beta is None else beta
        gamma = dg if gamma is None else gamma
    return a, b, c, alpha, beta, gamma


def enforce_symmetry(unitCell, crystalSystem):
    """Return a copy of the unit cell with the symmetry ties applied
    (e.g. b = a and gamma = 120 for hexagonal)."""
    uc = [float(x) for x in unitCell]
    sym = normalise_crystal_system(crystalSystem)
    if sym == 'CUBIC':
        uc[1] = uc[2] = uc[0]
        uc[3] = uc[4] = uc[5] = 90.0
    elif sym in ('TETRAGONAL', 'HEXAGONAL'):
        uc[1] = uc[0]
        uc[3] = uc[4] = 90.0
        uc[5] = 120.0 if sym == 'HEXAGONAL' else 90.0
    elif sym == 'RHOMBOHEDRAL':
        uc[1] = uc[2] = uc[0]
        uc[4] = uc[5] = uc[3]
    elif sym == 'ORTHORHOMBIC':
        uc[3] = uc[4] = uc[5] = 90.0
    elif sym == 'MONOCLINIC':
        uc[3] = uc[5] = 90.0
    return uc


# ============================================================================
#                                 VOLUME
# ============================================================================
def unit_cell_volume(unitCell):
    """
    Volume (A^3) of a unit cell [a, b, c, alpha, beta, gamma].

    Returns None when the cell geometry is invalid (the angles do not form
    a closable parallelepiped, i.e. a negative radicand) or the input is
    not numeric. Callers must check for None before using the result.
    """
    try:
        a, b, c = float(unitCell[0]), float(unitCell[1]), float(unitCell[2])
        alp = math.radians(float(unitCell[3]))
        bet = math.radians(float(unitCell[4]))
        gam = math.radians(float(unitCell[5]))
    except (TypeError, ValueError, IndexError):
        return None
    radicand = (1 - math.cos(alp)**2 - math.cos(bet)**2 - math.cos(gam)**2
                + 2 * math.cos(alp) * math.cos(bet) * math.cos(gam))
    if not math.isfinite(radicand) or radicand <= 0:
        return None
    volume = a * b * c * math.sqrt(radicand)
    return volume if math.isfinite(volume) else None


# ============================================================================
#                           d-SPACING AND 2-THETA
# ============================================================================
def d_spacing_to_two_theta(d, WL):
    """d-spacing (A) -> 2theta (deg) for wavelength WL (A)."""
    return 2 * np.arcsin(WL / (2 * d)) * RAD_TO_DEG


def two_theta_to_d_spacing(twoTheta, WL):
    """2theta (deg) -> d-spacing (A) for wavelength WL (A).

    Returns +inf at 2theta = 0 (the limit d -> inf), which callers use as
    an open upper bound of a d-range; the divide-by-zero warning is
    suppressed on purpose.
    """
    with np.errstate(divide='ignore'):
        return 1 / (2 * np.sin(twoTheta / RAD_TO_DEG / 2) / WL)


def reciprocal_metric(unitCell):
    """Reciprocal metric tensor G* (3x3, A^-2) of a unit cell.

    Returns None if the cell is singular (invalid angles / zero lengths).
    """
    a, b, c = float(unitCell[0]), float(unitCell[1]), float(unitCell[2])
    al, be, ga = np.radians([float(unitCell[3]), float(unitCell[4]),
                             float(unitCell[5])])
    G = np.array([
        [a * a,              a * b * np.cos(ga), a * c * np.cos(be)],
        [a * b * np.cos(ga), b * b,              b * c * np.cos(al)],
        [a * c * np.cos(be), b * c * np.cos(al), c * c]])
    try:
        Gs = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        return None
    return Gs if np.all(np.isfinite(Gs)) else None


def hkl_stencil(HKL):
    """Pre-computed index products for fast vectorised 1/d^2 evaluation.

    Returns an (n, 6) array with columns h^2, k^2, l^2, 2hk, 2kl, 2hl so that
    1/d^2 = stencil @ [G*11, G*22, G*33, G*12, G*23, G*13].
    """
    hkl = np.asarray(HKL, dtype=np.float64)
    H, K, L = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    return np.column_stack((H * H, K * K, L * L, 2 * H * K, 2 * K * L, 2 * H * L))


def d_spacings(unitCell, stencil):
    """d-spacings (A) of all reflections of a `hkl_stencil` for a cell.

    Entries are NaN where 1/d^2 <= 0 or the cell is invalid.
    """
    Gs = reciprocal_metric(unitCell)
    if Gs is None:
        return np.full(stencil.shape[0], np.nan)
    g = np.array([Gs[0, 0], Gs[1, 1], Gs[2, 2], Gs[0, 1], Gs[1, 2], Gs[0, 2]])
    inv_d2 = stencil @ g
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(inv_d2 > 0, 1.0 / np.sqrt(np.abs(inv_d2)), np.nan)


def degenerate_groups(HKL, crystalSystem):
    """Groups of reflections that overlap EXACTLY for every cell of the
    crystal system (e.g. (300)/(221) in cubic): their d-spacings are equal
    for two unrelated test cells. Returns a list of index lists (rows of
    HKL), each with >= 2 members, in HKL order."""
    hkl = np.asarray(HKL, dtype=float)
    if hkl.shape[0] < 2:
        return []
    st = hkl_stencil(hkl)
    keys = []
    for cell in ([1.00, 1.13, 1.29, 83.0, 97.0, 105.0],
                 [1.07, 1.21, 1.41, 79.0, 101.0, 111.0]):
        c = enforce_symmetry(cell, crystalSystem)
        if normalise_crystal_system(crystalSystem) == 'RHOMBOHEDRAL':
            c[3] = c[4] = c[5] = cell[3]
        d = d_spacings(c, st)
        keys.append(np.round(1.0 / d ** 2, 9))
    groups = {}
    for i, key in enumerate(zip(*keys)):
        if all(np.isfinite(key)):
            groups.setdefault(key, []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def d_spacing_general(unitCell, h, k, l):
    """d(hkl) in A for an arbitrary cell via the reciprocal metric tensor.
    Returns NaN for an invalid cell or reflection."""
    return float(d_spacings(unitCell, hkl_stencil([[h, k, l]]))[0])


def d_and_twotheta(unitCell, crystalSystem, h, k, l, WL):
    """(d, 2theta) for one reflection of a cell (symmetry ties applied).
    d is None for an invalid cell / reflection; 2theta is None when WL is
    unknown or the reflection cannot be reached (d < WL/2)."""
    d = d_spacing_general(enforce_symmetry(unitCell, crystalSystem), h, k, l)
    if not np.isfinite(d) or d <= 0:
        return None, None
    tt = None
    if WL:
        s = float(WL) / (2.0 * d)
        if 0 < s <= 1:
            tt = 2.0 * math.degrees(math.asin(s))
    return d, tt


def reflection_list(unitCell, HKL, WL, twoThetaMin=5, twoThetaMax=30,
                    crystalSystem='CUBIC'):
    '''
    Reflection list of ONE phase inside the 2theta window.

    Returns (dk_reflection_list, twoTheta_reflection_list), both of shape
    (n_visible, 4) with columns [d or 2theta, H, K, L], in the order of the
    HKL array. The symmetry ties of `crystalSystem` are applied to the cell
    first, so only the independent lattice parameters matter.
    '''
    HKL_arr = np.asarray(HKL, dtype=np.float64)
    if HKL_arr.size == 0:
        empty = np.zeros((0, 4))
        return empty, empty
    cell = enforce_symmetry(unitCell, crystalSystem)
    dk = d_spacings(cell, hkl_stencil(HKL_arr))

    d_min = two_theta_to_d_spacing(twoThetaMax, WL)
    d_max = two_theta_to_d_spacing(twoThetaMin, WL)
    mask = np.isfinite(dk) & (dk > d_min) & (dk < d_max)

    dk_in = dk[mask]
    twoTh_in = d_spacing_to_two_theta(dk_in, WL)
    H_in, K_in, L_in = HKL_arr[mask, 0], HKL_arr[mask, 1], HKL_arr[mask, 2]

    twoTheta_reflection_list = np.column_stack((twoTh_in, H_in, K_in, L_in))
    dk_reflection_list = np.column_stack((dk_in, H_in, K_in, L_in))
    return dk_reflection_list, twoTheta_reflection_list
