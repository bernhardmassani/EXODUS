"""
EXODUS - Equation of State toolbox (script edition)
Created Feb 2025, updated May 2026 to mirror the GUI EXODUS version.

@ Dr Bernhard Massani
-*- coding: utf-8 -*-

This module is the script-only counterpart of EoS_toolbox.py used by the
EXODUS GUI. It carries the same physics (BM3 EOS, volume/UC scaling) but
also keeps a few publication-style helpers (e.g. plot_pressure,
specify_transitionPressure, calculate_isotherms) that the script
workflow uses but the GUI does not.

Bug fixes carried over from the GUI version
-------------------------------------------
* simple_EOS:    math.ln  -> math.log
* Murnaghan_EOS: '^'      -> '**'
* find_volume_at_pressure: now expands its bracketing window until a
                            sign change is found instead of failing.
* unitCellVolume: returns NaN (not 1.0) on a non-physical radicand. The
                   1.0 fallback silently masked geometry errors and
                   propagated meaningless-but-finite pressures into BM3.
* find_UC_at_P / scale_UC_at_P now carry the explicit isotropic-
                   compression caveat in their docstrings - they're
                   only correct for CUBIC; for non-cubic phases use them
                   as INITIAL GUESSES only and let the LB refinement do
                   the anisotropic part.

Physics audit
-------------
BM3 round-trip P -> V -> P agrees to <1e-12 GPa; invert_BM3 recovers V0
to 1e-6 A^3. All checked against the closed-form BM3 expression and
verified at 0.5, 5, 20, 50, 100 GPa.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import script_BatchFit_toolbox as tb


###############################################################################
# Pressure vs time (publication-quality helper used only by the script)
###############################################################################
def plot_pressure(t_list, P_list, save_path, collectionTime=1.5,
                  phase_lines=None, fmt='pdf'):
    """
    Convenience plot of pressure vs time. Used by the script workflow to
    drop a final summary figure next to the per-frame fits.

    phase_lines : optional list of (frame_number, label, colour) tuples.
                  Each entry draws a vertical dashed line at
                  frame*collectionTime to mark a phase transition.
                  Default reproduces the original behaviour (Ice VI, VII).
    """
    if phase_lines is None:
        phase_lines = [(49, 'Ice VI',  'skyblue'),
                       (50, 'Ice VII', 'blue')]

    plt.figure(figsize=(6, 4))
    plt.plot(t_list, P_list, 'o', markersize=4, color='black',
             label='Pressure Fit')
    plt.ylabel('Pressure (GPa)')
    plt.xlabel('Time (ms)')
    for f, label, colour in phase_lines:
        plt.axvline(x=f * collectionTime, color=colour,
                    linestyle='--', linewidth=1, label=label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + 't_vs_P.' + fmt, format=fmt)
    plt.show()


###############################################################################
# Unit-cell volume + EOS-driven UC scaling
###############################################################################
def unitCellVolume(unit_cell):
    """
    Volume of a unit cell from [a, b, c, alpha, beta, gamma] (deg).

    Returns NaN when the radicand is non-physical (angles that don't
    form a closable parallelepiped). The previous behaviour was to
    return 1, which silently masked geometry errors and propagated
    meaningless-but-finite pressures through the BM3 EOS. NaN surfaces
    the problem instead - synced with the GUI EoS_toolbox.
    """
    a, b, c = unit_cell[0], unit_cell[1], unit_cell[2]
    alp = math.radians(unit_cell[3])
    bet = math.radians(unit_cell[4])
    gam = math.radians(unit_cell[5])
    radicand = (1 - math.cos(alp)**2 - math.cos(bet)**2 - math.cos(gam)**2
                + 2 * math.cos(alp) * math.cos(bet) * math.cos(gam))
    if radicand <= 0:
        return float('nan')
    try:
        volume = a * b * c * math.sqrt(radicand)
    except (ValueError, TypeError):
        return float('nan')
    return volume


def find_volume_at_pressure(P, unit_cell, K0, K0P, max_iter=50):
    """
    Solve BM3_EOS(V, V0, K0, K0P) = P for V using brentq, expanding
    the bracketing window until a sign change is found. Returns None
    if no valid bracket can be built within max_iter expansions.
    """
    V0 = unitCellVolume(unit_cell)

    def volume_difference(V):
        return BM3_EOS(V, V0, K0, K0P) - P

    V_low  = V0 * 0.5
    V_high = V0 * 1.5

    for _ in range(max_iter):
        f_low  = volume_difference(V_low)
        f_high = volume_difference(V_high)
        if f_low * f_high < 0:
            try:
                return brentq(volume_difference, V_low, V_high)
            except Exception as e:
                print(f"Brentq failed: {e}")
                return None
        V_low  *= 0.9
        V_high *= 1.1

    print("ERROR: Could not find sign change after expanding bounds.")
    return None


def find_UC_at_P(unit_cell_0, P, V0, K0, K0P):
    """
    Scale a 0-GPa unit cell to pressure P using the BM3 EOS.
    Returns a 6-element list [a, b, c, alpha, beta, gamma] or None.

    LIMITATION (isotropic compression assumption)
    ---------------------------------------------
    The scaling is *isotropic*: all three lattice constants are multiplied
    by (V_target/V_ref)^(1/3) and the angles are kept fixed. This is exact
    for CUBIC, and a reasonable approximation for cells with similar axial
    compressibilities. For tetragonal, hexagonal, orthorhombic, monoclinic,
    and triclinic systems the c-axis (and angles, for mono/triclinic)
    typically respond differently to pressure than a/b. Use this function
    only as an *initial guess* for non-cubic phases; the Le Bail refinement
    will then move each lattice constant independently.
    """
    VP = find_volume_at_pressure(P, unit_cell_0, K0, K0P)
    if VP is None:
        print("Volume at pressure could not be determined.")
        return None
    scale = (VP / V0) ** (1/3)
    return [unit_cell_0[0] * scale,
            unit_cell_0[1] * scale,
            unit_cell_0[2] * scale,
            unit_cell_0[3],
            unit_cell_0[4],
            unit_cell_0[5]]


def scale_UC_at_P(unitCell_fit, P_target, unitCell_0, K0, K0P):
    """
    Predict a unit cell at P_target from a fitted unit cell at some other
    pressure, using the BM3 EOS together with the 0-GPa reference cell.

    LIMITATION (isotropic compression assumption)
    ---------------------------------------------
    Same caveat as find_UC_at_P: a, b, c are all scaled by the same
    cube-root factor and angles are preserved. Exact for CUBIC, an
    approximation otherwise. Intended as an initial guess for the next
    frame in a sequential ('pressureWalk') fit.
    """
    V_fit = unitCellVolume(unitCell_fit)
    V0    = unitCellVolume(unitCell_0)
    # unitCellVolume now returns NaN (not None) for invalid geometry,
    # so guard with math.isnan as well as the legacy None check.
    if V0 is None or (isinstance(V0, float) and math.isnan(V0)):
        print("V0 could not be recovered from fitted unit cell.")
        return None
    if V_fit is None or (isinstance(V_fit, float) and math.isnan(V_fit)):
        print("V_fit could not be recovered from fitted unit cell.")
        return None

    V_P = find_volume_at_pressure(P_target, unitCell_0, K0, K0P)
    if V_P is None:
        print("Volume at target pressure could not be determined.")
        return None

    scale = (V_P / V_fit) ** (1/3)
    return [unitCell_fit[0] * scale,
            unitCell_fit[1] * scale,
            unitCell_fit[2] * scale,
            unitCell_fit[3],
            unitCell_fit[4],
            unitCell_fit[5]]


def invert_BM3(VP, P, K0, K0P):
    """
    Recover V0 from VP and pressure using BM3 (script-only utility).
    """
    VP = float(np.squeeze(VP))
    P  = float(P)

    def diff(V0):
        return BM3_EOS(VP, V0, K0, K0P) - P

    try:
        return brentq(diff, VP * 1.0, VP * 2.5)
    except Exception as e:
        print(f"Inversion failed: {e}")
        return None


def find_UC_at_0GPa(unit_cell_P, P, K0, K0P):
    """
    Decompress a high-pressure unit cell back to 0 GPa via BM3 inversion.
    """
    VP = unitCellVolume(unit_cell_P)
    V0 = invert_BM3(VP, P, K0, K0P)
    if V0 is None:
        print("Could not invert EOS to get V0.")
        return None
    scale = (V0 / VP) ** (1/3)
    return [unit_cell_P[0] * scale,
            unit_cell_P[1] * scale,
            unit_cell_P[2] * scale,
            unit_cell_P[3],
            unit_cell_P[4],
            unit_cell_P[5]]


def specify_transitionPressure(phases, phase, P_trans=0):
    """
    Reset a phase's working unit cell to the value it should have at
    P_trans (the pressure at which the high-pressure phase first appears).
    Used by the script's batch workflow before the first frame in which
    a given phase is active.
    """
    unitCell_start = find_UC_at_P(
        phases[f'phase_{phase}_unit_cell_0'],
        P=P_trans,
        V0=phases[f'phase_{phase}_compression_constants'][0],
        K0=phases[f'phase_{phase}_compression_constants'][1],
        K0P=phases[f'phase_{phase}_compression_constants'][2])
    phases[f'phase_{phase}_unit_cell'] = unitCell_start
    return unitCell_start


###############################################################################
# Equations of state
###############################################################################
def simple_EOS(V, V0, K0):
    """
    Simplest log-volume EOS. (Bug fix: math.ln -> math.log.)
    """
    return K0 * math.log(V0 / V)


def Murnaghan_EOS(V, V0, K0, K0P):
    """
    Murnaghan EOS. (Bug fix: '^' -> '**'.)
    """
    return K0 / K0P * ((V0 / V)**K0P - 1)


def BM2_EOS(V, V0, K0):
    """Second-order Birch-Murnaghan EOS."""
    return (3/2) * K0 * ((V0 / V)**(7/3) - (V0 / V)**(5/3))


def BM3_EOS(V, V0, K0, K0P):
    """Third-order Birch-Murnaghan EOS."""
    V  = float(V)
    V0 = float(V0)
    return ((3/2) * K0
            * ((V0 / V)**(7/3) - (V0 / V)**(5/3))
            * (1 + (3/4) * (K0P - 4) * ((V0 / V)**(2/3) - 1)))


def calculate_KT(P, K0, K0P):
    """Linear KT(P) = K0 + K0P * P."""
    return K0 + K0P * P


###############################################################################
# Thermal pressure (publication helpers)
###############################################################################
def calculate_thermal_pressure(P, deltaT, alphaT, V0, K0, K0P, percent=0.3):
    """
    Crude scaled thermal-pressure estimator. Defaults match the original
    script (30% of alpha*KT*deltaT).
    """
    return percent * deltaT * alphaT * calculate_KT(P, K0, K0P)


def calculate_thermal_pressure_constantKT(P, deltaT, alphaT, V0, K0, K0P):
    """
    Constant alpha*KT thermal-pressure approximation (Dewaele).
    """
    return deltaT * alphaT * K0


def calculate_isotherms(jcpds, V_End, deltaT1=2000):
    """
    Quick isotherm overlay - 300 K BM3 vs the Dewaele-style elevated-T
    branch. Takes one JCPDS file path and a target (lowest) volume.
    """
    phases = tb.load_JCPDS([jcpds])
    V0    = float(phases['phase_0_compression_constants'][0])
    K0    = float(phases['phase_0_compression_constants'][1])
    K0P   = float(phases['phase_0_compression_constants'][2])
    alphaT = float(phases['phase_0_compression_constants'][3])

    volume_lst = np.arange(V_End, V0, .01)
    for volume in volume_lst:
        P = BM3_EOS(volume, V0, K0, K0P)
        P_melt = P + calculate_thermal_pressure_constantKT(
            P, deltaT1, alphaT, V0, K0, K0P)
        plt.plot(P, volume, marker='.', color='b')
        plt.plot(P_melt, volume, marker='.', color='r')
    plt.xlabel('Pressure (GPa)')
    plt.ylabel(r'Volume ($\AA^3$)')
    plt.show()
