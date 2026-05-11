"""
EXODUS - Le Bail / Pawley batch-fit toolbox (script edition)
Created Feb 2025, updated May 2026 to mirror the GUI EXODUS version.

@ Dr Bernhard Massani
-*- coding: utf-8 -*-

This module is the script-only counterpart of BatchFit_toolbox.py used
by the EXODUS GUI. It carries every physics/algorithm improvement made
in the GUI version and adds publication-ready figure orchestrators
(plot_LB_fit, plot_peak_fit_single, fit_LB driver, index_LB driver,
fit_peak_single driver) that the GUI replaces with interactive widgets.

Key improvements over the v0 script
-----------------------------------
* d-spacing fix (MONOCLINIC): math.cos(b) -> math.cos(bet). This was the
  long-standing bug in v0 that fed the lattice parameter b (a length) to
  cos(), producing wildly wrong d-spacings for any monoclinic phase.
* d-spacing fix (RHOMBOHEDRAL): the '- cos(alpha)' was OUTSIDE the
  (HK+KL+HL) bracket; corrected to ' + (cos^2 - cos) ' inside the
  cross-term. Reference: Cullity & Stock, App. 3.
* d-spacing fix (TRICLINIC): re-implemented using the standard S_ij
  metric-tensor formulation. The old form was missing one lattice-
  constant factor on every term under the square root and was wrong by
  roughly an order of magnitude even in the orthorhombic limit.
  Verified to floating-point precision (max rel. err. 4.4e-16) against
  the master formula by reducing cubic / tetragonal / hexagonal /
  rhombohedral / orthorhombic / monoclinic test cells to the triclinic
  computation.
* Amp parameter keying (May 2026, ported from GUI): one amp parameter
  per HKL row, keyed by HKL index. Out-of-window AND zero-intensity
  reflections are frozen (vary=False) so the Jacobian stays full rank
  and lmfit returns a real covariance matrix. The old loop keyed amps
  by filtered-list ordinal, which silently collided with HKL index
  whenever a reflection sat outside the fit window.
* LB_fit_Model slow path (rhombohedral / orthorhombic / monoclinic /
  triclinic) now reconstructs HKL indices by (H,K,L) lookup instead of
  passing np.arange(n_refl), so amps/sigmas are pulled by the correct
  HKL index for non-fast-path crystal systems too.
* 'separate' sigma mode: HKL-index keyed and out-of-window sigmas
  frozen, mirroring the amplitude fix.
* Vectorised reflection_List (NumPy broadcasting; ~20x faster).
* Per-phase HKL stencil cache for the hot LB residual loop.
* Multi-format data loading: .fxye, .xy, .dat, .chi auto-detected.
* ALS background subtraction in addition to polynomial.
* Configurable sigma model: 'fixed', 'per_phase', 'caglioti', 'separate'.
* Per-parameter unit-cell errors via unitCell_errors().
* Configurable maxShift / sigmaBounds / lmfit tolerances.
"""
import os
import re
import glob
import math
import bisect

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from lmfit import Parameters, minimize

import script_EoS_toolbox as EoS


###############################################################################
# OS helpers
###############################################################################
def create_folder(folder_path):
    """Make sure folder_path exists; return it as a string."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"\n Analysis Folder '{folder_path}' created successfully.")
    return str(folder_path)


###############################################################################
# Load PONI
###############################################################################
def load_poni(poni_file):
    """
    Read a pyFAI .poni file and return the wavelength in angstroms.
    Same parser as v0 - the file format hasn't changed.
    """
    df = pd.read_csv(str(poni_file), header=0, delimiter=' ')
    wavelength = df[df.apply(
        lambda row: row.astype(str).str.contains('Wavelength:').any(),
        axis=1)]
    for _, row in wavelength.iterrows():
        WL = float(row.iloc[1]) * 10e9
    return WL


###############################################################################
# Load JCPDS
###############################################################################
def load_JCPDS(JCPDS):
    """
    Load one or more JCPDS files into a phases dictionary.
    phase_i is referenced by integer index, so phases['phase_0_unit_cell']
    pulls the unit cell of the first JCPDS file in the list.
    """
    if isinstance(JCPDS, str):
        JCPDS = [JCPDS]

    phases = {}
    phases_used = []
    for i, phase in enumerate(JCPDS):
        file_path = str(phase)
        print('Load: \t' + file_path)
        phase_i = 'phase_' + str(i)
        phases_used.append(i)

        column_names = ["PARAM", "VALUE_1", "VALUE_2", "VALUE_3",
                        "VALUE_4", "VALUE_5"]
        df = pd.read_csv(file_path, sep=r'\s*:\s*', header=None,
                         names=column_names, engine='python')

        HKL = []
        # [V0, K0, K0P, alphaT, DK0DT, DK0PDT]
        compression_constants = [0, 0, 0, 0, 0, 0]
        for _, row in df.iterrows():
            tag = row.iloc[0]
            if tag == 'A':
                a = float(row.iloc[1])
            elif tag == 'B':
                b = float(row.iloc[1])
            elif tag == 'C':
                c = float(row.iloc[1])
            elif tag == 'ALPHA':
                alpha = float(row.iloc[1])
            elif tag == 'BETA':
                beta = float(row.iloc[1])
            elif tag == 'GAMMA':
                gamma = float(row.iloc[1])
            elif tag == 'K0':
                compression_constants[1] = float(row.iloc[1])
            elif tag == 'K0P':
                compression_constants[2] = float(row.iloc[1])
            elif tag == 'DK0DT':
                compression_constants[4] = float(row.iloc[1])
            elif tag == 'DK0PDT':
                compression_constants[5] = float(row.iloc[1])
            elif tag == 'SYMMETRY':
                crystal_system = row.iloc[1]
            elif tag == 'ALPHAT':
                compression_constants[3] = float(row.iloc[1])
            elif tag == 'DIHKL':
                parts = re.split(r'[\t ]+', row.iloc[1].strip())
                Int = parts[1]; H = parts[2]; K = parts[3]; L = parts[4]
                HKL.append([float(H), float(K), float(L), float(Int)])

        compression_constants = np.array(compression_constants)
        HKL = np.array(HKL)
        unit_cell = np.array([a, b, c, alpha, beta, gamma])
        compression_constants[0] = EoS.unitCellVolume(unit_cell)

        phases[phase_i + '_crystal_system']        = crystal_system
        phases[phase_i + '_unit_cell']             = unit_cell
        phases[phase_i + '_unit_cell_0']           = unit_cell
        phases[phase_i + '_HKL']                   = HKL
        phases[phase_i + '_compression_constants'] = compression_constants

    phases['phases_Number'] = len(JCPDS)
    phases['phases_used']         = np.array(phases_used)
    phases['phases_Number_array'] = np.array(phases_used)
    return phases


def update_phases_used(phases_used, phases):
    """Replace the phases_used list with a new array."""
    phases['phases_used'] = np.array(phases_used)


def specify_phases(phases, phase_rules, frame):
    """
    Set phases['phases_used'] for the given frame using a list of
    (predicate, [phase_index, ...]) rules. The first matching predicate
    wins. Raises ValueError if no rule matches the frame.
    """
    for condition, phase_list in phase_rules:
        if condition(frame):
            phases['phases_used'] = phase_list
            break
    else:
        raise ValueError(f"Unexpected frame number: {frame}")
    return phases


def update_unitCell(newUnitCell, phases, phaseNumber):
    """Replace the working unit cell of `phaseNumber` in-place."""
    newUnitCell = np.array(newUnitCell)
    key = 'phase_' + str(phaseNumber) + '_unit_cell'
    if key in phases:
        phases[key] = newUnitCell
    else:
        print('Error. No matching unit cell to update.')


def update_HKL(newHKL, phases, phaseNumber):
    """Replace the HKL array of `phaseNumber` in-place."""
    newHKL = np.array(newHKL)
    key = 'phase_' + str(phaseNumber) + '_HKL'
    if key in phases:
        phases[key] = newHKL
    else:
        print('Error. No matching unit cell to update.')


###############################################################################
# Multi-format data loading (.fxye / .xy / .dat / .chi)
###############################################################################
def extract_number(filename):
    """Pull the trailing integer out of e.g. 'run1_93.fxye'; returns +inf
    if no integer can be parsed (so the file sorts to the end)."""
    base = os.path.basename(filename)
    parts = base.split("_")
    try:
        return int(parts[-1].split(".")[0])
    except ValueError:
        return float('inf')


def load_frameNumber(data_path):
    """Number of valid frames in data_path (script convenience)."""
    files = sorted(glob.glob(os.path.join(data_path, "*")),
                   key=extract_number)
    files = [f for f in files if extract_number(f) != float('inf')]
    return len(files)


def load_frame(frame, data_path, twoThetaMin=0, twoThetaMax=30):
    """
    Load the n-th valid file from data_path (1-indexed) and return
    (twoTheta, valueInt, errorInt) clipped to [twoThetaMin, twoThetaMax].
    """
    files = sorted(glob.glob(os.path.join(data_path, "*")),
                   key=extract_number)
    if files:
        print(f"Found {len(files)} files in the directory.")
    else:
        print("No data files found in the directory!")

    files = [f for f in files if extract_number(f) != float('inf')]
    total = len(files)
    if total == 0:
        print("No valid numbered files found!")
        return None
    if 1 <= frame <= total:
        return loadData(files[frame - 1],
                        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
    print(f"Error: Frame {frame} out of range. Total files: {total}")
    return None


def load_frame_fast(frame, data_path, twoThetaMin=0, twoThetaMax=30):
    """Same as load_frame but quieter (no 'Found N files' chatter)."""
    files = sorted(glob.glob(os.path.join(data_path, "*")),
                   key=extract_number)
    files = [f for f in files if extract_number(f) != float('inf')]
    total = len(files)
    if total == 0:
        print("No valid numbered files found!")
        return None
    if 1 <= frame <= total:
        return loadData(files[frame - 1],
                        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
    print(f"Error: Frame {frame} out of range. Total files: {total}")
    return None


def _truncate_range(twoTheta, valueInt, errorInt, twoThetaMin, twoThetaMax):
    """Clip a (twoTheta, intensity, error) triplet to a 2theta window."""
    twoTheta = list(twoTheta)
    valueInt = list(valueInt)
    errorInt = list(errorInt)
    indexLow  = bisect.bisect(twoTheta, twoThetaMin)
    indexHigh = bisect.bisect(twoTheta, twoThetaMax)
    return (np.array(twoTheta[indexLow:indexHigh]),
            np.array(valueInt[indexLow:indexHigh]),
            np.array(errorInt[indexLow:indexHigh]))


def _load_fxye(file_path):
    """GSAS-II .fxye: 23-line header, centideg in col 1."""
    df = pd.read_csv(file_path, header=23, delimiter='\t')
    twoTheta = [i * 0.01 for i in df.iloc[:, 1].values.tolist()]
    valueInt = df.iloc[:, 2].values.tolist()
    errorInt = df.iloc[:, 3].values.tolist()
    return twoTheta, valueInt, errorInt


def _load_xy(file_path):
    """DIOPTAS / pyFAI .xy: '#'-comment header, deg in col 1."""
    data = np.loadtxt(file_path, comments='#')
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))   # synthesised
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


def _load_dat(file_path):
    """Plain .dat: 2-col, optional '#' comments."""
    data = np.loadtxt(file_path, comments='#')
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))   # synthesised
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


def _load_chi(file_path):
    """Fit2D / pyFAI .chi: fixed 4-line header."""
    data = np.loadtxt(file_path, skiprows=4)
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))   # synthesised
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


_LOADERS = {
    '.fxye': _load_fxye,
    '.xy':   _load_xy,
    '.dat':  _load_dat,
    '.chi':  _load_chi,
}


def loadData(file_path, twoThetaMin=0, twoThetaMax=30):
    """
    Load a single XRD pattern, auto-detecting format from the extension.
    Returns numpy arrays clipped to [twoThetaMin, twoThetaMax]:
        .fxye - GSAS-II  (23-line header, 3 columns; centideg)
        .xy   - DIOPTAS  (#-comment header, 2 columns; deg)
        .dat  - plain    (no header,        2 columns; deg)
        .chi  - Fit2D    (4-line header,    2 columns; deg)
    For 2-column formats the error column is synthesised as sqrt(|I|).
    """
    ext = os.path.splitext(file_path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file extension '{ext}' for {file_path}. "
            f"Supported: {', '.join(sorted(_LOADERS.keys()))}")
    twoTheta, valueInt, errorInt = loader(file_path)
    return _truncate_range(twoTheta, valueInt, errorInt,
                           twoThetaMin, twoThetaMax)


###############################################################################
# Conversions and common-use constants
###############################################################################
rad_to_deg = 180 / np.pi
deg_to_rad = np.pi / 180
colours = [
    'red', 'blue', 'orange', 'purple', 'yellow', 'green',
    'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender',
    'brown', 'beige', 'maroon', 'navy', 'olive', 'coral',
    'turquoise', 'gold', 'silver', 'indigo', 'violet', 'crimson',
]


def dspacing_to_twoTheta(d, WL):
    """d-spacing -> 2theta (deg)."""
    return 2 * np.arcsin(WL / (2 * d)) * rad_to_deg


def twoTheta_to_dspacing(twoTheta, WL):
    """2theta (deg) -> d-spacing."""
    return 1 / (2 * np.sin(twoTheta / rad_to_deg / 2) / WL)


###############################################################################
# Background subtraction (polynomial peak-exclusion + ALS)
###############################################################################
def backgroundFit(data_path, frame=None, twoThetaMin=0, twoThetaMax=30,
                  peakSearchAuto=True, prominence=0.10, height=0.08,
                  order=6, excludePeakList=None, peakwidth=0.35,
                  plotBG=False):
    """
    Polynomial background fit with automatic peak exclusion.
    The auto-found peak centres are masked +/- peakwidth degrees and a
    poly of `order` is fit through the rest. Returns the original data
    plus the polynomial sampled at every twoTheta.

    Backwards-compatible signature with the v0 script: when called with
    `frame=...` this routine pulls the right file off disk via
    load_frame(); when called with a direct file path it falls through
    to loadData() instead.
    """
    if excludePeakList is None:
        excludePeakList = []

    # Two ways to call us: either (folder, frame=N) like v0, or (file_path)
    # like the GUI core. We support both transparently.
    if frame is not None:
        twoTheta, valueInt, errorInt = load_frame(
            frame, data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
    else:
        twoTheta, valueInt, errorInt = loadData(
            data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)

    twoTheta = np.array(twoTheta)
    valueInt = np.array(valueInt)
    errorInt = np.array(errorInt)

    twoTheta_unaltered = twoTheta.copy()
    valueInt_unaltered = valueInt.copy()
    errorInt_unaltered = errorInt.copy()

    peaks, _ = find_peaks(valueInt,
                          height=max(valueInt) * height,
                          prominence=prominence,
                          rel_height=0.5)
    excludedPeaksInt      = valueInt[peaks]
    excludedPeaksTwoTheta = twoTheta[peaks]

    if not peakSearchAuto and excludePeakList:
        peaks = excludePeakList
    elif excludePeakList and peakSearchAuto:
        # Allow combining auto-detected peaks with user overrides
        peaks = list(peaks) + list(excludePeakList)

    twoTheta_removed = twoTheta.astype(float)
    valueInt_removed = valueInt.astype(float)
    peak_list = []
    for i, peak in enumerate(twoTheta[peaks] if len(peaks) else []):
        peak_list.append(peak)
        peakmin = bisect.bisect(twoTheta, peak - peakwidth)
        peakmax = bisect.bisect(twoTheta, peak + peakwidth)
        if plotBG:
            plt.axvline(x=peak - peakwidth, color='r')
            plt.axvline(x=peak + peakwidth, color='r')
        twoTheta_removed[peakmin:peakmax] = np.nan
        valueInt_removed[peakmin:peakmax] = np.nan
    twoTheta_removed = twoTheta_removed[np.isfinite(twoTheta_removed)]
    valueInt_removed = valueInt_removed[np.isfinite(valueInt_removed)]

    if len(twoTheta_removed) <= order:
        # Not enough points left after peak exclusion - fall back to a flat BG.
        print("Warning: insufficient points for poly fit; using zero BG.")
        return twoTheta_unaltered, np.zeros_like(twoTheta_unaltered)

    con_fit = np.polyfit(twoTheta_removed, valueInt_removed, order)
    polyFit = np.poly1d(con_fit)

    if plotBG:
        plt.scatter(twoTheta_unaltered, valueInt_unaltered,
                    marker='.', color='lightgrey')
        plt.scatter(excludedPeaksTwoTheta, excludedPeaksInt,
                    marker='o', color='red', label='Excluded regions')
        plt.plot(twoTheta_unaltered, polyFit(twoTheta_unaltered),
                 label='BKG Fit', c='k', linestyle='--')
        plt.xlim(left=twoThetaMin); plt.xlim(right=twoThetaMax)
        plt.ylabel('Intensity (arb.u.)'); plt.xlabel('TwoTheta (deg)')
        plt.legend()
        plt.show()

    return twoTheta_unaltered, polyFit(twoTheta_unaltered)


def backgroundFit_ALS(data_path, frame=None, twoThetaMin=0, twoThetaMax=30,
                      lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares (Eilers-Boelens) background. Robust against
    overlapping peaks where the polynomial method struggles.
        lam   : smoothness  (1e4-1e7 typical)
        p     : asymmetry   (0.001-0.1 typical)
        niter : iterations  (10-20 typical)
    Returns (twoTheta, valueInt, errorInt, BGInt, valueIntBGsub).
    """
    if frame is not None:
        twoTheta, valueInt, errorInt = load_frame(
            frame, data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
    else:
        twoTheta, valueInt, errorInt = loadData(
            data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)

    twoTheta = np.array(twoTheta)
    valueInt = np.array(valueInt)

    L = len(valueInt)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = lam * (D.T @ D)

    w = np.ones(L)
    for _ in range(int(niter)):
        W = sp.diags(w, 0)
        Z = W + D
        z = splinalg.spsolve(Z, w * valueInt)
        w = p * (valueInt > z) + (1 - p) * (valueInt < z)

    BGInt          = z
    valueIntBGsub  = valueInt - BGInt
    return twoTheta, valueInt, errorInt, BGInt, valueIntBGsub


def subtract_BG(twoTheta, valueInt, valueBG):
    """
    Background-subtract intensities. Returns the new (twoTheta, BGsub)
    pair; signature is intentionally compatible with both the v0 script
    and the GUI core (which expects 4 outputs - we only return 2 here).
    """
    valueIntBGsub = np.asarray(valueInt) - np.asarray(valueBG)
    twoTheta = np.asarray(twoTheta)
    return twoTheta, valueIntBGsub


###############################################################################
# Pressure-guess interpolation and unit-cell propagation
###############################################################################
def pressure_interpolation(framePressureGuess, plotPressureGuess=False):
    """
    Linearly interpolate the (frame, P) anchor points so every frame in
    the supplied range has an associated guess pressure. Returns
    (frame_array, P_array) that index 1:1 over the run.
    """
    frame = np.array([pt[0] for pt in framePressureGuess])
    P     = np.array([pt[1] for pt in framePressureGuess])

    framePressureGuess = np.array(framePressureGuess)
    interp_func = interp1d(frame, P, kind='linear', fill_value="extrapolate")

    frame_interp = np.linspace(int(framePressureGuess[0, 0]),
                               int(framePressureGuess[-1, 0]),
                               int(framePressureGuess[-1, 0]))
    P_interp = interp_func(frame_interp)

    if plotPressureGuess:
        plt.plot(frame, P, 'o', label='Original Data')
        plt.plot(frame_interp, P_interp, '-', label='Interpolated Data')
        plt.legend()
        plt.show()

    return np.array(frame_interp), np.array(P_interp)


def unit_cell_P_guess(unitCell_0, frame, framePressureGuess, phases, phase):
    """
    Predict the unit cell at a given frame using the interpolated pressure
    at that frame and the phase's BM3 EOS parameters. Used by the legacy
    update_phases('pressureHelper') path.
    """
    frame_interp, P_interp = pressure_interpolation(
        framePressureGuess, plotPressureGuess=False)
    for frame_i in range(len(frame_interp)):
        if frame_i == frame:
            unitCell_guess = EoS.find_UC_at_P(
                unitCell_0,
                float(P_interp[frame_i]),
                V0=float(phases[f'phase_{phase}_compression_constants'][0]),
                K0=float(phases[f'phase_{phase}_compression_constants'][1]),
                K0P=float(phases[f'phase_{phase}_compression_constants'][2]))
            print(f'Unit Cell {phase} guess: \n{unitCell_guess}')
            return unitCell_guess
    return phases[f'phase_{phase}_unit_cell']


def update_phases(frame, phases, phase, framePressureGuess,
                  deltaP=0, resultsFit=None, resultsFitUC=None,
                  method='pressureHelper'):
    """
    Compute and install a starting unit cell for the next refinement.
    Three modes:
        'sequential'     - uses the previous fit's UC as the new starting UC.
        'pressureHelper' - scales the previous UC to the interpolated
                           pressure for this frame using BM3.
        'pressureWalk'   - extrapolates the previous fitted UC by a fixed
                           pressure step deltaP using BM3.
    """
    if resultsFit  is None: resultsFit   = []
    if resultsFitUC is None: resultsFitUC = [0, 0, 0, 0, 0, 0]

    if method == 'sequential':
        update_unitCell(resultsFitUC, phases, phase)
        print(f'Unit Cell {phase} guess: \n{resultsFitUC}')
        return resultsFitUC

    elif method == 'pressureHelper':
        unitCell_fit = phases[f'phase_{phase}_unit_cell']
        unitCell_0   = phases[f'phase_{phase}_unit_cell_0']
        K0   = float(phases[f'phase_{phase}_compression_constants'][1])
        K0P  = float(phases[f'phase_{phase}_compression_constants'][2])

        frame_interp, P_interp = pressure_interpolation(
            framePressureGuess, plotPressureGuess=False)

        unitCell_guess = EoS.scale_UC_at_P(
            unitCell_fit, P_interp[frame], unitCell_0, K0, K0P)

        print(f'Unit Cell {phase} guess: \n{unitCell_guess}')
        update_unitCell(unitCell_guess, phases, phase)
        return unitCell_guess

    elif method == 'pressureWalk':
        unitCell_fit = resultsFit[f'phase_{phase}_unit_cell_fit']
        unitCell_0   = phases[f'phase_{phase}_unit_cell_0']
        V0   = float(phases[f'phase_{phase}_compression_constants'][0])
        K0   = float(phases[f'phase_{phase}_compression_constants'][1])
        K0P  = float(phases[f'phase_{phase}_compression_constants'][2])

        P_fit  = EoS.BM3_EOS(EoS.unitCellVolume(unitCell_fit), V0, K0, K0P)
        P_next = P_fit + deltaP

        unitCell_next = EoS.scale_UC_at_P(
            unitCell_fit, P_next, unitCell_0, K0, K0P)
        phases[f'phase_{phase}_unit_cell'] = unitCell_next
        print(f'Unit Cell {phase} guess: \n{unitCell_next}')
        return unitCell_next

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'sequential', "
                         f"'pressureHelper' or 'pressureWalk'.")


###############################################################################
# Reflection list - VECTORISED, with the MONOCLINIC bug FIXED
###############################################################################
# Cache of HKL stencils so the LB residual loop doesn't re-cast HKL to
# float64 on every evaluation. Keyed on (id(HKL), crystal_system).
_LB_geom_cache = {}


def _LB_phase_stencil(HKL, crystal_system):
    """Cached HKL float64 split + per-system pre-computed terms."""
    key = (id(HKL), crystal_system)
    cached = _LB_geom_cache.get(key)
    if cached is not None:
        return cached
    HKL_arr = np.asarray(HKL)
    H = HKL_arr[:, 0].astype(np.float64, copy=False)
    K = HKL_arr[:, 1].astype(np.float64, copy=False)
    L = HKL_arr[:, 2].astype(np.float64, copy=False)
    if crystal_system == 'CUBIC':
        denom = H*H + K*K + L*L
        cached = ('CUBIC', H, K, L, denom)
    elif crystal_system in ('TETRAGONAL', 'HEXAGONAL'):
        if crystal_system == 'HEXAGONAL':
            ab_term = (4.0/3.0) * (H*H + H*K + K*K)
        else:
            ab_term = H*H + K*K
        c_term = L*L
        cached = (crystal_system, H, K, L, ab_term, c_term)
    else:
        cached = (crystal_system, H, K, L)
    _LB_geom_cache[key] = cached
    return cached


def _LB_clear_geom_cache():
    """Empty the per-phase HKL stencil cache. Call after JCPDS reload."""
    _LB_geom_cache.clear()


def reflection_List(unit_cell, HKL, WL,
                    twoThetaMin=5, twoThetaMax=30, crystal_system='CUBIC'):
    """
    Reflection list (in d and in 2theta) for a single phase.
    NumPy-vectorised over reflections. Returns two ndarrays of shape
    (n_visible, 4) with columns [value, H, K, L].

    The MONOCLINIC formula is the headline bug-fix from v0:
       v0 had  ... - 2 * H * L * cos(b) / (a*c)   <-- cos of LATTICE LENGTH b
       fixed:  ... - 2 * H * L * cos(beta) / (a*c)
    """
    a, b, c = unit_cell[0], unit_cell[1], unit_cell[2]
    alp = math.radians(unit_cell[3])
    bet = math.radians(unit_cell[4])
    gam = math.radians(unit_cell[5])

    HKL_arr = np.asarray(HKL)
    H = HKL_arr[:, 0].astype(np.float64, copy=False)
    K = HKL_arr[:, 1].astype(np.float64, copy=False)
    L = HKL_arr[:, 2].astype(np.float64, copy=False)

    if crystal_system == 'CUBIC':
        dk = a / np.sqrt(H*H + K*K + L*L)
    elif crystal_system == 'TETRAGONAL':
        dk = 1.0 / np.sqrt((H*H + K*K) / (a*a) + L*L / (c*c))
    elif crystal_system == 'HEXAGONAL':
        dk = 1.0 / np.sqrt((4.0/3.0) * (H*H + H*K + K*K) / (a*a)
                           + L*L / (c*c))
    elif crystal_system == 'RHOMBOHEDRAL':
        # *** BUG FIX vs v0 / pre-audit code ***
        # The old form had ' + 2*(H*K + K*L + H*L) * cos^2(alpha) - cos(alpha)'
        # i.e. the '- cos(alpha)' was OUTSIDE the (HK+KL+HL) bracket. The
        # correct expression factors (cos^2(alpha) - cos(alpha)) per cross-term.
        # Reference: Cullity & Stock, 'Elements of X-Ray Diffraction', App. 3.
        cos_a = math.cos(alp)
        sin_a = math.sin(alp)
        num = ((H*H + K*K + L*L) * sin_a*sin_a
               + 2.0 * (H*K + K*L + H*L) * (cos_a*cos_a - cos_a))
        den = a*a * (1.0 - 3.0*cos_a*cos_a + 2.0*cos_a*cos_a*cos_a)
        dk = 1.0 / np.sqrt(num / den)
    elif crystal_system == 'ORTHOROMBIC':
        dk = 1.0 / np.sqrt(H*H / (a*a) + K*K / (b*b) + L*L / (c*c))
    elif crystal_system == 'MONOCLINIC':
        # *** BUG FIX vs v0 ***
        # v0 had: math.cos(b)  - which is cos of the lattice constant!
        # The correct term is cos(beta), the monoclinic angle.
        sin_b = math.sin(bet)
        cos_b = math.cos(bet)
        dk = 1.0 / np.sqrt(
            (1.0 / (sin_b*sin_b)) *
            (H*H / (a*a) + K*K * sin_b*sin_b / (b*b) + L*L / (c*c)
             - 2.0 * H * L * cos_b / (a * c)))
    elif crystal_system == 'TRICLINIC':
        # *** BUG FIX vs v0 / pre-audit code ***
        # The old form was missing one lattice-constant factor on EVERY
        # term (e.g. h^2*a^2*sin^2(beta) instead of h^2*b^2*c^2*sin^2(alpha)).
        # The result was wrong by ~order-of-magnitude even in the
        # orthorhombic limit. The implementation below uses the standard
        # S_ij coefficients of the metric tensor formulation (Cullity).
        ca, cb, cg = math.cos(alp), math.cos(bet), math.cos(gam)
        sa, sb, sg = math.sin(alp), math.sin(bet), math.sin(gam)
        V_cell = EoS.unitCellVolume(unit_cell)
        S11 = b*b * c*c * sa*sa
        S22 = a*a * c*c * sb*sb
        S33 = a*a * b*b * sg*sg
        S12 = a * b * c*c * (ca*cb - cg)
        S23 = a*a * b * c * (cb*cg - ca)
        S13 = a * b*b * c * (cg*ca - cb)
        inv_d2 = (1.0 / (V_cell*V_cell)) * (
            S11*H*H + S22*K*K + S33*L*L
            + 2.0*S12*H*K + 2.0*S23*K*L + 2.0*S13*H*L)
        dk = 1.0 / np.sqrt(inv_d2)
    else:
        raise ValueError(f"Unknown crystal_system: {crystal_system}")

    d_min = twoTheta_to_dspacing(twoThetaMax, WL)
    d_max = twoTheta_to_dspacing(twoThetaMin, WL)
    mask = (dk > d_min) & (dk < d_max)

    dk_in   = dk[mask]
    twoTh_in = dspacing_to_twoTheta(dk_in, WL)
    H_in, K_in, L_in = H[mask], K[mask], L[mask]

    twoTheta_reflection_list = np.column_stack((twoTh_in, H_in, K_in, L_in))
    dk_reflection_list       = np.column_stack((dk_in,   H_in, K_in, L_in))
    return dk_reflection_list, twoTheta_reflection_list


###############################################################################
# Le Bail / Pawley parameter initialisation
###############################################################################
def setup_sigma_params(params, sigma_mode, phase,
                       HKL, sigGuess, sigmaBounds=2.0,
                       in_window_mask=None):
    """
    Add the right sigma parameter(s) to `params` for the chosen mode.
        'fixed'      - one global sig_global (cheapest)
        'per_phase'  - one sigma per phase
        'caglioti'   - one global base sigma; angle scaling 1/cos(theta)
                       applied at evaluation time
        'separate'   - one sigma per HKL entry (most flexible, slowest)
    Upper bound on sigma is sigGuess * sigmaBounds in every mode.

    Parameters
    ----------
    HKL : np.ndarray or list
        The full HKL array for the phase, shape (n_hkl, >=4). Used as
        the index space for per-peak sigmas in 'separate' mode so the
        keys stay HKL-aligned (matches the amp-keying scheme).
    in_window_mask : np.ndarray of bool or None
        Per-HKL-index bool array marking which reflections fall inside
        the actual fit window. Used in 'separate' mode to freeze sigmas
        for out-of-window peaks (vary=False) so they don't introduce
        zero-Jacobian columns. If None, every sigma starts varying.

    Notes
    -----
    Bug fix vs v0 (synced with GUI EXODUS):
    'separate' mode used to key sigmas by filtered-list ordinal
    (`for j in range(len(dk_reflection_list))`). _add_phase_contribution
    looks up sigmas by HKL index, so the keys diverged whenever any HKL
    entry sat outside the fit window. A runtime fallback substituted the
    highest-index sigma when a key was missing, hiding the bug at the
    cost of using the wrong sigma. This function now keys by HKL index,
    the same way amplitudes do.
    """
    sig_max = sigGuess * sigmaBounds
    HKL_arr = np.asarray(HKL)
    n_hkl = HKL_arr.shape[0]

    if sigma_mode == "fixed":
        if 'sig_global' not in params:
            params.add('sig_global', value=sigGuess, min=0.0, max=sig_max)
    elif sigma_mode == "per_phase":
        params.add(f'sig_{phase}', value=sigGuess, min=0.0, max=sig_max)
    elif sigma_mode == "caglioti":
        if 'sig_global' not in params:
            params.add('sig_global', value=sigGuess, min=0.0, max=sig_max)
    elif sigma_mode == "separate":
        for hkl_idx in range(n_hkl):
            params.add(f'sig_{phase}_{hkl_idx+1}',
                       value=sigGuess, min=0.0, max=sig_max)
            # Freeze sigmas for reflections outside the fit window so
            # they don't introduce zero-Jacobian columns. They stay in
            # the dict so _add_phase_contribution always finds them.
            if in_window_mask is not None and not in_window_mask[hkl_idx]:
                params[f'sig_{phase}_{hkl_idx+1}'].set(vary=False)
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")


def get_sigma(params, sigma_mode, phase, j, cen, deg_to_rad):
    """Look up (or compute) the sigma to use for reflection j of `phase`."""
    if sigma_mode == "fixed":
        return params['sig_global'].value
    elif sigma_mode == "per_phase":
        return params[f'sig_{phase}'].value
    elif sigma_mode == "caglioti":
        theta = cen / 2 * deg_to_rad
        return params['sig_global'].value / np.cos(theta)
    elif sigma_mode == "separate":
        key = f'sig_{phase}_{j+1}'
        if key not in params:
            existing = [k for k in params if k.startswith(f'sig_{phase}_')]
            if existing:
                key = sorted(existing,
                             key=lambda k: int(k.rsplit('_', 1)[-1]))[-1]
            else:
                raise KeyError(f"No sigma parameters found for phase {phase}")
        return params[key].value
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")


def _add_lattice_param_block(params, phase, unit_cell_vec, maxShift,
                             crystal_system):
    """
    Add the six lattice-parameter lmfit Parameters with the right
    constraints for the crystal system. Encapsulating this here keeps
    initialise_parameters_LB readable.
    """
    a0, b0, c0, alp0, bet0, gam0 = (unit_cell_vec[0], unit_cell_vec[1],
                                    unit_cell_vec[2], unit_cell_vec[3],
                                    unit_cell_vec[4], unit_cell_vec[5])
    p = phase  # alias

    def _add_free(name, value):
        params.add(name, value=value,
                   min=value * (1 - maxShift),
                   max=value * (1 + maxShift))

    if crystal_system == 'CUBIC':
        _add_free(f'a_{p}', a0)
        params.add(f'b_{p}', expr=f'a_{p}')
        params.add(f'c_{p}', expr=f'a_{p}')
        for nm, v in zip(['alp', 'bet', 'gam'], [alp0, bet0, gam0]):
            _add_free(f'{nm}_{p}', v); params[f'{nm}_{p}'].set(vary=False)
    elif crystal_system == 'TETRAGONAL':
        _add_free(f'a_{p}', a0)
        params.add(f'b_{p}', expr=f'a_{p}')
        _add_free(f'c_{p}', c0)
        for nm, v in zip(['alp', 'bet', 'gam'], [alp0, bet0, gam0]):
            _add_free(f'{nm}_{p}', v); params[f'{nm}_{p}'].set(vary=False)
    elif crystal_system == 'HEXAGONAL':
        _add_free(f'a_{p}', a0)
        params.add(f'b_{p}', expr=f'a_{p}')
        _add_free(f'c_{p}', c0)
        for nm, v in zip(['alp', 'bet', 'gam'], [alp0, bet0, gam0]):
            _add_free(f'{nm}_{p}', v); params[f'{nm}_{p}'].set(vary=False)
    elif crystal_system == 'RHOMBOHEDRAL':
        _add_free(f'a_{p}', a0)
        params.add(f'b_{p}', expr=f'a_{p}')
        params.add(f'c_{p}', expr=f'a_{p}')
        _add_free(f'alp_{p}', alp0)
        params.add(f'bet_{p}', expr=f'alp_{p}')
        params.add(f'gam_{p}', expr=f'alp_{p}')
    elif crystal_system == 'ORTHOROMBIC':
        _add_free(f'a_{p}', a0); _add_free(f'b_{p}', b0); _add_free(f'c_{p}', c0)
        for nm, v in zip(['alp', 'bet', 'gam'], [alp0, bet0, gam0]):
            _add_free(f'{nm}_{p}', v); params[f'{nm}_{p}'].set(vary=False)
    elif crystal_system == 'MONOCLINIC':
        _add_free(f'a_{p}', a0); _add_free(f'b_{p}', b0); _add_free(f'c_{p}', c0)
        _add_free(f'alp_{p}', alp0); params[f'alp_{p}'].set(vary=False)
        _add_free(f'bet_{p}', bet0)  # angle is allowed to vary
        _add_free(f'gam_{p}', gam0); params[f'gam_{p}'].set(vary=False)
    else:  # TRICLINIC or unknown - everything free
        _add_free(f'a_{p}', a0); _add_free(f'b_{p}', b0); _add_free(f'c_{p}', c0)
        _add_free(f'alp_{p}', alp0); _add_free(f'bet_{p}', bet0)
        _add_free(f'gam_{p}', gam0)


def initialise_parameters_LB(phases, WL, twoThetaMin=5, twoThetaMax=40,
                             sigma_mode='caglioti',
                             ampGuess=1, sigGuess=0.05,
                             sigmaBounds=2.0, maxShift=0.5):
    """
    Build the lmfit Parameters for a Le Bail / Pawley-style fit.

    Important details
    -----------------
    * Lattice parameters are bounded to value*(1+/-maxShift). maxShift==0
      freezes them entirely.
    * Amplitude parameters are HKL-INDEX keyed (one per HKL entry, not
      one per buffered-filtered-list entry). Out-of-window reflections
      AND reflections with zero/negative JCPDS intensity are frozen
      (vary=False) so they don't introduce zero-Jacobian columns. lmfit
      can then return a real covariance matrix and per-parameter stderrs.
    * Sigma model is selected by `sigma_mode` (see setup_sigma_params).

    Bug fix vs v0 (synced with GUI EXODUS, May 2026)
    -----------------------------------------------
    Previously the amp loop used `for j in range(len(dk_reflection_list))`
    where dk_reflection_list was the buffered (twoThetaMax+5) filtered
    output of reflection_List. That keyed amps by position in the
    FILTERED list. LB_fit_Model / _add_phase_contribution, however,
    look up amps by HKL index (visible_indices = np.flatnonzero(mask)
    against the FULL HKL array). The two indices coincide only when
    every HKL entry sits inside the buffered window - the moment one
    drops out, every subsequent amp lookup is off by one, the fit gets
    the wrong amplitudes, and the residual silently regresses to a
    biased BG. The fix: create one amp parameter per HKL row,
    HKL-index keyed; freeze the ones that won't actually be fit.
    """
    print('Initialising Parameters')
    params = Parameters()

    params.add('BG', value=0.0, min=-.5, max=.5)
    params.add('WL', value=WL, min=WL*0.95, max=WL*1.15)
    params['WL'].set(vary=False)

    for phase in phases['phases_used']:
        cs = phases[f'phase_{phase}_crystal_system']
        uc = phases[f'phase_{phase}_unit_cell']
        HKL_full = np.asarray(phases[f'phase_{phase}_HKL'], dtype=float)
        n_hkl = HKL_full.shape[0]

        # 6 lattice parameters with crystal-system constraints
        _add_lattice_param_block(params, phase, uc, maxShift, cs)

        # Compute 2theta for ALL HKL entries (no 2theta filter) so we
        # can build an HKL-index-aligned in_window mask. Forbidden /
        # non-physical reflections that don't survive reflection_List's
        # internal filter (e.g. NaN d-spacings) get +inf and so are
        # treated as out-of-window without raising on comparison.
        _, tt_all = reflection_List(
            uc, HKL_full, WL,
            twoThetaMin=0.0, twoThetaMax=180.0,
            crystal_system=cs)

        tt_by_hkl_idx = np.full(n_hkl, np.inf)
        if tt_all.size:
            hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): i
                        for i, (h, k, l) in enumerate(HKL_full[:, :3])}
            for row in tt_all:
                key = (int(round(row[1])), int(round(row[2])),
                       int(round(row[3])))
                idx = hkl_keys.get(key)
                if idx is not None:
                    tt_by_hkl_idx[idx] = row[0]

        in_window = (tt_by_hkl_idx >= twoThetaMin) & (tt_by_hkl_idx <= twoThetaMax)

        # One amp parameter per HKL row, keyed by HKL index. Generous
        # upper bound (3x the initial guess) so the optimiser has
        # headroom without rejecting non-zero JCPDS intensities.
        for hkl_idx in range(n_hkl):
            amp_init = ampGuess / 100.0 * HKL_full[hkl_idx, 3]
            upper = amp_init * 3.0 if amp_init > 0 else ampGuess * 3.0
            # Guard against lmfit's min==max==0 rejection (happens when
            # ampGuess <= 0, e.g. an empty/zero pattern).
            if upper <= 0:
                upper = 1.0
            params.add(f'amp_{phase}_{hkl_idx+1}',
                       value=amp_init, min=0.0, max=upper)
            # Freeze amp if (a) reflection is outside the fit window or
            # (b) JCPDS intensity is zero/negative. A reflection with no
            # structure factor has no amplitude to fit and would otherwise
            # contribute a zero column to the Jacobian, making J^T J
            # singular and the covariance matrix unavailable.
            if (not in_window[hkl_idx]) or (HKL_full[hkl_idx, 3] <= 0.0):
                params[f'amp_{phase}_{hkl_idx+1}'].set(vary=False)

        # Sigma parameters - HKL-index keyed in 'separate' mode
        setup_sigma_params(params, sigma_mode, phase,
                           phases[f'phase_{phase}_HKL'], sigGuess,
                           sigmaBounds=sigmaBounds,
                           in_window_mask=in_window)

        if maxShift == 0:
            for lp in ['a_', 'b_', 'c_', 'alp_', 'bet_', 'gam_']:
                key = lp + str(phase)
                if key in params and params[key].expr is None:
                    params[key].set(vary=False)

    return params


###############################################################################
# LB residual / model
###############################################################################
def _add_phase_contribution(pv, phase, cens, visible_indices, sigma_mode,
                            x, y_fit):
    """
    In-place: add a phase's Gaussian peak superposition to y_fit using
    NumPy broadcasting (no per-reflection Python loop).
    """
    n_refl = len(cens)
    if n_refl == 0:
        return

    amps = np.fromiter(
        (pv[f'amp_{phase}_{int(idx)+1}'] for idx in visible_indices),
        dtype=np.float64, count=n_refl)

    if sigma_mode == 'caglioti':
        sigs = pv['sig_global'] / np.cos(cens / 2 * deg_to_rad)
    elif sigma_mode == 'fixed':
        sigs = np.full(n_refl, pv['sig_global'])
    elif sigma_mode == 'per_phase':
        sigs = np.full(n_refl, pv[f'sig_{phase}'])
    elif sigma_mode == 'separate':
        available_keys = [k for k in pv if k.startswith(f'sig_{phase}_')]
        if not available_keys:
            raise KeyError(f"No sigma parameters found for phase {phase}")
        last_key = max(available_keys,
                       key=lambda k: int(k.rsplit('_', 1)[-1]))
        sigs = np.fromiter(
            (pv.get(f'sig_{phase}_{int(idx)+1}', pv[last_key])
             for idx in visible_indices),
            dtype=np.float64, count=n_refl)
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    diff = (x[None, :] - cens[:, None]) / sigs[:, None]
    y_fit += (amps[:, None] * np.exp(-0.5 * diff * diff)).sum(axis=0)


def LB_fit_Model(params, x, y, phases, twoThetaMin=0, twoThetaMax=30,
                 sigma_mode='caglioti'):
    """
    Vectorised residual.
    Single params.valuesdict() lookup, cached HKL stencils, NumPy
    broadcasting over peaks - same physics as v0 (with the d-spacing
    fixes) but ~one order of magnitude faster on a typical batch.

    Bug fix vs v0 (synced with GUI EXODUS, May 2026)
    -----------------------------------------------
    The slow-path branch (rhombohedral / orthorhombic / monoclinic /
    triclinic) used to pass `np.arange(n_refl)` as visible_indices to
    _add_phase_contribution. That coincides with the HKL index only
    when every HKL entry falls inside the fit window. As soon as one
    or more are filtered out by reflection_List, the indices diverge
    and the residual reads the wrong amps / sigmas, giving a biased
    fit for any non-cubic / non-tetragonal / non-hexagonal phase.
    The fix below rebuilds the visible_indices by looking each surviving
    (H, K, L) row back up in the full HKL array - matching the
    HKL-index keying used in initialise_parameters_LB.
    """
    pv = params.valuesdict()
    y_fit = np.full_like(x, pv['BG'])
    WL    = pv['WL']

    d_min = twoTheta_to_dspacing(twoThetaMax, WL)
    d_max = twoTheta_to_dspacing(twoThetaMin, WL)

    for phase in phases['phases_used']:
        HKL = phases[f'phase_{phase}_HKL']
        cs  = phases[f'phase_{phase}_crystal_system']
        a   = pv[f'a_{phase}']

        stencil = _LB_phase_stencil(HKL, cs)
        if stencil[0] == 'CUBIC':
            _, H, K, L, denom = stencil
            dk = a / np.sqrt(denom)
        elif stencil[0] in ('TETRAGONAL', 'HEXAGONAL'):
            _, H, K, L, ab_term, c_term = stencil
            c = pv[f'c_{phase}']
            dk = 1.0 / np.sqrt(ab_term / (a*a) + c_term / (c*c))
        else:
            # Slow path: rhombohedral / orthorhombic / monoclinic / triclinic.
            # reflection_List filters the HKL array down to in-window
            # reflections AND drops their original row index, so we
            # have to reconstruct visible_indices by hashing (H, K, L).
            unit_cell = (a, pv[f'b_{phase}'], pv[f'c_{phase}'],
                         pv[f'alp_{phase}'], pv[f'bet_{phase}'],
                         pv[f'gam_{phase}'])
            _, twoTh_full = reflection_List(
                unit_cell, HKL, WL,
                crystal_system=cs,
                twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
            n_refl = len(twoTh_full)
            if n_refl == 0:
                continue
            cens = twoTh_full[:, 0]

            HKL_full = np.asarray(HKL)
            hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): i
                        for i, (h, k, l) in enumerate(HKL_full[:, :3])}
            visible_indices = np.empty(n_refl, dtype=np.int64)
            for r, row in enumerate(twoTh_full):
                key = (int(round(row[1])), int(round(row[2])),
                       int(round(row[3])))
                visible_indices[r] = hkl_keys.get(key, -1)

            ok = visible_indices >= 0
            if not ok.all():
                cens = cens[ok]
                visible_indices = visible_indices[ok]
                if cens.size == 0:
                    continue

            _add_phase_contribution(pv, phase, cens, visible_indices,
                                    sigma_mode, x, y_fit)
            continue

        mask = (dk > d_min) & (dk < d_max)
        if not mask.any():
            continue
        dk_in = dk[mask]
        cens  = dspacing_to_twoTheta(dk_in, WL)
        visible_indices = np.flatnonzero(mask)

        _add_phase_contribution(pv, phase, cens, visible_indices,
                                sigma_mode, x, y_fit)

    return y_fit - y


###############################################################################
# Result helpers (unit-cell extraction, tickmarks, components)
###############################################################################
def unitCell_fit(result, phase):
    """6-element ndarray of [a, b, c, alpha, beta, gamma] from a fit."""
    return np.array([result.params[f'a_{phase}'].value,
                     result.params[f'b_{phase}'].value,
                     result.params[f'c_{phase}'].value,
                     result.params[f'alp_{phase}'].value,
                     result.params[f'bet_{phase}'].value,
                     result.params[f'gam_{phase}'].value])


def unitCell_errors(result, phase, fallback=None):
    """
    Per-parameter 1-sigma errors as a 6-element ndarray.
    Constraint expressions are followed (a slaved 'b = a' inherits its
    master's stderr). Fixed parameters return 0. A varying parameter that
    has no stderr (covar missing or sat on a bound) returns `fallback`
    if provided, else NaN.
    """
    names = [f'a_{phase}', f'b_{phase}', f'c_{phase}',
             f'alp_{phase}', f'bet_{phase}', f'gam_{phase}']
    errs = np.zeros(6)
    for i, name in enumerate(names):
        par = result.params[name]
        if par.expr is not None:
            stderr = par.stderr
            if stderr is None:
                master = result.params.get(par.expr.strip())
                stderr = master.stderr if master is not None else None
        elif not par.vary:
            stderr = 0.0
        else:
            stderr = par.stderr
        errs[i] = stderr if stderr is not None else (
            fallback if fallback is not None else np.nan)
    return errs


def create_ticks_guess(phases, twoThetaMin, twoThetaMax, WL):
    """List of per-phase 2theta tickmark lists computed from the GUESS UC."""
    tickArray = []
    for phase in phases['phases_used']:
        _, twoTheta_reflection_list = reflection_List(
            phases[f'phase_{phase}_unit_cell'],
            phases[f'phase_{phase}_HKL'],
            WL,
            twoThetaMin=twoThetaMin,
            twoThetaMax=twoThetaMax,
            crystal_system=phases[f'phase_{phase}_crystal_system'])
        tickArray.append(twoTheta_reflection_list[:, 0].tolist())
    return tickArray


def create_ticks(result, phases, twoThetaMin, twoThetaMax, WL):
    """Same as create_ticks_guess but uses the FITTED UC instead."""
    tickArray = []
    for phase in phases['phases_used']:
        _, twoTheta_reflection_list = reflection_List(
            unitCell_fit(result, phase),
            phases[f'phase_{phase}_HKL'],
            WL,
            twoThetaMin=twoThetaMin,
            twoThetaMax=twoThetaMax,
            crystal_system=phases[f'phase_{phase}_crystal_system'])
        tickArray.append(twoTheta_reflection_list[:, 0].tolist())
    return tickArray


def create_guessComponents(twoTheta, phases, params,
                           twoThetaMin, twoThetaMax,
                           sigma_mode='caglioti',
                           printInputComponents=False):
    """Build the initial-guess fit envelope (BG + sum of Gaussians)."""
    components = {'BG': params['BG'].value}
    for phase in phases['phases_used']:
        HKL = phases[f'phase_{phase}_HKL']
        dk_reflection_list, twoTheta_reflection_list = reflection_List(
            [params[f'a_{phase}'], params[f'b_{phase}'], params[f'c_{phase}'],
             params[f'alp_{phase}'], params[f'bet_{phase}'],
             params[f'gam_{phase}']],
            HKL, WL=params['WL'],
            crystal_system=phases[f'phase_{phase}_crystal_system'],
            twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
        for j in range(len(dk_reflection_list)):
            cen = twoTheta_reflection_list[j][0]
            amp = params[f'amp_{phase}_{j+1}'].value
            sig = get_sigma(params, sigma_mode, phase, j, cen, deg_to_rad)
            if printInputComponents:
                print(f'xc {cen}\tamp {amp}\tsig {sig}')
            components[f'comp_{phase}_{j+1}'] = (
                amp * np.exp(-0.5 * ((twoTheta - cen) / sig) ** 2))
    return sum(components.values()), components


def create_fitComponents(twoTheta, phases, params, result, save_path,
                         frame, valueIntBGsub,
                         twoThetaMin, twoThetaMax,
                         sigma_mode='caglioti',
                         printOutputComponents=False,
                         saveTextOutput=True):
    """
    Build the BEST-FIT envelope and (optionally) write the per-frame fit
    text file under save_path/fits/. The save behaviour is preserved
    from v0 - useful for publication tables.
    """
    components = {}
    peak_fit   = {}
    for phase in phases['phases_used']:
        peak_fit_list = []
        HKL = phases[f'phase_{phase}_HKL']
        dk_reflection_list, twoTheta_reflection_list = reflection_List(
            [result.params[f'a_{phase}'], result.params[f'b_{phase}'],
             result.params[f'c_{phase}'], result.params[f'alp_{phase}'],
             result.params[f'bet_{phase}'], result.params[f'gam_{phase}']],
            HKL, WL=result.params['WL'],
            crystal_system=phases[f'phase_{phase}_crystal_system'],
            twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)

        for j in range(len(dk_reflection_list)):
            cen = twoTheta_reflection_list[j][0]
            amp = result.params[f'amp_{phase}_{j+1}'].value
            sig = get_sigma(result.params, sigma_mode, phase, j, cen,
                            deg_to_rad)
            if printOutputComponents:
                print(f'xc {cen}\tamp {amp}\tsig {sig}')
            peak_fit_list.append([cen, amp, sig])
            components[f'comp_{phase}_{j+1}'] = (
                amp * np.exp(-0.5 * ((twoTheta - cen) / sig) ** 2))
        peak_fit[f'phase_{phase}'] = np.array(peak_fit_list)

    components['BG'] = result.params['BG'].value
    bestFit = sum(components.values())

    if save_path and saveTextOutput:
        subfolder_path = save_path + 'fits'
        os.makedirs(subfolder_path, exist_ok=True)
        with open(os.path.join(subfolder_path, f'frame_{frame}.txt'),
                  'w') as f:
            f.write(f'Frame {frame}\n2Theta\tInt\tFit\n')
            np.savetxt(f,
                       np.column_stack((twoTheta, valueIntBGsub, bestFit)),
                       fmt='%.6f', delimiter='\t')

    return bestFit, components


###############################################################################
# Fit-quality reporting
###############################################################################
def calculate_Rw(y_obs, y_calc, weights=None):
    """Weighted profile R-factor (small = good)."""
    if weights is None:
        weights = np.ones_like(y_obs)
    numerator   = np.sum(weights * (y_obs - y_calc) ** 2)
    denominator = np.sum(weights * y_obs ** 2)
    return np.sqrt(numerator / denominator)


def fitStatistics(result, twoTheta, fullFitReport=False, verbose=False):
    """
    Print a summary of the fit and return sqrt(reduced chi^2) for use as
    a global error estimate. Per-parameter stderrs (when available) sit
    on result.params[name].stderr - use unitCell_errors() to pull them.
    """
    print(f"\nFitting success: {result.success}")
    if not result.success:
        print("Fit was not successful. No error estimates available.")
        return None

    dof = max(1, len(twoTheta) - len(result.var_names))
    chi2_red = result.chisqr / dof
    estimated_error_global = np.sqrt(chi2_red)
    print(f"Reduced Chi-squared: {chi2_red:.6e}")

    has_covar = result.covar is not None
    if has_covar:
        n_with = sum(1 for p in result.params.values()
                     if p.vary and p.stderr is not None)
        n_total = sum(1 for p in result.params.values() if p.vary)
        print(f"Covariance matrix: available "
              f"(stderrs on {n_with}/{n_total} varying params)")
    else:
        print("Covariance matrix: NOT available "
              "(fit may be ill-conditioned or sitting on bounds)")

    if fullFitReport:
        print("\nFit Report")
        print("=" * 50)
        print(f"Chi-square        : {result.chisqr:.4f}")
        print(f"Reduced Chi-square: {result.redchi:.4f}")
        print(f"Akaike info crit  : {result.aic:.4f}")
        print(f"Bayesian info crit: {result.bic:.4f}")
        print(f"nfev              : {result.nfev}")
        print("=" * 50)
        print("\nFitted Parameters:")
        for name, par in result.params.items():
            stderr_str = (f"+/-{par.stderr:.4g}" if par.stderr is not None
                          else "stderr=N/A")
            on_bound = ""
            if par.vary and par.min > -np.inf and par.max < np.inf:
                tol = 1e-6 * max(abs(par.min), abs(par.max), 1.0)
                if abs(par.value - par.min) < tol:
                    on_bound = "  [AT MIN]"
                elif abs(par.value - par.max) < tol:
                    on_bound = "  [AT MAX]"
            print(f"  {name:<14} = {par.value:>12.6g}  {stderr_str:<14} "
                  f"vary={str(par.vary):<5}{on_bound}")

    return estimated_error_global


###############################################################################
# Single-frame Le Bail driver (script entry point)
###############################################################################
def fit_LB(data_path, phases, poni_file, save_path,
           frame=10, twoThetaMin=0, twoThetaMax=10,
           framePressureGuess=None,
           prominence=0.10, height=0.08, order=20,
           ampGuess=1, sigGuess=0.05,
           sigma_mode='caglioti', sigmaBounds=2.0,
           ampPrefactor=1.0, maxShift=0.5,
           bg_method='POLY', bg_lam=1e5, bg_p=0.01, bg_niter=10,
           peakwidth=0.35,
           ftol=1e-5, xtol=1e-5, gtol=1e-5,
           excludePeakList=None, plotLB=True,
           saveTextOutput=True):
    """
    BG-subtract one frame and refine it with the Le Bail / Pawley model.

    New optional knobs (all default to v0-equivalent behaviour)
    -----------------------------------------------------------
    sigma_mode     : 'caglioti' (default), 'fixed', 'per_phase', 'separate'
    sigmaBounds    : sigma_max = sigGuess * sigmaBounds
    ampPrefactor   : multiplies max(I) when scaling ampGuess
    maxShift       : lattice-parameter freedom (0=frozen, 0.5=v0 default)
    bg_method      : 'POLY' (default) or 'ALS'
    bg_lam/p/niter : ALS parameters (only used if bg_method=='ALS')
    peakwidth      : exclusion half-width for the polynomial BG (deg)
    ftol/xtol/gtol : least-squares tolerances passed straight to scipy
    """
    if framePressureGuess is None:
        framePressureGuess = [[0, 0], [3000, 0]]
    if excludePeakList is None:
        excludePeakList = []

    # 1. Load the raw frame
    twoTheta, valueInt, errorInt = load_frame(
        frame, data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)

    # 2. Background subtraction
    if bg_method.upper() == 'ALS':
        # ALS BG works on the file directly via its own loader path
        files = sorted(glob.glob(os.path.join(data_path, "*")),
                       key=extract_number)
        files = [f for f in files if extract_number(f) != float('inf')]
        try:
            twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = \
                backgroundFit_ALS(files[frame - 1],
                                  twoThetaMin=twoThetaMin,
                                  twoThetaMax=twoThetaMax,
                                  lam=bg_lam, p=bg_p, niter=bg_niter)
        except Exception as e:
            print(f'ALS BG failed: {e} - falling back to flat BG.')
            valueBG = np.zeros_like(twoTheta)
            twoTheta, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)
    else:
        try:
            twoTheta, valueBG = backgroundFit(
                data_path, frame=frame,
                twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                peakSearchAuto=True, plotBG=False,
                prominence=prominence, height=height,
                order=order, peakwidth=peakwidth,
                excludePeakList=excludePeakList)
        except Exception as e:
            print(f'No Background subtraction performed ({e}).')
            valueBG = np.zeros_like(twoTheta)
        twoTheta, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)

    # 3. Plot raw + initial guess (optional)
    if plotLB:
        plt.figure()
        plt.scatter(twoTheta, valueIntBGsub,
                    marker='.', color='gray', label='Raw Data', s=3)

    # 4. Initialise parameters and evaluate the initial guess
    WL = load_poni(poni_file)
    params = initialise_parameters_LB(
        phases, WL=WL,
        ampGuess=max(valueIntBGsub) * ampPrefactor,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        sigma_mode=sigma_mode, sigGuess=sigGuess,
        sigmaBounds=sigmaBounds, maxShift=maxShift)

    guessFit, _ = create_guessComponents(
        twoTheta, phases, params,
        twoThetaMin, twoThetaMax,
        sigma_mode=sigma_mode, printInputComponents=False)

    if plotLB:
        plt.plot(twoTheta, guessFit, c='blue', ls='dotted', lw=1,
                 label='Initial guess')

    # 5. Run the fit
    try:
        result = minimize(
            lambda params: LB_fit_Model(params, twoTheta, valueIntBGsub,
                                        phases, twoThetaMin, twoThetaMax,
                                        sigma_mode=sigma_mode),
            params, method='least_squares',
            calc_covar=True,
            ftol=ftol, xtol=xtol, gtol=gtol,
        )
    except Exception as e:
        print(f'Error in frame {frame}: {e}')
        return {}

    # 6. Build the best-fit envelope (and write the per-frame text file)
    bestFit, _ = create_fitComponents(
        twoTheta, phases, params, result, save_path, frame, valueIntBGsub,
        twoThetaMin, twoThetaMax,
        sigma_mode=sigma_mode, printOutputComponents=False,
        saveTextOutput=saveTextOutput)

    # 7. Statistics
    estimatedError = fitStatistics(result, twoTheta)
    Rw = calculate_Rw(y_obs=valueIntBGsub, y_calc=bestFit)
    print(f'Rw for this fit is {Rw:.4g}')

    # 8. Tickmarks (per-phase)
    tickArray = create_ticks(result, phases, twoThetaMin, twoThetaMax, WL)

    # 9. Per-frame plot (optional)
    if plotLB and save_path is not None:
        tickStep = max(valueIntBGsub) / 10
        for j in range(len(tickArray)):
            for tick in tickArray[j]:
                plt.scatter(tick, -tickStep - (j + 2) * tickStep,
                            marker='|', linewidths=2, s=100,
                            color=colours[j])
        plt.plot(twoTheta, bestFit, c='red', ls='-', lw=1, label='Best fit')
        plt.plot(twoTheta, valueIntBGsub - bestFit - tickStep,
                 c='blue', ls='-', lw=1, label='Residuals')
        plt.xlim(left=twoThetaMin); plt.xlim(right=twoThetaMax)
        plt.legend()
        plt.ylabel('Intensity (arb.u.)'); plt.xlabel('TwoTheta (deg)')
        os.makedirs(save_path + 'fits', exist_ok=True)
        plt.savefig(save_path + 'fits/' + f'frame_{frame}.pdf', format='pdf')
        plt.show()
        plt.clf()

    # 10. Pack results
    resultsFit = {'Rw': float(Rw)}
    for i, phase in enumerate(phases['phases_used']):
        UC_fit  = unitCell_fit(result, phase)
        UC_err  = unitCell_errors(result, phase, fallback=estimatedError)

        resultsFit[f'phase_{phase}_unit_cell_fit']   = UC_fit
        resultsFit[f'phase_{phase}_unit_cell_error'] = UC_err
        # Kept for backwards compatibility with v0 scripts that read by 'i'
        resultsFit[f'phase_{i}_unit_cell_error']     = UC_err

        V = float(EoS.unitCellVolume(UC_fit))
        resultsFit[f'phase_{phase}_V_fit'] = V

        comp = phases[f'phase_{phase}_compression_constants']
        resultsFit[f'phase_{phase}_P_fit'] = float(EoS.BM3_EOS(
            V, V0=float(comp[0]), K0=float(comp[1]), K0P=float(comp[2])))

        resultsFit[f'phase_{phase}_BG_fit']            = result.params['BG'].value
        resultsFit[f'phase_{phase}_peakPosition_fit']  = tickArray[i]
        resultsFit['data_BGsub']                       = np.array([twoTheta, valueIntBGsub])
        resultsFit['data_fit']                         = np.array([twoTheta, bestFit])
        print(f'Unit Cell {phase} fit: \n{UC_fit}')

    print('\n=============================================\n')
    return resultsFit


###############################################################################
# Indexing-only driver (script entry point)
###############################################################################
def index_LB(data_path, phases, poni_file, save_path,
             frame=10, twoThetaMin=0, twoThetaMax=30, plotIndex=True,
             framePressureGuess=None,
             prominence=0.10, height=0.08, order=20, excludePeakList=None,
             ampGuess=1, sigGuess=0.05,
             sigma_mode='caglioti', sigmaBounds=2.0, ampPrefactor=1.0,
             maxShift=0.5, peakwidth=0.35,
             bg_method='POLY', bg_lam=1e5, bg_p=0.01, bg_niter=10):
    """
    Subtract BG and overlay the *initial-guess* peak positions for one
    frame. No fit is performed - this is the cheap "does my JCPDS even
    point at the right peaks?" sanity check.
    """
    if framePressureGuess is None:
        framePressureGuess = [[0, 0], [3000, 0]]
    if excludePeakList is None:
        excludePeakList = []

    twoTheta, valueInt, errorInt = load_frame(
        frame, data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)

    if bg_method.upper() == 'ALS':
        files = sorted(glob.glob(os.path.join(data_path, "*")),
                       key=extract_number)
        files = [f for f in files if extract_number(f) != float('inf')]
        try:
            twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = \
                backgroundFit_ALS(files[frame - 1],
                                  twoThetaMin=twoThetaMin,
                                  twoThetaMax=twoThetaMax,
                                  lam=bg_lam, p=bg_p, niter=bg_niter)
        except Exception as e:
            print(f'ALS BG failed: {e} - falling back to flat BG.')
            valueBG = np.zeros_like(twoTheta)
            twoTheta, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)
    else:
        try:
            twoTheta, valueBG = backgroundFit(
                data_path, frame=frame,
                twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                peakSearchAuto=True, plotBG=False,
                prominence=prominence, height=height,
                order=order, peakwidth=peakwidth,
                excludePeakList=excludePeakList)
        except Exception as e:
            print(f'No Background subtraction performed ({e}).')
            valueBG = np.zeros_like(twoTheta)
        twoTheta, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)

    if plotIndex:
        plt.figure()
        plt.scatter(twoTheta, valueIntBGsub,
                    marker='.', color='gray', label='Raw Data', s=3)

    WL = load_poni(poni_file)
    params = initialise_parameters_LB(
        phases, WL=WL,
        ampGuess=max(valueIntBGsub) * ampPrefactor,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        sigma_mode=sigma_mode, sigGuess=sigGuess,
        sigmaBounds=sigmaBounds, maxShift=maxShift)

    guessFit, _ = create_guessComponents(
        twoTheta, phases, params,
        twoThetaMin, twoThetaMax, sigma_mode=sigma_mode)

    if plotIndex:
        plt.plot(twoTheta, guessFit, c='blue', ls='dotted', lw=1,
                 label='Initial guess')
        tickArray = create_ticks_guess(phases, twoThetaMin, twoThetaMax, WL)
        tickStep = max(valueIntBGsub) / 10
        for j in range(len(tickArray)):
            for tick in tickArray[j]:
                plt.scatter(tick, -tickStep - (j + 2) * tickStep,
                            marker='|', linewidths=2, s=100,
                            color=colours[j])
        plt.xlim(left=twoThetaMin); plt.xlim(right=twoThetaMax)
        plt.legend()
        plt.ylabel('Intensity (arb.u.)'); plt.xlabel('TwoTheta (deg)')
        plt.show()


###############################################################################
# Batch Le Bail driver - publication-grade waterfall + fit panel
###############################################################################
def plot_LB_fit(data_path, phases, poni_file, save_path,
                framePressureGuess, phase_rules=None,
                collectionTime=1, ylabel='Frame',
                frameStart=0, frameEnd=1, frameStep=1,
                twoThetaMin=8, twoThetaMax=18,
                plotLB=False, logPlot=True, CorrNegativeValues=0.02,
                prominence=0.10, height=0.08, order=1, excludePeakList=None,
                ampGuess=1, sigGuess=0.05,
                sigma_mode='caglioti', sigmaBounds=2.0, ampPrefactor=1.0,
                maxShift=0.5, peakwidth=0.35,
                bg_method='POLY', bg_lam=1e5, bg_p=0.01, bg_niter=10,
                ftol=1e-5, xtol=1e-5, gtol=1e-5,
                method='pressureWalk', deltaP=0.5,
                offsetStep=0.05,
                saveTextOutput=True,
                figsize=(8, 8), pcolor_cmap='binary_r'):
    """
    Batch fit a sequence of frames and produce the publication figure
    (2D waterfall on top, refined patterns + fits on the bottom).

    Parameters
    ----------
    method : 'sequential', 'pressureWalk', or 'pressureHelper'
    """
    if phase_rules is None:
        phase_rules = [(lambda f: f >= 0, [0])]
    if excludePeakList is None:
        excludePeakList = []
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = fig.add_gridspec(3, 4)

    # Top panel - 2D waterfall of the raw data
    ax1 = fig.add_subplot(spec[0:1, :2])
    XRD_patterns = {'frame': [], 'twoTheta': [], 'valueInt': []}
    for frame in np.arange(frameStart, frameEnd + 1, frameStep):
        twoTheta, valueIntBGsub, _ = load_frame_fast(
            frame, data_path, twoThetaMin, twoThetaMax)
        XRD_patterns['frame'].append([frame * collectionTime])
        if logPlot:
            valueIntBGsub = np.log(np.asarray(valueIntBGsub) + CorrNegativeValues)
        XRD_patterns['twoTheta'].append(twoTheta)
        XRD_patterns['valueInt'].append(valueIntBGsub)

    im = ax1.pcolormesh(XRD_patterns['twoTheta'],
                        XRD_patterns['frame'],
                        XRD_patterns['valueInt'],
                        cmap=pcolor_cmap)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Intensity (counts)')
    ax1.set(ylabel=ylabel)
    ax1.set_aspect('auto')
    ax1.set_xlim((twoThetaMin, twoThetaMax))
    ax1.set_ylim((frameStart * collectionTime, frameEnd * collectionTime))

    # Bottom panel - refined patterns + best-fit overlays
    ax2 = fig.add_subplot(spec[1:3, :2])
    ax2.set(ylabel='Intensity (arb.u.)')
    ax2.set(xlabel='Two Theta (deg)')
    ax2.set_xlim((twoThetaMin, twoThetaMax))

    offset = 0
    frameList = list(range(frameStart, frameEnd + 1, frameStep))

    # Per-phase result accumulators
    result_lists = {}
    for i, phase in enumerate(phases['phases_Number_array']):
        result_lists[f'P_list_{i}']     = []
        result_lists[f't_list_{i}']     = []
        result_lists[f'phase_list_{i}'] = []
        result_lists[f'error_list_{i}'] = []
        result_lists[f'Rw_list_{i}']    = []

    # Main loop
    for frame in frameList:
        specify_phases(phases, phase_rules, frame)
        print(f'For frame {frame} the phases {phases["phases_used"]} are used')

        if method == 'pressureHelper':
            for phase in phases['phases_used']:
                update_unitCell(phases[f'phase_{phase}_unit_cell_0'],
                                phases, phase)
                update_phases(frame, phases, phase, framePressureGuess,
                              method='pressureHelper')

        resultsFit = fit_LB(
            data_path, phases, poni_file, save_path,
            frame=frame,
            twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
            framePressureGuess=framePressureGuess,
            prominence=prominence, height=height, order=order,
            ampGuess=ampGuess, sigGuess=sigGuess,
            sigma_mode=sigma_mode, sigmaBounds=sigmaBounds,
            ampPrefactor=ampPrefactor, maxShift=maxShift,
            bg_method=bg_method, bg_lam=bg_lam, bg_p=bg_p, bg_niter=bg_niter,
            peakwidth=peakwidth,
            ftol=ftol, xtol=xtol, gtol=gtol,
            excludePeakList=excludePeakList, plotLB=False,
            saveTextOutput=saveTextOutput)

        if not resultsFit:
            print(f'  Skipping frame {frame} (fit failed).')
            continue

        ax2.scatter(resultsFit['data_BGsub'][0],
                    resultsFit['data_BGsub'][1] + offset,
                    marker='.', color='lightgrey')
        ax2.plot(resultsFit['data_fit'][0],
                 resultsFit['data_fit'][1] + offset, color='red')
        offset += offsetStep

        for k, phase in enumerate(phases['phases_used']):
            peaks = resultsFit[f'phase_{phase}_peakPosition_fit']
            frames = [frame * collectionTime] * len(peaks)
            ax1.scatter(peaks, frames, color=colours[phase], s=0.7)

        for i, phase in enumerate(phases['phases_Number_array']):
            try:
                result_lists[f'phase_list_{i}'].append(
                    resultsFit[f'phase_{phase}_unit_cell_fit'])
                result_lists[f'P_list_{i}'].append(
                    resultsFit[f'phase_{phase}_P_fit'])
                result_lists[f't_list_{i}'].append(frame * collectionTime)
                result_lists[f'error_list_{i}'].append(
                    resultsFit[f'phase_{phase}_unit_cell_error'])
                result_lists[f'Rw_list_{i}'].append(resultsFit['Rw'])
            except KeyError:
                pass

        # Propagate to the next frame
        if method == 'sequential':
            for phase in phases['phases_used']:
                update_phases(frame, phases, phase, framePressureGuess,
                              deltaP=deltaP, resultsFit=resultsFit,
                              resultsFitUC=resultsFit[
                                  f'phase_{phase}_unit_cell_fit'],
                              method='sequential')
        elif method == 'pressureWalk':
            for phase in phases['phases_used']:
                update_phases(frame, phases, phase, framePressureGuess,
                              deltaP=deltaP, resultsFit=resultsFit,
                              resultsFitUC=resultsFit[
                                  f'phase_{phase}_unit_cell_fit'],
                              method='pressureWalk')

    if save_path:
        plt.savefig(save_path + 'Fit.pdf', format='pdf')
    plt.show()

    # Per-phase summary text files (with proper per-parameter errors)
    if save_path:
        for i, phase in enumerate(phases['phases_Number_array']):
            with open(save_path + f'UnitCell_Fit_{phase}.txt', 'w') as f:
                f.write('time\tPressure (GPa)\tRw\t'
                        'a\tb\tc\talpha\tbeta\tgamma\t'
                        'da\tdb\tdc\tdalpha\tdbeta\tdgamma\n')
                for t, P, Rw, UC, dUC in zip(
                        result_lists[f't_list_{i}'],
                        result_lists[f'P_list_{i}'],
                        result_lists[f'Rw_list_{i}'],
                        result_lists[f'phase_list_{i}'],
                        result_lists[f'error_list_{i}']):
                    uc_str  = '\t'.join(f'{v:.6f}' for v in UC)
                    err_str = '\t'.join(f'{v:.6f}' for v in dUC)
                    f.write(f'{t:.3f}\t{P:.3f}\t{Rw:.4g}\t'
                            f'{uc_str}\t{err_str}\n')

    return result_lists


###############################################################################
# Single-peak fitting (script-only - GUI does this differently)
###############################################################################
def find_matching_HKL(reflectionList, HKL_input):
    """
    Find the 2theta (or d) value of the reflection whose [H, K, L] match
    HKL_input. Returns None if no row matches.
    """
    reflectionList = np.array(reflectionList, dtype=float)
    HKL_input = np.array(HKL_input, dtype=float)
    matches = np.all(np.isclose(reflectionList[:, 1:], HKL_input), axis=1)
    matched = reflectionList[matches, 0]
    if matched.size > 0:
        return matched[0] if len(matched) == 1 else matched
    print('No matching HKL found')
    return None


def initialise_parameters_single_peak(twoThetaGuess, ampGuess=1, sigGuess=0.05):
    """lmfit Parameters for a single Gaussian peak."""
    params = Parameters()
    params.add('BG',  value=0, min=-.5, max=.5)
    params['BG'].set(vary=False)
    params.add('cen', value=twoThetaGuess)
    params.add('amp', value=ampGuess, min=ampGuess * 0.0, max=ampGuess * 1.5)
    params.add('sig', value=sigGuess, min=sigGuess * 0.0, max=sigGuess * 1.5)
    print('Initialising Parameters')
    return params


def single_peak_fit_Model(params, x, y):
    """Single-Gaussian residual."""
    BG  = params['BG'].value
    cen = params['cen'].value
    amp = params['amp'].value
    sig = params['sig'].value
    return BG + amp * np.exp(-0.5 * ((x - cen) / sig) ** 2) - y


def fit_peak_single(data_path, phases, poni_file, save_path,
                    frame, twoThetaMin, twoThetaMax,
                    framePressureGuess,
                    phase_used=0, HKL_input=(0, 0, 0),
                    prominence=0.10, height=0.08, order=1,
                    ampGuess=1, sigGuess=0.05,
                    excludePeakList=None, plotFit=False,
                    bg_method='POLY', bg_lam=1e5, bg_p=0.01, bg_niter=10,
                    peakwidth=0.35,
                    saveTextOutput=True):
    """
    Background-subtract one frame and fit a single Gaussian to the
    reflection that matches HKL_input. Returns a result dictionary in
    the same shape as v0 plus per-parameter errors.
    """
    if excludePeakList is None:
        excludePeakList = []
    if list(HKL_input) == [0, 0, 0]:
        print('No HKL value specified')
        return {}

    # Load + BG (same dispatch as the LB driver)
    twoTheta, valueInt, errorInt = load_frame_fast(
        frame, data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)

    if bg_method.upper() == 'ALS':
        files = sorted(glob.glob(os.path.join(data_path, "*")),
                       key=extract_number)
        files = [f for f in files if extract_number(f) != float('inf')]
        try:
            twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = \
                backgroundFit_ALS(files[frame - 1],
                                  twoThetaMin=twoThetaMin,
                                  twoThetaMax=twoThetaMax,
                                  lam=bg_lam, p=bg_p, niter=bg_niter)
        except Exception as e:
            print(f'ALS BG failed: {e} - falling back to flat BG.')
            valueBG = np.zeros_like(twoTheta)
            twoTheta, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)
    else:
        try:
            twoTheta, valueBG = backgroundFit(
                data_path, frame=frame,
                twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                peakSearchAuto=True, plotBG=False,
                prominence=prominence, height=height,
                order=order, peakwidth=peakwidth,
                excludePeakList=excludePeakList)
        except Exception as e:
            print(f'No Background subtraction performed ({e}).')
            valueBG = np.zeros_like(twoTheta)
        twoTheta, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)

    # Build the reflection list and find the right peak's 2theta
    HKL = phases[f'phase_{phase_used}_HKL']
    _, twoTheta_reflection_list = reflection_List(
        phases[f'phase_{phase_used}_unit_cell'], HKL,
        WL=load_poni(poni_file),
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        crystal_system=phases[f'phase_{phase_used}_crystal_system'])
    print(f'Reflections for unit cell {phase_used} are:')
    print(twoTheta_reflection_list)
    twoTheta_HKL = find_matching_HKL(twoTheta_reflection_list, HKL_input)
    if twoTheta_HKL is None:
        print('Cannot fit single peak - HKL not in this UC reflection list.')
        return {}
    print(f'\nPeak fitted for the {list(HKL_input)} reflection.')

    if plotFit:
        plt.scatter(twoTheta, valueIntBGsub, marker='.', color='gray', s=3)

    # Fit
    params = initialise_parameters_single_peak(
        twoTheta_HKL, ampGuess=ampGuess, sigGuess=sigGuess)
    try:
        result = minimize(
            lambda p: single_peak_fit_Model(p, twoTheta, valueIntBGsub),
            params, method='least_squares')
        print(f'\nFitting success: {result.success}')
    except Exception as e:
        print(f'Fitting error in frame {frame}: {e}')
        return {}

    BG  = result.params['BG'].value
    cen = result.params['cen'].value
    amp = result.params['amp'].value
    sig = result.params['sig'].value
    cen_err = result.params['cen'].stderr
    amp_err = result.params['amp'].stderr
    sig_err = result.params['sig'].stderr

    best_fit = amp * np.exp(-0.5 * ((twoTheta - cen) / sig) ** 2) + BG
    print(f'Fit parameters:\t{[cen, amp, sig]}')
    print(f'Fit errors:\t{[cen_err, amp_err, sig_err]}')

    if save_path and saveTextOutput:
        subfolder_path = save_path + 'fits_singlePeak'
        os.makedirs(subfolder_path, exist_ok=True)
        with open(os.path.join(subfolder_path, f'frame_{frame}.txt'),
                  'w') as f:
            f.write(f'Frame {frame}\n2Theta\tInt\tFit\n')
            np.savetxt(f,
                       np.column_stack((twoTheta, valueIntBGsub, best_fit)),
                       fmt='%.6f', delimiter='\t')

    if plotFit:
        plt.plot(twoTheta, best_fit, c='red', ls='-', lw=1, label='Best fit')
        plt.xlim(left=twoThetaMin); plt.xlim(right=twoThetaMax)
        plt.legend()
        if save_path:
            os.makedirs(save_path + 'fits', exist_ok=True)
            plt.savefig(save_path + 'fits/' + f'frame_{frame}.png',
                        format='png')
        plt.show()
        plt.clf()

    return {
        'fitted_HKL':    np.array(HKL_input),
        'fitParameters': np.array([cen, amp, sig]),
        'fitErrors':     np.array([cen_err, amp_err, sig_err]),
        'data_BGsub':    np.array([twoTheta, valueIntBGsub]),
        'data_fit':      np.array([twoTheta, best_fit]),
    }


def plot_peak_fit_single(data_path, phases, poni_file, save_path,
                         phase_rules=None,
                         twoThetaMin=0, twoThetaMax=30,
                         framePressureGuess=None,
                         collectionTime=1, ylabel='Frame',
                         frameStart=40, frameEnd=90, frameStep=1,
                         phase_used=0, HKL_input=(1, 1, 1),
                         plotFit=False, logPlot=True,
                         CorrNegativeValues=0.02,
                         prominence=0.10, height=0.08, order=1,
                         excludePeakList=None,
                         ampGuess=1, sigGuess=0.05,
                         method='pressureWalk', deltaP=0.5,
                         offsetStep=0.05,
                         bg_method='POLY', bg_lam=1e5, bg_p=0.01,
                         bg_niter=10, peakwidth=0.35,
                         figsize=(8, 8), pcolor_cmap='binary_r',
                         saveTextOutput=True):
    """
    Batch single-peak fit + waterfall figure. Mirrors plot_LB_fit but
    operates on one HKL of one phase instead of running the full
    Le Bail refinement.
    """
    if phase_rules is None:
        phase_rules = [(lambda f: f >= 0, [0])]
    if excludePeakList is None:
        excludePeakList = []
    if framePressureGuess is None:
        framePressureGuess = [[0, 0], [3000, 0]]
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    spec = fig.add_gridspec(3, 4)

    # Top - waterfall
    ax1 = fig.add_subplot(spec[0:1, :2])
    XRD_patterns = {'frame': [], 'twoTheta': [], 'valueInt': []}
    for frame in np.arange(frameStart, frameEnd + 1, frameStep):
        twoTheta, valueIntBGsub, _ = load_frame_fast(
            frame, data_path, twoThetaMin, twoThetaMax)
        XRD_patterns['frame'].append([frame * collectionTime])
        if logPlot:
            valueIntBGsub = np.log(np.asarray(valueIntBGsub) + CorrNegativeValues)
        XRD_patterns['twoTheta'].append(twoTheta)
        XRD_patterns['valueInt'].append(valueIntBGsub)

    im = ax1.pcolormesh(XRD_patterns['twoTheta'],
                        XRD_patterns['frame'],
                        XRD_patterns['valueInt'],
                        cmap=pcolor_cmap)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label('Intensity (counts)')
    ax1.set(ylabel=ylabel)
    ax1.set_aspect('auto')
    ax1.set_xlim((twoThetaMin, twoThetaMax))

    # Bottom - per-frame fits
    ax2 = fig.add_subplot(spec[1:3, :2])
    ax2.set(ylabel='Intensity (arb.u.)')
    ax2.set(xlabel='Two Theta (deg)')
    ax2.set_xlim((twoThetaMin, twoThetaMax))

    offset = 0
    frameList = list(range(frameStart, frameEnd + 1, frameStep))

    cen_list, cen_err_list, frame_list_out = [], [], []

    for frame in frameList:
        specify_phases(phases, phase_rules, frame)

        if method == 'pressureHelper':
            for phase in phases['phases_used']:
                update_unitCell(phases[f'phase_{phase}_unit_cell_0'],
                                phases, phase)
                update_phases(frame, phases, phase, framePressureGuess,
                              method='pressureHelper')

        resultsFit = fit_peak_single(
            data_path, phases, poni_file, save_path,
            frame, twoThetaMin, twoThetaMax, framePressureGuess,
            phase_used, HKL_input,
            prominence, height, order,
            ampGuess, sigGuess,
            excludePeakList=excludePeakList, plotFit=plotFit,
            bg_method=bg_method, bg_lam=bg_lam, bg_p=bg_p, bg_niter=bg_niter,
            peakwidth=peakwidth, saveTextOutput=saveTextOutput)

        if not resultsFit:
            print(f'  Skipping frame {frame} (single-peak fit failed).')
            continue

        ax1.scatter(resultsFit['fitParameters'][0],
                    frame * collectionTime,
                    color='red', marker='.', s=0.7,
                    label=str(resultsFit['fitted_HKL']))

        offset += offsetStep
        ax2.scatter(resultsFit['data_BGsub'][0],
                    resultsFit['data_BGsub'][1] + offset,
                    marker='.', color='lightgrey')
        ax2.plot(resultsFit['data_fit'][0],
                 resultsFit['data_fit'][1] + offset, color='red')
        offset += offsetStep

        cen_list.append(float(resultsFit['fitParameters'][0]))
        cen_err = resultsFit['fitErrors'][0]
        cen_err_list.append(float(cen_err) if cen_err is not None
                            else float('nan'))
        frame_list_out.append(frame * collectionTime)

    if save_path:
        plt.savefig(save_path + 'singleFit.pdf', format='pdf')
    plt.show()

    # Summary text file with per-frame peak position + error
    if save_path:
        with open(save_path
                  + f'SinglePeak_HKL_{HKL_input[0]}{HKL_input[1]}'
                  + f'{HKL_input[2]}_phase_{phase_used}.txt', 'w') as f:
            f.write('time\tcen (deg)\tcen_err (deg)\n')
            for t, c, ce in zip(frame_list_out, cen_list, cen_err_list):
                f.write(f'{t:.3f}\t{c:.6f}\t{ce:.6f}\n')

    return {'time': frame_list_out, 'cen': cen_list, 'cen_err': cen_err_list}
