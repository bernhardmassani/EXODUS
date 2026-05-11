"""
EXODUS - Equation-of-state X-ray Observation and Diffraction Unit-cell Solver
Script edition (v1) - May 2026

@ Dr Bernhard Massani

This file is the script-style entry point for publication-quality figures.
It mirrors the Exodus layout (toggle-able blocks at the bottom) and uses
the script_*_toolbox modules in this directory; those toolboxes carry
every physics / algorithm improvement from the GUI EXODUS codebase plus a
handful of publication-only helpers (plot_LB_fit, plot_peak_fit_single,
specify_transitionPressure, calculate_isotherms).

Improvements over v0
--------------------
Physics / maths (verified)
  *  TRICLINIC d-spacing uses the standard S_ij metric-tensor form
     (Cullity App. 3). The old form was missing one lattice-constant
     factor on every term and was off by ~order-of-magnitude even in
     the orthorhombic limit. Now agrees with the master formula to
     <5e-16 relative error for every test cell.
  *  RHOMBOHEDRAL d-spacing: '- cos(alpha)' moved INSIDE the (HK+KL+HL)
     bracket so the (cos^2 - cos) factor multiplies each cross term.
  *  MONOCLINIC d-spacing: math.cos(b) -> math.cos(bet) (was passing
     the lattice LENGTH b to cos()).
  *  BM3 round-trip P -> V -> P is exact to <1e-12 GPa.
  *  unitCellVolume returns NaN (not 1.0) on a non-physical radicand,
     so geometry errors surface instead of silently producing finite
     but meaningless pressures.

Engineering
  *  Multi-format data loading (.fxye, .xy, .dat, .chi).
  *  Optional ALS background subtraction (bg_method='ALS').
  *  Configurable sigma model (caglioti / fixed / per_phase / separate).
  *  Per-parameter unit-cell errors written to UnitCell_Fit_<phase>.txt.
  *  Amp parameters are now HKL-INDEX keyed (one per HKL row), with
     out-of-window AND zero-intensity reflections frozen. Keeps the
     Jacobian full rank, so lmfit returns a real covariance matrix
     and per-parameter stderrs.
  *  LB_fit_Model slow path (rhombohedral / orthorhombic / monoclinic /
     triclinic) now rebuilds HKL indices by (H,K,L) lookup, matching
     the keying used at parameter creation. Previously the slow path
     used np.arange(n_refl), which collided with HKL index as soon as
     any reflection was filtered out of the fit window.

Dependencies:
    pip install lmfit
    or:  conda install -c conda-forge lmfit
"""

import time
import sys
import numpy as np

import script_EoS_toolbox      as EoS
import script_BatchFit_toolbox as tb
import script_Constants        as const

np.set_printoptions(threshold=sys.maxsize)


###############################################################################
#                              GENERAL INPUT                                  #
###############################################################################
totalFrames    = 50      # Total number of frames in the dataset
collectionTime = 1.5     # Collection time per frame (any units; affects time-axis)
start = time.time()

# 2theta window (degrees)
twoThetaMin, twoThetaMax = 8, 18

# Folder containing the *.fxye / *.xy / *.dat / *.chi pattern files
data_path = './PressureWalk_PressureHelper_data/Data_BG_sub/'

# pyFAI poni file - used to read the X-ray wavelength
poni_file = 'C:/Users/bmass/Desktop/University/Projects/H2O_dDAC/Data/LaB6_calibration_06_09_23_1900h.poni'

# JCPDS files - one per candidate phase
JCPDS = [
    "C:/Users/bmass/Desktop/University/Projects/H2O_dDAC/Data/au_Fei.jcpds",
    "C:/Users/bmass/Desktop/University/Projects/H2O_dDAC/Data/iceVII_2_vit.jcpds",
    "C:/Users/bmass/Desktop/University/Projects/H2O_dDAC/Data/IceVI_T.jcpds",
]

# Output directory for figures, fits/, and the per-phase UnitCell_Fit_*.txt
save_path = './PressureWalk_PressureHelper_data/PressureWalk_Analysis/'


###############################################################################
#                            BACKGROUND SETTINGS                              #
###############################################################################
# Two background methods are available:
#   bg_method = 'POLY' - polynomial fit through peak-excluded regions (v0)
#   bg_method = 'ALS'  - asymmetric least squares (handles overlapping peaks)
bg_method = 'POLY'

# --- Polynomial BG knobs (used when bg_method == 'POLY') ---
peakSearchAuto  = True
prominence      = 0.10
height          = 0.08
order           = 20
peakwidth       = 0.35    # half-width (deg) of the peak-exclusion windows
excludePeakList = []      # advanced: hand-pick exclusion regions

# --- ALS BG knobs (used when bg_method == 'ALS') ---
als_lam   = 1e5     # smoothness  (1e4 - 1e7)
als_p     = 0.01    # asymmetry   (0.001 - 0.1)
als_niter = 10      # iterations  (10 - 20)


###############################################################################
#                              PEAK FITTING                                   #
###############################################################################
# Initial amplitude (=ampGuess/100 * jcpds intensity) and width guess.
# The toolbox sets per-peak min/max around these values.
ampGuess = 1
sigGuess = 0.05

# Peak-width model:
#   'caglioti'  - one global base sigma; angle-dependent via 1/cos(theta)
#   'fixed'     - one global sigma for all peaks (cheapest)
#   'per_phase' - one sigma per phase
#   'separate'  - one sigma per peak (most flexible, slowest)
sigma_mode   = 'caglioti'
sigmaBounds  = 2.0    # sigma_max = sigGuess * sigmaBounds
ampPrefactor = 1.0    # multiplies max(I) when scaling ampGuess
maxShift     = 0.5    # lattice-parameter freedom; 0=frozen, 0.5 = +/-50%

# scipy.optimize.least_squares tolerances (1e-5 is ~2.5x faster than 1e-8
# at the same lattice-parameter accuracy in our regression tests)
ftol, xtol, gtol = 1e-5, 1e-5, 1e-5


###############################################################################
#                              SPECIFY PHASES                                 #
###############################################################################
# Which phases participate in which frames. The first matching predicate wins.
phase_rules = [(lambda f: f <= 48, [0]),
               (lambda f: f == 49, [0, 2]),
               (lambda f: f == 50, [0, 1, 2]),
               (lambda f: f  > 50, [0, 1])]

# Pressure (GPa) at which each phase first appears - used to seed the unit
# cell of a high-pressure phase before its first refinement frame.
transition_pressures = {0: 0.0,
                        1: 2.1,
                        2: 1.2}


###############################################################################
#                          BATCH-FIT MODE OPTIONS                             #
###############################################################################
# 'sequential'     - last fit's UC is the next frame's starting UC
# 'pressureWalk'   - last fit's pressure + delta_P -> next frame's UC via BM3
# 'pressureHelper' - per-frame pressure interpolated from framePressureGuess
delta_P = 0.5    # GPa - only used in 'pressureWalk' mode

# Anchor (frame, P_GPa) points for the 'pressureHelper' interpolation
framePressureGuess = np.array([[0, 0.5], [48, 1.4], [49, 1.8], [50, 1.9],
                               [52, 1.9], [53, 4],  [58, 12], [64, 27],
                               [67, 33.5], [82, 56], [100, 56]])


###############################################################################
###############################################################################
##                                                                           ##
##                            USEFUL FUNCTIONS                               ##
##                                                                           ##
##  Each block is gated by its own bool. Toggle ONE on at a time.            ##
##                                                                           ##
##  Result dictionaries (resultsFit) provide:                                ##
##      resultsFit['phase_<i>_unit_cell_fit']                                ##
##      resultsFit['phase_<i>_unit_cell_error']  <- now per-parameter        ##
##      resultsFit['phase_<i>_V_fit']                                        ##
##      resultsFit['phase_<i>_P_fit']                                        ##
##      resultsFit['phase_<i>_BG_fit']                                       ##
##      resultsFit['phase_<i>_peakPosition_fit']                             ##
##      resultsFit['data_BGsub'], resultsFit['data_fit'], resultsFit['Rw']   ##
##                                                                           ##
###############################################################################
###############################################################################


###############################################################################
# 1. Background sanity check - look at ONE pattern's BG subtraction
###############################################################################
BG_sub = False
# BG_sub = True
if BG_sub:
    twoTheta, valueBG = tb.backgroundFit(
        data_path, frame=50,
        twoThetaMin=5, twoThetaMax=20,
        peakSearchAuto=peakSearchAuto, plotBG=True,
        prominence=prominence, height=height,
        order=order, peakwidth=peakwidth,
        excludePeakList=excludePeakList)


###############################################################################
# 2. Indexing - overlay the JCPDS peak positions on ONE frame (no fit)
###############################################################################
index_pattern = False
# index_pattern = True
if index_pattern:
    frame  = 49
    phases = tb.load_JCPDS(JCPDS)
    tb.specify_phases(phases, phase_rules, frame)

    for phase in phases['phases_used']:
        tb.update_phases(frame, phases, phase, framePressureGuess,
                         method='pressureHelper')

    tb.index_LB(data_path, phases, poni_file, save_path,
                frame=frame,
                twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                plotIndex=True,
                framePressureGuess=[[0, 0], [3000, 0]],
                prominence=prominence, height=height, order=order,
                peakwidth=peakwidth, excludePeakList=excludePeakList,
                ampGuess=ampGuess, sigGuess=sigGuess,
                sigma_mode=sigma_mode, sigmaBounds=sigmaBounds,
                ampPrefactor=ampPrefactor, maxShift=maxShift,
                bg_method=bg_method,
                bg_lam=als_lam, bg_p=als_p, bg_niter=als_niter)


###############################################################################
# 3. Single-frame Le Bail fit on ONE frame
###############################################################################
simpleTest = False
# simpleTest = True
if simpleTest:
    frame  = 52
    phases = tb.load_JCPDS(JCPDS)
    tb.specify_phases(phases, phase_rules, frame)

    for phase in phases['phases_used']:
        tb.update_phases(frame, phases, phase, framePressureGuess,
                         method='pressureHelper')

    resultsFit = tb.fit_LB(
        data_path, phases, poni_file, save_path,
        frame=frame,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        plotLB=True,
        framePressureGuess=framePressureGuess,
        prominence=prominence, height=height, order=order,
        peakwidth=peakwidth, excludePeakList=excludePeakList,
        ampGuess=ampGuess, sigGuess=sigGuess,
        sigma_mode=sigma_mode, sigmaBounds=sigmaBounds,
        ampPrefactor=ampPrefactor, maxShift=maxShift,
        bg_method=bg_method,
        bg_lam=als_lam, bg_p=als_p, bg_niter=als_niter,
        ftol=ftol, xtol=xtol, gtol=gtol)

    print(f'Pressure in this frame is '
          f'{resultsFit["phase_0_P_fit"]:.3f} GPa.')


###############################################################################
# 4. BATCH Le Bail fit - the publication figure
###############################################################################
fitAll = False
# fitAll = True
if fitAll:
    phases = tb.load_JCPDS(JCPDS)

    # Seed each high-pressure phase at its onset pressure
    for phase in phases['phases_Number_array']:
        P_trans = transition_pressures.get(phase, 0.0)
        EoS.specify_transitionPressure(phases, phase, P_trans)

    tb.plot_LB_fit(
        data_path, phases, poni_file, save_path,
        framePressureGuess=framePressureGuess,
        phase_rules=phase_rules,
        collectionTime=collectionTime, ylabel='time (ms)',
        frameStart=40, frameEnd=90, frameStep=1,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        plotLB=False, logPlot=True, CorrNegativeValues=0.02,
        prominence=prominence, height=height, order=order,
        peakwidth=peakwidth, excludePeakList=excludePeakList,
        ampGuess=ampGuess, sigGuess=sigGuess,
        sigma_mode=sigma_mode, sigmaBounds=sigmaBounds,
        ampPrefactor=ampPrefactor, maxShift=maxShift,
        bg_method=bg_method,
        bg_lam=als_lam, bg_p=als_p, bg_niter=als_niter,
        ftol=ftol, xtol=xtol, gtol=gtol,
        method='pressureHelper', deltaP=delta_P,
        offsetStep=0.05)


###############################################################################
# 5. Single-peak fit on ONE frame (one HKL of one phase)
###############################################################################
singleTest = False
# singleTest = True
if singleTest:
    frame  = 50
    phases = tb.load_JCPDS(JCPDS)
    tb.specify_phases(phases, phase_rules, frame)

    for phase in phases['phases_used']:
        tb.update_phases(frame, phases, phase, framePressureGuess,
                         method='pressureHelper')

    resultsFit = tb.fit_peak_single(
        data_path, phases, poni_file, save_path,
        frame, twoThetaMin, twoThetaMax, framePressureGuess,
        phase_used=1, HKL_input=[1, 1, 0],
        prominence=prominence, height=height, order=order,
        peakwidth=peakwidth, excludePeakList=excludePeakList,
        ampGuess=ampGuess, sigGuess=sigGuess,
        bg_method=bg_method,
        bg_lam=als_lam, bg_p=als_p, bg_niter=als_niter,
        plotFit=True)


###############################################################################
# 6. BATCH single-peak fit (one HKL across many frames)
###############################################################################
singleAll = False
singleAll = True
if singleAll:
    phases = tb.load_JCPDS(JCPDS)

    for phase in phases['phases_Number_array']:
        P_trans = transition_pressures.get(phase, 0.0)
        EoS.specify_transitionPressure(phases, phase, P_trans)

    tb.plot_peak_fit_single(
        data_path, phases, poni_file, save_path,
        phase_rules=phase_rules,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        framePressureGuess=framePressureGuess,
        collectionTime=collectionTime, ylabel='time (ms)',
        frameStart=40, frameEnd=90, frameStep=1,
        phase_used=0, HKL_input=[1, 1, 1],
        plotFit=False, logPlot=True, CorrNegativeValues=0.02,
        prominence=prominence, height=height, order=order,
        peakwidth=peakwidth, excludePeakList=excludePeakList,
        ampGuess=ampGuess, sigGuess=sigGuess,
        bg_method=bg_method,
        bg_lam=als_lam, bg_p=als_p, bg_niter=als_niter,
        method='pressureHelper', deltaP=delta_P,
        offsetStep=0.05)


print('\n\t It took', round(time.time() - start),
      'seconds to run this script.')
