# EXODUS_core.py for EXODUS v0.1.4b
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
# Workflow layer between the GUI (EXODUS_main.py) and the toolboxes: runs
# ONE Pawley fit of one background-subtracted pattern and packs everything
# the GUI and the output files need into a results dictionary.
#
#   EXODUS_main.py            GUI only
#   EXODUS_core.py            workflow (this file)
#   *_toolbox.py              maths / physics / file formats / pop-ups
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import numpy as np
from lmfit import minimize

import crystallography_toolbox as cryst
import eos_toolbox as eos
import pawley_toolbox as pawley
import statistics_toolbox as stats


# ============================================================================
#                                SETTINGS
# ============================================================================
# Error reported for a lattice parameter whose standard error could not be
# obtained because the covariance matrix failed (v0.1.4b). Up to v0.1.4 the
# global sqrt(reduced chi^2) was used instead; that route is kept, commented
# out, in fit_pawley below.
COVARIANCE_FAILURE_ERROR = 0.0

# A phase is flagged as LOST in a frame (results['_phase_flags']) if the fit
# did not keep it in the pattern: every refined amplitude is below
# LOST_AMP_FRACTION * max|I|, no reflection of it is refined in the 2theta
# window, or a free lattice parameter ended on its +-maxShift bound. The
# batch fit does not use the cell of a lost phase as start for the next
# frame (EXODUS_main._process_next_frame).
LOST_AMP_FRACTION = 1e-4


# ============================================================================
#                               TICK MARKS
# ============================================================================
def calculate_tickmarks(phases, WL, twoThetaMin, twoThetaMax, i):
    """Reflection list [2theta, H, K, L] of phase i for its current
    (starting) unit cell inside the 2theta window."""
    _dk, twoTheta_reflection_list = cryst.reflection_list(
        list(phases[f'phase_{i}_unit_cell']),
        HKL=phases[f'phase_{i}_HKL'],
        crystalSystem=phases[f'phase_{i}_crystal_system'],
        WL=WL,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
    return twoTheta_reflection_list


# ============================================================================
#                              PAWLEY FIT
# ============================================================================
def _covariance_priority(name):
    """Order in which stats.robust_covariance keeps parameters: lattice
    parameters first, then global profile parameters, per-phase and
    per-reflection widths, and amplitudes last."""
    stem = name.split('_', 1)[0]
    if stem in cryst.LATTICE_NAMES:
        return 0
    if name in ('BG', 'sig_global', 'cag_U', 'cag_V', 'cag_W', 'eta'):
        return 1
    if stem == 'sig':
        return 2
    return 3


def phase_fit_flag(params, phase, yScale):
    """Reason why `phase` was lost by the fit, or None if it looks healthy
    (see LOST_AMP_FRACTION)."""
    amps = [p for n, p in params.items()
            if n.startswith(f'amp_{phase}_') and p.vary]
    if not amps:
        return 'no reflection refined in the 2theta window'
    if yScale > 0 and max(float(p.value) for p in amps) < LOST_AMP_FRACTION * yScale:
        return 'all amplitudes ~0'
    for nm in cryst.LATTICE_NAMES:
        p = params.get(f'{nm}_{phase}')
        if p is None or not p.vary or p.expr:
            continue
        tol = 1e-6 * max(abs(float(p.value)), 1.0)
        if (np.isfinite(p.min) and abs(p.value - p.min) < tol) or \
           (np.isfinite(p.max) and abs(p.value - p.max) < tol):
            return f'{nm} on its max-shift bound'
    return None


def fit_pawley(twoTheta, valueIntBGsub, phases, WL,
           frame=10, twoThetaMin=0, twoThetaMax=10,
           ampGuess=1, sigGuess=0.05,
           sigmaMode='scherrer', sigmaBounds=2.0, ampBounds=3.0,
           ampPrefactor=1.0, maxShift=0.5,
           ftol=1e-5, xtol=1e-5, gtol=1e-5,
           valueIntRaw=None, profile='gaussian',
           etaGuess=pawley.DEFAULT_ETA,
           cagliotiUVW=pawley.DEFAULT_CAGLIOTI_UVW):
    '''
    Pawley fit of one background-subtracted pattern.

    twoTheta, valueIntBGsub : pattern on an ascending 2theta grid
    phases        : EXODUS phases dict; phases['phases_used'] are refined
    WL            : wavelength (A)
    ampPrefactor  : start amplitude scale = max(valueIntBGsub) * ampPrefactor
                    (ampGuess is kept for API compatibility and not used)
    sigmaMode, sigGuess, sigmaBounds, ampBounds, maxShift, profile,
    etaGuess, cagliotiUVW : see pawley_toolbox.initialise_parameters
    ftol, xtol, gtol : convergence tolerances passed to scipy's
                    least_squares (via lmfit)
    valueIntRaw   : raw intensities (before background subtraction) on the
                    same grid; only used for the Poisson weights of
                    Rwp / chi^2 (statistics_toolbox.profile_statistics)

    Returns a dict (empty if the fit failed) with, per used phase X:
        phase_X_unit_cell_fit, phase_X_unit_cell_error,
        phase_X_V_fit, phase_X_V_error, phase_X_P_fit, phase_X_P_error,
        phase_X_peakPosition_fit
    and the frame-wide entries data_fit, data_BGsub, data_raw, Rp, Rwp, ...
    '_phases_used' is the list of phases refined in THIS fit (the output
    files use it, not the global phases['phases_used'], which changes from
    frame to frame in a batch run).
    '_errors_fallback' is True if at least one lattice-parameter error could
    not be obtained from the covariance matrix; such errors (and the V / P
    errors that depend on them) are set to COVARIANCE_FAILURE_ERROR (0).
    '_phase_flags' maps a phase index to the reason it was lost (see
    phase_fit_flag); healthy phases are not listed.
    '''
    twoTheta = np.asarray(twoTheta, dtype=float)
    valueIntBGsub = np.asarray(valueIntBGsub, dtype=float)
    # The windowed peak evaluation needs an ascending 2theta grid.
    if twoTheta.size > 1 and np.any(np.diff(twoTheta) <= 0):
        order = np.argsort(twoTheta, kind='stable')
        twoTheta, valueIntBGsub = twoTheta[order], valueIntBGsub[order]
        if valueIntRaw is not None:
            valueIntRaw = np.asarray(valueIntRaw, dtype=float)[order]

    # Only reflections inside the data can be refined: a reflection between
    # the last data point and the window edge would get an amplitude with
    # no effect on the pattern (singular fit).
    if twoTheta.size:
        twoThetaMin = max(float(twoThetaMin), float(twoTheta[0]))
        twoThetaMax = min(float(twoThetaMax), float(twoTheta[-1]))

    # Intensity scale of the pattern: bounds the constant background offset
    # (pawley.BG_OFFSET_FRACTION of it), so the bound is independent of
    # whether the data are counts or normalised intensities.
    yScale = float(np.max(np.abs(valueIntBGsub))) if valueIntBGsub.size else 1.0

    params = pawley.initialise_parameters(
        phases, WL,
        bgBound=pawley.BG_OFFSET_FRACTION * yScale,
        ampGuess=max(valueIntBGsub) * ampPrefactor,
        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
        sigmaMode=sigmaMode, sigGuess=sigGuess,
        sigmaBounds=sigmaBounds, ampBounds=ampBounds, maxShift=maxShift,
        profile=profile, etaGuess=etaGuess, cagliotiUVW=cagliotiUVW)

    # Lattice parameters held fixed in THIS fit (symmetry-free ones only),
    # reported in the console and the *_fitParam header, so per-frame
    # fixes ('Fix' column / 'Edit' pop-up) can be checked.
    fixedLattice = {}
    for p in phases['phases_used']:
        flags = cryst.free_lattice_flags(phases[f'phase_{p}_crystal_system'])
        names = [nm for i, nm in enumerate(cryst.LATTICE_NAMES)
                 if flags[i] and f'{nm}_{p}' in params
                 and not params[f'{nm}_{p}'].vary]
        if names:
            fixedLattice[int(p)] = names
            print(f'Frame {frame}, phase {p}: lattice parameters held fixed '
                  f'-> {", ".join(names)}')

    stencils = {p: cryst.hkl_stencil(phases[f'phase_{p}_HKL'])
                for p in phases['phases_used']}

    try:
        result = minimize(
            lambda prm: pawley.pawley_residual(prm, twoTheta, valueIntBGsub,
                                            phases, twoThetaMin, twoThetaMax,
                                            sigmaMode=sigmaMode,
                                            profile=profile,
                                            stencils=stencils),
            params,
            method='least_squares',
            calc_covar=True,
            ftol=ftol, xtol=xtol, gtol=gtol,
            # Scale every parameter by the norm of its Jacobian column.
            # Without it the trust region treats a step of 1e-3 in the
            # Caglioti U, V, W (deg^2) like a step of 1e-3 A in a lattice
            # parameter, and the Caglioti fit can stop in a wrong minimum
            # (broad peaks, half the amplitudes) from a starting cell that
            # is only 0.3 % off.
            x_scale='jac')
    except Exception as e:
        print(f'Error in frame {frame}: {e}')
        return {}
    if result is None:
        return {}

    # Covariance: if lmfit's is missing or (near) singular, rebuild it from
    # the Jacobian without the parameters the data cannot determine.
    excluded = {}
    if not stats.covariance_ok(result) and getattr(result, 'jac', None) is not None:
        excluded = stats.robust_covariance(result, priority=_covariance_priority)
        if excluded:
            print('Covariance: left out (treated as fixed) -> '
                  + ', '.join(f'{n} ({why})' for n, why in excluded.items()))

    # Fitted curve (exactly the refined model)
    bestFit, _components = pawley.create_fit_components(
        twoTheta, phases, result.params, twoThetaMin, twoThetaMax,
        sigmaMode=sigmaMode, profile=profile)

    # Fit statistics. fit_statistics also returns sqrt(reduced chi^2),
    # which was the error fallback up to v0.1.4 (see below).
    estimatedError = stats.fit_statistics(result, twoTheta)  # noqa: F841
    profileStats = stats.profile_statistics(valueIntBGsub, bestFit,
                                            yRaw=valueIntRaw,
                                            nParams=len(result.var_names))
    print(f"Rp = {profileStats['Rp']:.4g}   Rwp = {profileStats['Rwp']:.4g}   "
          f"chi2 = {profileStats['chi2']:.4g}   "
          f"chi2_red = {profileStats['chi2_red']:.4g}")

    tickArray = pawley.create_ticks(result, phases, twoThetaMin, twoThetaMax, WL)

    resultsFit = {}
    for k in stats.PROFILE_STAT_KEYS:
        resultsFit[k] = profileStats[k]
    resultsFit['weighting'] = profileStats['weighting']
    # lmfit's own (unweighted) sum of squared residuals, kept for reference
    resultsFit['chisqr'] = float(result.chisqr)
    resultsFit['redchi'] = float(result.redchi)
    resultsFit['nfev'] = int(result.nfev)
    resultsFit['data_BGsub'] = np.array([twoTheta, valueIntBGsub])
    resultsFit['data_fit'] = np.array([twoTheta, bestFit])
    if valueIntRaw is not None:
        resultsFit['data_raw'] = np.array([twoTheta, np.asarray(valueIntRaw, dtype=float)])

    # Fit context for the output files (per-peak amplitudes, widths and
    # their standard errors are rebuilt from the lmfit Parameters).
    resultsFit['_lmfit_params'] = result.params
    resultsFit['_sigma_mode'] = sigmaMode
    resultsFit['_profile'] = profile
    resultsFit['_WL'] = float(result.params['WL'].value)
    resultsFit['_twoThetaMin'] = float(twoThetaMin)
    resultsFit['_twoThetaMax'] = float(twoThetaMax)
    # Phases refined in THIS fit (v0.1.4b): the writers of the output files
    # use this list, so 'Save All' writes every phase of every frame.
    resultsFit['_phases_used'] = [int(p) for p in phases['phases_used']]

    usedFallback = False
    phaseFlags = {}
    for i, phase in enumerate(phases['phases_used']):
        uc = pawley.unit_cell_fit(result, phase)
        print('Unit Cell ' + str(phase) + ' fit: \n' + str(uc))
        resultsFit[f'phase_{phase}_unit_cell_fit'] = uc

        # Standard errors from the covariance matrix. If it could not be
        # built (e.g. a parameter sits on a bound), no error can be
        # obtained and it is set to COVARIANCE_FAILURE_ERROR (= 0).
        #
        # Fallback up to v0.1.4 - global sqrt(reduced chi^2) as error:
        # errs, fb = stats.unit_cell_errors(result, phase, fallback=estimatedError)
        errs, fb = stats.unit_cell_errors(result, phase,
                                          fallback=COVARIANCE_FAILURE_ERROR)
        usedFallback = usedFallback or fb
        resultsFit[f'phase_{phase}_unit_cell_error'] = errs

        # Volume and pressure with their errors - computed only here.
        cov = stats.cell_covariance(result.params, phase)
        V, dV = stats.cell_volume_and_error(
            uc, cov, uncErrors=errs,
            crystalSystem=phases[f'phase_{phase}_crystal_system'])
        if fb and not np.isfinite(dV):
            dV = COVARIANCE_FAILURE_ERROR
        comp = phases[f'phase_{phase}_compression_constants']
        P, dP = eos.bm3_pressure_and_error(V, dV, float(comp[0]),
                                           float(comp[1]), float(comp[2]))
        if fb and not np.isfinite(dP):
            dP = COVARIANCE_FAILURE_ERROR
        resultsFit[f'phase_{phase}_V_fit'] = V
        resultsFit[f'phase_{phase}_V_error'] = dV
        resultsFit[f'phase_{phase}_P_fit'] = P
        resultsFit[f'phase_{phase}_P_error'] = dP
        resultsFit[f'phase_{phase}_BG_fit'] = result.params['BG'].value
        resultsFit[f'phase_{phase}_peakPosition_fit'] = tickArray[i]

        flag = phase_fit_flag(result.params, phase, yScale)
        if flag:
            phaseFlags[int(phase)] = flag
            print(f'WARNING: phase {phase} lost in frame {frame}: {flag}')

    resultsFit['_errors_fallback'] = bool(usedFallback)
    resultsFit['_covariance_excluded'] = excluded
    resultsFit['_phase_flags'] = phaseFlags
    resultsFit['_fixed_lattice'] = fixedLattice
    print('\n============================================= \n')
    return resultsFit
