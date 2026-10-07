# pawley_toolbox.py for EXODUS v0.1.4
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
# Whole-pattern Pawley refinement model [1] (no Qt):
#
#   * lmfit parameter set: lattice parameters (with the symmetry ties of each
#     crystal system), one refined amplitude per reflection (exactly
#     overlapping reflections share one amplitude), peak widths
#     according to the chosen broadening model, profile mixing eta, and a
#     constant background offset,
#   * the residual (model - data) minimised by EXODUS_core.fit_pawley,
#   * fitted curve, per-reflection components and tick marks.
#
# Peak profiles (A = peak height, H = HWHM = FWHM/2, u = (2theta - c) / H)
#   'gaussian'      y = A exp(-ln2 u^2)                (= A exp(-x^2 / 2 sigma^2))
#   'pseudo_voigt'  y = A [eta / (1 + u^2) + (1 - eta) exp(-ln2 u^2)]   [2]
#                   (Gaussian and Lorentzian with the same FWHM; one global
#                   eta in [0, 1], held fixed at the value set in the GUI)
#
# Peak broadening (sigma = Gaussian sigma, FWHM = 2 sqrt(2 ln2) sigma)
#   'fixed'      one sigma for all reflections of all phases
#   'scherrer'   sigma = sigma0 / cos(theta) (one global sigma0); the
#                angular dependence of size broadening [3]
#   'caglioti'   FWHM^2 = U tan^2(theta) + V tan(theta) + W (global U, V, W,
#                FWHM in degrees 2theta) [4]
#   'separate'   one sigma per reflection
#   'per_phase'  one sigma per phase
#
# Speed: every reflection is evaluated only on the data points within
# +-PEAK_WINDOW_FWHM[profile] FWHM of its centre (the 2theta grid must be
# ascending, which all supported file formats are).
#
# References
# ----------
# [1] G. S. Pawley, J. Appl. Cryst. 14, 357-361 (1981).
# [2] G. K. Wertheim, M. A. Butler, K. W. West, D. N. E. Buchanan,
#     Rev. Sci. Instrum. 45, 1369-1371 (1974).
# [3] P. Scherrer, Nachr. Ges. Wiss. Goettingen, Math.-Phys. Kl.,
#     98-100 (1918).
# [4] G. Caglioti, A. Paoletti, F. P. Ricci, Nucl. Instrum. 3, 223-228
#     (1958).
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import numpy as np
from lmfit import Parameters

import crystallography_toolbox as cryst


# ============================================================================
#                                CONSTANTS
# ============================================================================
SIGMA_MODES = ('fixed', 'scherrer', 'caglioti', 'separate', 'per_phase')
PROFILES = ('gaussian', 'pseudo_voigt')

LN2 = np.log(2.0)
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * LN2)     # FWHM = 2.3548 sigma
FWHM_MIN = 1e-4                               # deg, floor of the Caglioti FWHM

# Half-width (in FWHM) of the 2theta window in which a peak is evaluated.
# Gaussian: 3 FWHM = 7.1 sigma (tail < 1e-10 of the peak height).
# Pseudo-Voigt: 30 FWHM (Lorentzian tail < 3e-4 of the peak height).
PEAK_WINDOW_FWHM = {'gaussian': 3.0, 'pseudo_voigt': 30.0}

# Default starting values of the new GUI controls.
DEFAULT_ETA = 0.5
DEFAULT_CAGLIOTI_UVW = (0.001, -0.0005, 0.002)    # deg^2

# Bound of the constant background offset C, as a fraction of the largest
# absolute intensity of the (background-subtracted) pattern. The offset only
# absorbs a small residual of the background subtraction, so its bound has
# to scale with the intensity units of the data (counts or normalised).
BG_OFFSET_FRACTION = 0.1


# ============================================================================
#                           PARAMETER SET-UP
# ============================================================================
def initialise_parameters(phases, WL, twoThetaMin=5, twoThetaMax=40,
                             sigmaMode='scherrer', ampGuess=1, sigGuess=0.06,
                             sigmaBounds=2.0, ampBounds=3.0, maxShift=0.5,
                             profile='gaussian', etaGuess=DEFAULT_ETA,
                             cagliotiUVW=DEFAULT_CAGLIOTI_UVW,
                             bgBound=None):
    """
    lmfit Parameters of a Pawley fit of the phases in phases['phases_used'].

    ampGuess    : amplitude scale; the start amplitude of a reflection is
                  ampGuess * I_JCPDS / 100
    ampBounds   : upper bound multiplier for amplitudes; each amplitude is
                  bounded by [0, amp_init * ampBounds]
    sigGuess    : start sigma (deg) for every sigma mode except 'caglioti'
    sigmaBounds : upper bound multiplier for sigma (max = sigGuess*sigmaBounds)
    maxShift    : fractional bounds of the lattice parameters (0.0-1.0),
                  e.g. 0.5 = +-50 %; 0 fixes the lattice of every phase
    profile     : 'gaussian' or 'pseudo_voigt'
    etaGuess    : pseudo-Voigt mixing parameter eta (held fixed)
    cagliotiUVW : start values (U, V, W) in deg^2 for sigmaMode 'caglioti'
    bgBound     : bound of the constant background offset, BG in
                  [-bgBound, +bgBound] (intensity units of the data).
                  EXODUS_core.fit_pawley sets it to
                  BG_OFFSET_FRACTION * max|y|; None falls back to
                  BG_OFFSET_FRACTION * ampGuess.

    Per-phase, per-frame overrides set by the GUI (pressure-guess 'Edit'
    pop-up, see fix_parameters_toolbox) are read from the phases dict:
        phase_{X}_max_shift  : lattice-bounds fraction for this phase
                               (0 -> fix every lattice parameter)
        phase_{X}_amp_scale  : multiplier on ampGuess for this phase
        phase_{X}_amp_bounds : amplitude upper-bound multiplier
        phase_{X}_fix_params : lattice names to hold fixed
        phase_{X}_fix_lattice: True -> hold the whole lattice fixed
    """
    print('Initialising Parameters')
    if sigmaMode not in SIGMA_MODES:
        raise ValueError(f"Unknown sigma_mode: {sigmaMode}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")

    # maxShift = 0 means "fix the lattice of every phase". lmfit forbids
    # min == max for a parameter, so the bounds use +-50 % and the lattice
    # is frozen with vary=False below.
    fix_lattice_globally = (maxShift is not None and maxShift <= 0)
    global_bounds_shift = 0.5 if fix_lattice_globally else maxShift

    params = Parameters()
    # Constant background offset on top of the subtracted background. The
    # pattern is already background-subtracted, so it is kept close to 0.
    # The bound scales with the intensity of the data (it used to be a fixed
    # +-0.5, which is negligible for count data but larger than the peaks
    # of normalised data).
    if bgBound is None or not np.isfinite(bgBound) or bgBound <= 0:
        bgBound = BG_OFFSET_FRACTION * abs(float(ampGuess))
    if not np.isfinite(bgBound) or bgBound <= 0:
        bgBound = 1.0
    params.add('BG', value=0.0, min=-bgBound, max=bgBound)
    # Wavelength: taken from the PONI file and not refined.
    params.add('WL', value=WL, min=WL*0.95, max=WL*1.15)
    params['WL'].set(vary=False)

    # Global profile / broadening parameters
    if profile == 'pseudo_voigt':
        # Mixing parameter: fixed at the GUI value (not refined).
        eta0 = min(max(float(etaGuess), 0.0), 1.0)
        params.add('eta', value=eta0, min=0.0, max=1.0, vary=False)
    if sigmaMode == 'caglioti':
        U0, V0, W0 = (float(x) for x in cagliotiUVW)
        params.add('cag_U', value=max(U0, 0.0), min=0.0,
                   max=max(1.0, 10.0 * abs(U0)))
        params.add('cag_V', value=V0, min=-max(1.0, 10.0 * abs(V0)),
                   max=max(1.0, 10.0 * abs(V0)))
        params.add('cag_W', value=max(W0, 0.0), min=0.0,
                   max=max(1.0, 10.0 * abs(W0)))

    for phase in phases['phases_used']:
        crystal_system = cryst.normalise_crystal_system(
            phases[f'phase_{phase}_crystal_system'])
        unit_cell = phases[f'phase_{phase}_unit_cell']

        # ---- per-phase, per-frame overrides ----------------------------
        ms_override = phases.get(f'phase_{phase}_max_shift', None)
        if ms_override is not None:
            fix_all_this_phase = float(ms_override) <= 0
            maxShift = 0.5 if fix_all_this_phase else float(ms_override)
        else:
            fix_all_this_phase = fix_lattice_globally
            maxShift = global_bounds_shift
        amp_scale = phases.get(f'phase_{phase}_amp_scale', None)
        phase_ampGuess = (ampGuess * float(amp_scale) if amp_scale is not None
                          else ampGuess)
        ab_override = phases.get(f'phase_{phase}_amp_bounds', None)
        phase_ampBounds = (float(ab_override) if ab_override is not None
                           else ampBounds)

        # ---- lattice parameters -----------------------------------------
        # Free parameters vary within +-maxShift, tied ones follow their
        # master (expr), all others are held at their symmetry value.
        free = cryst.free_lattice_flags(crystal_system)
        tied = cryst.TIED_LATTICE.get(crystal_system, {})
        for i, nm in enumerate(cryst.LATTICE_NAMES):
            key = f'{nm}_{phase}'
            if nm in tied:
                params.add(key, expr=f'{tied[nm]}_{phase}')
                continue
            v = float(unit_cell[i])
            params.add(key, value=v, min=v * (1 - maxShift),
                       max=v * (1 + maxShift))
            params[key].set(vary=bool(free[i]))

        # Whole lattice fixed: per-frame 'Fix' tick (pressure-guess table)
        # or 0 in the 'Max shift' spin box. Values stay at phases[...].
        if phases.get(f'phase_{phase}_fix_lattice', False) or fix_all_this_phase:
            for nm in cryst.LATTICE_NAMES:
                key = f'{nm}_{phase}'
                if key in params and not params[key].expr:
                    params[key].set(vary=False)

        # Single lattice parameters fixed by a per-frame override
        for nm in (phases.get(f'phase_{phase}_fix_params', None) or ()):
            key = f'{nm}_{phase}'
            if key in params and not params[key].expr:
                params[key].set(vary=False)

        # ---- amplitudes: one per HKL row --------------------------------
        # Keyed by the row index in the FULL HKL array (amp_<phase>_<i+1>),
        # so a reflection keeps its parameter when the lattice moves it in
        # or out of the 2theta window. Reflections outside the window (or
        # with zero JCPDS intensity) start frozen, so they do not add
        # zero columns to the Jacobian.
        HKL_full = np.asarray(phases[f'phase_{phase}_HKL'], dtype=float)
        n_hkl = HKL_full.shape[0]
        tt_by_hkl_idx = cryst.d_spacing_to_two_theta(
            cryst.d_spacings(cryst.enforce_symmetry(unit_cell, crystal_system),
                             cryst.hkl_stencil(HKL_full)), WL)
        tt_by_hkl_idx = np.where(np.isfinite(tt_by_hkl_idx), tt_by_hkl_idx,
                                 np.inf)
        in_window = (tt_by_hkl_idx >= twoThetaMin) & (tt_by_hkl_idx <= twoThetaMax)

        for hkl_idx in range(n_hkl):
            amp_init = phase_ampGuess / 100.0 * HKL_full[hkl_idx, 3]
            upper = (amp_init * phase_ampBounds if amp_init > 0
                     else phase_ampGuess * phase_ampBounds)
            if upper <= 0:          # lmfit rejects min == max == 0
                upper = 1.0
            key = f'amp_{phase}_{hkl_idx+1}'
            params.add(key, value=amp_init, min=0.0, max=upper)
            if (not in_window[hkl_idx]) or (HKL_full[hkl_idx, 3] <= 0.0):
                params[key].set(vary=False)

        # Exactly overlapping reflections (same d for every cell of this
        # crystal system) cannot be separated in a Pawley fit [1]: refining
        # each amplitude makes the normal matrix singular. Only the first
        # amplitude of a group is refined; the others follow it in the
        # ratio of their JCPDS intensities.
        for group in cryst.degenerate_groups(HKL_full, crystal_system):
            first = group[0]
            I_first = HKL_full[first, 3]
            for j in group[1:]:
                key = f'amp_{phase}_{j+1}'
                if I_first > 0:
                    params[key].set(expr=f'amp_{phase}_{first+1} * '
                                         f'{HKL_full[j, 3] / I_first:.12g}')
                else:
                    params[key].set(value=0.0, vary=False)

        # ---- peak widths --------------------------------------------------
        setup_sigma_params(params, sigmaMode, phase, HKL_full, sigGuess,
                           sigmaBounds=sigmaBounds, inWindowMask=in_window)

    return params


def setup_sigma_params(params, sigmaMode, phase,
                       HKL, sigGuess, sigmaBounds=2.0,
                       inWindowMask=None):
    """
    Add the sigma (peak-width) parameters of one phase to `params`.

    sigmaMode     : one of SIGMA_MODES ('caglioti' parameters are global
                     and created in initialise_parameters)
    HKL            : full HKL array of the phase (its length is the index
                     space of the per-peak sigmas in 'separate' mode)
    sigGuess       : start sigma (deg)
    sigmaBounds    : upper bound multiplier, max = sigGuess * sigmaBounds
    inWindowMask : per-HKL bool array; in 'separate' mode sigmas of
                     reflections outside the fit window start frozen
    """
    sig_max = sigGuess * sigmaBounds
    n_hkl = np.asarray(HKL).shape[0]

    if sigmaMode in ('fixed', 'scherrer'):
        if 'sig_global' not in params:
            params.add('sig_global', value=sigGuess, min=0.0, max=sig_max)
    elif sigmaMode == 'per_phase':
        params.add(f'sig_{phase}', value=sigGuess, min=0.0, max=sig_max)
    elif sigmaMode == 'separate':
        for hkl_idx in range(n_hkl):
            params.add(f'sig_{phase}_{hkl_idx+1}',
                       value=sigGuess, min=0.0, max=sig_max)
            if inWindowMask is not None and not inWindowMask[hkl_idx]:
                params[f'sig_{phase}_{hkl_idx+1}'].set(vary=False)
    elif sigmaMode == 'caglioti':
        pass
    else:
        raise ValueError(f"Unknown sigma_mode: {sigmaMode}")


# ============================================================================
#                        PEAK WIDTHS AND PROFILES
# ============================================================================
def peak_sigmas(pv, sigmaMode, phase, cens, hklIndices):
    """Gaussian sigma (deg) of each reflection.

    pv          : dict of parameter values (Parameters.valuesdict())
    cens        : peak centres (deg 2theta)
    hklIndices  : row indices of the reflections in the full HKL array
    """
    cens = np.asarray(cens, dtype=float)
    n = cens.size
    if sigmaMode == 'fixed':
        return np.full(n, pv['sig_global'])
    if sigmaMode == 'per_phase':
        return np.full(n, pv[f'sig_{phase}'])
    if sigmaMode == 'scherrer':
        return pv['sig_global'] / np.cos(cens / 2 * cryst.DEG_TO_RAD)
    if sigmaMode == 'caglioti':
        t = np.tan(cens / 2 * cryst.DEG_TO_RAD)
        fwhm2 = pv['cag_U'] * t * t + pv['cag_V'] * t + pv['cag_W']
        return np.sqrt(np.maximum(fwhm2, FWHM_MIN ** 2)) / FWHM_PER_SIGMA
    if sigmaMode == 'separate':
        # A reflection that only drifted into the window during the fit
        # reuses the sigma with the highest index of this phase.
        keys = [k for k in pv if k.startswith(f'sig_{phase}_')]
        if not keys:
            raise KeyError(f"No sigma parameters found for phase {phase}")
        last = max(keys, key=lambda k: int(k.rsplit('_', 1)[-1]))
        return np.fromiter((pv.get(f'sig_{phase}_{int(i)+1}', pv[last])
                            for i in hklIndices), dtype=np.float64, count=n)
    raise ValueError(f"Unknown sigma_mode: {sigmaMode}")



def peak_shape(u, eta, profile):
    """Normalised profile (height 1) at u = (2theta - centre) / HWHM."""
    g = np.exp(-LN2 * u * u)
    if profile == 'gaussian':
        return g
    return eta / (1.0 + u * u) + (1.0 - eta) * g


def add_peaks(x, yFit, cens, amps, sigmas, eta=0.0, profile='gaussian'):
    """Add the peaks (centres `cens`, heights `amps`, Gaussian `sigmas`) to
    `yFit` in place, evaluating each peak only on the points of the
    ascending grid `x` within +-PEAK_WINDOW_FWHM[profile] FWHM."""
    n = len(cens)
    if n == 0:
        return
    hwhm = 0.5 * FWHM_PER_SIGMA * np.asarray(sigmas, dtype=float)
    half = 2.0 * PEAK_WINDOW_FWHM[profile] * hwhm
    lo = np.searchsorted(x, cens - half, side='left')
    hi = np.searchsorted(x, cens + half, side='right')
    width = int(np.max(hi - lo))
    if width <= 0:
        return
    idx = lo[:, None] + np.arange(width)[None, :]
    valid = idx < hi[:, None]
    idx = np.where(valid, idx, 0)
    u = (x[idx] - cens[:, None]) / hwhm[:, None]
    vals = np.asarray(amps, dtype=float)[:, None] * peak_shape(u, eta, profile)
    yFit += np.bincount(idx[valid], weights=vals[valid], minlength=x.size)


def integrated_intensity(amp, sigma, eta=0.0):
    """Area of a peak of height `amp` (pseudo-Voigt with mixing eta;
    eta = 0 is the Gaussian): amp [eta pi HWHM + (1 - eta) sigma sqrt(2 pi)]."""
    hwhm = 0.5 * FWHM_PER_SIGMA * sigma
    return amp * (eta * np.pi * hwhm + (1.0 - eta) * sigma * np.sqrt(2.0 * np.pi))


# ============================================================================
#                              RESIDUAL
# ============================================================================
def phase_reflections(pv, phase, phases, dMin, dMax, stencil=None):
    """(centres, HKL row indices) of the reflections of one phase inside
    the d-range, for the current parameter values `pv`."""
    if stencil is None:
        stencil = cryst.hkl_stencil(phases[f'phase_{phase}_HKL'])
    cell = [pv[f'{nm}_{phase}'] for nm in cryst.LATTICE_NAMES]
    dk = cryst.d_spacings(cell, stencil)
    mask = np.isfinite(dk) & (dk > dMin) & (dk < dMax)
    idx = np.flatnonzero(mask)
    return cryst.d_spacing_to_two_theta(dk[mask], pv['WL']), idx


def pawley_residual(params, x, y, phases, twoThetaMin=0, twoThetaMax=30,
                 sigmaMode='scherrer', profile='gaussian', stencils=None):
    """
    Residual (model - data) of the Pawley fit, vectorised over reflections.

    x must be ascending. `stencils` ({phase: cryst.hkl_stencil(HKL)}) can be
    pre-computed once per fit; otherwise they are built on every call.
    """
    pv = params.valuesdict()
    y_fit = np.full_like(x, pv['BG'], dtype=float)
    d_min = cryst.two_theta_to_d_spacing(twoThetaMax, pv['WL'])
    d_max = cryst.two_theta_to_d_spacing(twoThetaMin, pv['WL'])
    eta = pv.get('eta', 0.0)

    for phase in phases['phases_used']:
        st = stencils.get(phase) if stencils else None
        cens, idx = phase_reflections(pv, phase, phases, d_min, d_max, st)
        if cens.size == 0:
            continue
        amps = np.fromiter((pv[f'amp_{phase}_{int(i)+1}'] for i in idx),
                           dtype=np.float64, count=idx.size)
        sigs = peak_sigmas(pv, sigmaMode, phase, cens, idx)
        add_peaks(x, y_fit, cens, amps, sigs, eta, profile)

    return y_fit - y


# ============================================================================
#                        FIT RESULTS FOR THE GUI
# ============================================================================
def unit_cell_fit(result, phase):
    '''Fitted unit cell [a, b, c, alpha, beta, gamma] of a phase.'''
    return np.array([result.params[f'{nm}_{phase}'].value
                     for nm in cryst.LATTICE_NAMES])


def create_ticks(result, phases, twoThetaMin, twoThetaMax, WL):
    '''
    2theta of the reflections of every used phase for the fitted cells,
    as a list (one list per phase).
    '''
    tickArray = []
    for phase in phases['phases_used']:
        _, twoTheta_reflection_list = cryst.reflection_list(
            unit_cell_fit(result, phase),
            phases[f'phase_{phase}_HKL'],
            WL,
            twoThetaMin=twoThetaMin,
            twoThetaMax=twoThetaMax,
            crystalSystem=phases[f'phase_{phase}_crystal_system'])
        tickArray.append(twoTheta_reflection_list[:, 0].tolist())
    return tickArray


def create_fit_components(twoTheta, phases, params, twoThetaMin, twoThetaMax,
                         sigmaMode='scherrer', profile='gaussian'):
    '''
    Fitted curve and its per-reflection components.

    Returns (bestFit, components): bestFit is the model exactly as it was
    refined (windowed peak evaluation, incl. the BG offset); components maps
    'comp_<phase>_<HKL row+1>' to that reflection evaluated over the full
    2theta grid, plus 'BG'.
    '''
    x = np.asarray(twoTheta, dtype=float)
    bestFit = pawley_residual(params, x, np.zeros_like(x), phases,
                           twoThetaMin, twoThetaMax, sigmaMode, profile)
    pv = params.valuesdict()
    d_min = cryst.two_theta_to_d_spacing(twoThetaMax, pv['WL'])
    d_max = cryst.two_theta_to_d_spacing(twoThetaMin, pv['WL'])
    eta = pv.get('eta', 0.0)
    components = {'BG': pv['BG']}
    for phase in phases['phases_used']:
        cens, idx = phase_reflections(pv, phase, phases, d_min, d_max)
        sigs = peak_sigmas(pv, sigmaMode, phase, cens, idx)
        for cen, i, sig in zip(cens, idx, sigs):
            amp = pv[f'amp_{phase}_{int(i)+1}']
            u = (x - cen) / (0.5 * FWHM_PER_SIGMA * sig)
            components[f'comp_{phase}_{int(i)+1}'] = amp * peak_shape(u, eta, profile)
    return bestFit, components
