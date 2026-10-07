# results_toolbox.py for EXODUS v0.1.4
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
# Output files of a fitted pattern (no Qt). The same writers are used by
# "Save" (single fit) and "Save All" (sequential fit):
#
#   <stem>_fit<ext>        profile: 2theta, intensity, BG, fit, residual and
#                          one tick-mark column per phase
#   <stem>_lattParam<ext>  one row per phase: P, dP, V, dV, cell, cell errors,
#                          fit statistics (columns LATT_COLUMNS)
#   <stem>_fitParam<ext>   per-reflection table (hkl, 2theta, d, Q, amplitude,
#                          width, integrated intensity, all with errors)
#
# V, dV, P and dP are the values stored with the fit (EXODUS_core.fit_pawley),
# so all files and the analysis pop-ups agree.
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import numpy as np

import crystallography_toolbox as cryst
import eos_toolbox as eos
import pawley_toolbox as pawley
import statistics_toolbox as stats


# ============================================================================
#                                CONSTANTS
# ============================================================================
EXODUS_VERSION = 'v0.1.4'

LATT_COLUMNS = ['Frame', 'Pressure_GPa', 'dP_GPa', 'V_A3', 'dV_A3',
                'a', 'b', 'c', 'alpha', 'beta', 'gamma',
                'da', 'db', 'dc', 'dalpha', 'dbeta', 'dgamma',
                'Rp', 'Rwp', 'chi2', 'chi2_red']

PROFILE_NAMES = {'gaussian': 'Gaussian', 'pseudo_voigt': 'pseudo-Voigt'}


# ============================================================================
#                                HELPERS
# ============================================================================
def fmt(v, form='.6f'):
    """Number -> string; 'nan' for missing / non-numeric values."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 'nan'
    return format(v, form) if np.isfinite(v) else 'nan'


def _value_and_error(params, name):
    """(value, stderr) of an lmfit parameter; NaN where unavailable."""
    if params is None or name not in params:
        return np.nan, np.nan
    p = params[name]
    err = float(p.stderr) if p.stderr is not None else (0.0 if not p.vary else np.nan)
    return float(p.value), err


def stats_header_lines(results, frame, sourceName='n/a', WL=np.nan,
                       includeDefinitions=True):
    """'# ...' lines describing the fit and its statistics.
    includeDefinitions=False drops the R/chi2 definition lines (used by
    the slim *_fitParam file)."""
    f = fmt
    WL = results.get('_WL', WL)
    profile = results.get('_profile', 'gaussian')
    params = results.get('_lmfit_params')
    lines = [
        f'# EXODUS {EXODUS_VERSION}',
        f'# Frame = {frame}   Source file = {sourceName}',
        f'# Wavelength_A = {f(WL)}   2theta window = '
        f'{f(results.get("_twoThetaMin"), ".4f")} - '
        f'{f(results.get("_twoThetaMax"), ".4f")} deg',
        f'# Profile = {PROFILE_NAMES.get(profile, profile)}   Sigma mode = '
        f'{results.get("_sigma_mode", "n/a")}',
    ]
    if profile == 'pseudo_voigt':
        v, e = _value_and_error(params, 'eta')
        if params is not None and 'eta' in params and not params['eta'].vary:
            lines.append(f'# eta = {f(v, ".5f")} (fixed)')
        else:
            lines.append(f'# eta = {f(v, ".5f")} +- {f(e, ".5f")}')
    if results.get('_sigma_mode') == 'caglioti':
        parts = []
        for nm in ('U', 'V', 'W'):
            v, e = _value_and_error(params, f'cag_{nm}')
            parts.append(f'{nm} = {f(v, ".6g")} +- {f(e, ".3g")}')
        lines.append('# Caglioti FWHM^2 = U tan^2(theta) + V tan(theta) + W '
                     '(deg^2):  ' + '   '.join(parts))
    if results.get('_errors_fallback'):
        lines.append('# WARNING: covariance matrix not available (a parameter '
                     'sits on a bound); lattice errors = sqrt(reduced chi2)')
    lines += [
        '# Fit = Pawley, unweighted least squares on background-subtracted data',
        f'# Statistics weighting = {results.get("weighting", "n/a")}',
        f'# Rp = {f(results.get("Rp"))}   Rwp = {f(results.get("Rwp"))}'
        f'   Rexp = {f(results.get("Rexp"))}',
        f'# Rp_bgcorr = {f(results.get("Rp_bgcorr"))}   Rwp_bgcorr = '
        f'{f(results.get("Rwp_bgcorr"))}',
        f'# chi2 = {f(results.get("chi2"), ".6g")}   chi2_red = '
        f'{f(results.get("chi2_red"), ".6g")}   N_points = '
        f'{results.get("N_points", "n/a")}   N_params = '
        f'{results.get("N_params", "n/a")}',
    ]
    if includeDefinitions:
        lines += [
            '# Definitions: r_i = I_obs_i - I_calc_i; w_i = 1/I_raw_i (Poisson,'
            ' raw = before BG subtraction, w_i = 0 for I_raw_i <= 0)',
            '#   Rp = sum|r| / sum I_raw;  Rwp = sqrt(sum w r^2 / sum w I_raw^2);'
            '  chi2 = sum w r^2;  chi2_red = chi2/(N-P);'
            '  *_bgcorr: denominators use I_obs - BG',
        ]
    return lines


def phase_cell_summary(results, phase, phases):
    """Fitted cell, its errors, the cell covariance, V +- dV and P +- dP
    (BM3) of a phase; None if the phase was not fitted in `results`."""
    UC = results.get(f'phase_{phase}_unit_cell_fit', results.get(f'phase_{phase}'))
    if UC is None:
        return None
    UC = [float(x) for x in UC]
    err = results.get(f'phase_{phase}_unit_cell_error',
                      results.get(f'phase_{phase}_error'))
    err = (np.full(6, np.nan) if err is None
           else np.asarray(err, dtype=float))
    err = [float(x) for x in err]
    params = results.get('_lmfit_params')
    cov = stats.cell_covariance(params, phase) if params is not None else None
    comp = phases[f'phase_{phase}_compression_constants']
    V0, K0, K0P = float(comp[0]), float(comp[1]), float(comp[2])
    V = results.get(f'phase_{phase}_V_fit')
    dV = results.get(f'phase_{phase}_V_error')
    if V is None or dV is None:     # results without stored volume
        V, dV = stats.cell_volume_and_error(
            UC, cov, uncErrors=err,
            crystalSystem=phases.get(f'phase_{phase}_crystal_system'))
    P, dP = eos.bm3_pressure_and_error(V, dV, V0, K0, K0P)
    return {'UC': UC, 'err': err, 'cov': cov, 'V': V, 'dV': dV,
            'P': P, 'dP': dP, 'V0': V0, 'K0': K0, 'K0P': K0P}


def latt_param_rows(frame, results, phases):
    """Rows (LATT_COLUMNS) of all fitted phases of one frame."""
    rows = []
    for phase_index in phases.get('phases_used', []):
        cs = phase_cell_summary(results, phase_index, phases)
        if cs is None:
            continue
        rows.append([float(frame), cs['P'], cs['dP'], cs['V'], cs['dV']]
                    + cs['UC'] + cs['err']
                    + [results.get(k, np.nan) for k in
                       ('Rp', 'Rwp', 'chi2', 'chi2_red')])
    return rows


def peak_width_and_area(params, sigmaMode, profile, phase, hklIdx, cen):
    """
    Width and integrated intensity of one reflection with 1-sigma errors.

    Returns (amp, d_amp, sigma, d_sigma, I_int, d_I) in deg / intensity
    units. Errors are propagated linearly from the covariance of every
    parameter involved (amplitude, width parameters, eta) [GUM, JCGM 100];
    the peak position is treated as exact.
    """
    nan = np.nan
    amp_key = f'amp_{phase}_{hklIdx + 1}'
    if amp_key not in params:
        return (nan,) * 6
    if sigmaMode in ('fixed', 'scherrer'):
        width_keys = ['sig_global']
    elif sigmaMode == 'per_phase':
        width_keys = [f'sig_{phase}']
    elif sigmaMode == 'caglioti':
        width_keys = ['cag_U', 'cag_V', 'cag_W']
    elif sigmaMode == 'separate':
        key = f'sig_{phase}_{hklIdx + 1}'
        if key not in params:
            keys = [k for k in params if k.startswith(f'sig_{phase}_')]
            if not keys:
                return (nan,) * 6
            key = max(keys, key=lambda k: int(k.rsplit('_', 1)[-1]))
        width_keys = [key]
    else:
        return (nan,) * 6
    names = [amp_key] + width_keys + (['eta'] if profile == 'pseudo_voigt'
                                      and 'eta' in params else [])
    if any(n not in params for n in names):
        return (nan,) * 6
    x0 = np.array([float(params[n].value) for n in names])

    def quantities(x):
        pv = dict(zip(names, x))
        if sigmaMode == 'separate':
            pv[f'sig_{phase}_{hklIdx + 1}'] = pv[width_keys[0]]
        sigma = float(pawley.peak_sigmas(pv, sigmaMode, phase, [cen], [hklIdx])[0])
        eta = pv.get('eta', 0.0)
        return np.array([sigma, pawley.integrated_intensity(pv[amp_key], sigma, eta)])

    sigma, area = quantities(x0)
    # covariance of the involved parameters (fixed ones contribute 0)
    n = len(names)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi, pj = params[names[i]], params[names[j]]
            if not pi.vary or not pj.vary:
                continue
            c = stats._param_cov(params, names[i], names[j])
            cov[i, j] = c
    jac = np.zeros((2, n))
    for i in range(n):
        h = 1e-6 * max(abs(x0[i]), 1e-6)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        jac[:, i] = (quantities(xp) - quantities(xm)) / (2 * h)
    var = np.diag(jac @ cov @ jac.T) if np.all(np.isfinite(cov)) else (nan, nan)
    d_sigma = float(np.sqrt(var[0])) if np.isfinite(var[0]) and var[0] >= 0 else nan
    d_area = float(np.sqrt(var[1])) if np.isfinite(var[1]) and var[1] >= 0 else nan
    amp, d_amp = _value_and_error(params, amp_key)
    return amp, d_amp, sigma, d_sigma, area, d_area


def fit_curve_header_lines(results, frame, sourceName='n/a', WL=np.nan):
    """'# ...' header of the *_fit file: run information and fit
    statistics only (the eta / Caglioti values and the definitions are in
    the *_fitParam file), closed by two empty '#' lines."""
    f = fmt
    WL = results.get('_WL', WL)
    profile = results.get('_profile', 'gaussian')
    return [
        f'# EXODUS {EXODUS_VERSION}',
        f'# Frame = {frame}   Source file = {sourceName}',
        f'# Wavelength_A = {f(WL)}   2theta window = '
        f'{f(results.get("_twoThetaMin"), ".4f")} - '
        f'{f(results.get("_twoThetaMax"), ".4f")} deg',
        f'# Profile = {PROFILE_NAMES.get(profile, profile)}   Sigma mode = '
        f'{results.get("_sigma_mode", "n/a")}',
        '# Fit = unweighted least squares on background-subtracted data',
        f'# Statistics weighting = {results.get("weighting", "n/a")}',
        f'# Rp = {f(results.get("Rp"))}   Rwp = {f(results.get("Rwp"))}'
        f'   Rexp = {f(results.get("Rexp"))}',
        f'# Rp_bgcorr = {f(results.get("Rp_bgcorr"))}   Rwp_bgcorr = '
        f'{f(results.get("Rwp_bgcorr"))}',
        f'# chi2 = {f(results.get("chi2"), ".6g")}   chi2_red = '
        f'{f(results.get("chi2_red"), ".6g")}   N_points = '
        f'{results.get("N_points", "n/a")}   N_params = '
        f'{results.get("N_params", "n/a")}',
        '#',
        '#',
    ]


# ============================================================================
#                                WRITERS
# ============================================================================
def write_fit_curve_file(outPath, frame, results, phases, tt, raw, bgsub, fit,
                         delimiter, sourceName='n/a', WL=np.nan):
    """Profile file: TwoTheta, Intensity, Intensity-BG, BG, Fit, Fit+BG,
    Residual, then one tick-mark column per used phase."""
    tt = np.asarray(tt, dtype=float)
    bgsub = np.asarray(bgsub, dtype=float)
    fit = np.asarray(fit, dtype=float)
    n = min(len(tt), len(bgsub), len(fit))
    if raw is not None:
        raw = np.asarray(raw, dtype=float)
        n = min(n, len(raw))
        raw = raw[:n]
        BG = raw - bgsub[:n]
    else:
        raw = np.full(n, np.nan)
        BG = np.full(n, np.nan)
    tt, bgsub, fit = tt[:n], bgsub[:n], fit[:n]
    resid = bgsub - fit

    tick_cols, tick_hdr = [], []
    for phase in (phases or {}).get('phases_used', []):
        col = np.full(n, np.nan)
        tick = results.get(f'phase_{phase}_peakPosition_fit')
        if tick is not None:
            tick = np.asarray(tick, dtype=float).ravel()
            m = min(len(tick), n)
            col[:m] = tick[:m]
        tick_cols.append(col)
        tick_hdr.append(f'Phase{phase}_Tick')

    data = np.column_stack([tt, raw, bgsub, BG, fit, fit + BG, resid] + tick_cols)
    cols = ['TwoTheta', 'Intensity', 'Intensity-BG', 'BG', 'Fit',
            'Fit+BG', 'Residual'] + tick_hdr
    header = '\n'.join(fit_curve_header_lines(results, frame, sourceName, WL)
                       + [delimiter.join(cols)])
    np.savetxt(outPath, data, delimiter=delimiter, header=header,
               comments='', fmt='%.6f')


def write_latt_param_file(outPath, frame, results, phases, delimiter):
    """One row per phase, columns LATT_COLUMNS. Same layout as the
    aggregated *_sequential_results.csv written by 'Save All'."""
    if results is None or phases is None:
        print("  Skipping lattice-parameter save: no fit results / JCPDS.")
        return
    rows = latt_param_rows(frame, results, phases)
    if not rows:
        print("  Skipping lattice-parameter save: no fitted phases.")
        return
    np.savetxt(outPath, np.asarray(rows, dtype=float),
               delimiter=delimiter, header=delimiter.join(LATT_COLUMNS),
               comments='', fmt='%.6g')
    print(f"Saved lattice parameters to: {outPath}")


def write_fit_param_file(outPath, frame, results, phases, WL, twoThetaMin,
                         twoThetaMax, delimiter, sourceName='n/a'):
    """Per-reflection table, one block per phase.

    Columns
      h k l
      2theta_0, d_0(A), Q_0(A^-1)  : from the JCPDS reference cell
      2theta_fit, d_fit, Q_fit     : from the fitted cell; d_* = 1-sigma
                                     errors propagated from the cell
                                     covariance (WL treated as exact)
      Amp, d_Amp                   : refined peak height
      sig_deg, dsig_deg            : Gaussian sigma at this peak (deg)
      FWHM_deg, dFWHM_deg          : 2 sqrt(2 ln 2) sigma
      I_int, dI                    : peak area, Gaussian: A sigma sqrt(2 pi),
                                     pseudo-Voigt: A [eta pi HWHM +
                                     (1 - eta) sigma sqrt(2 pi)]; error incl.
                                     the A-width correlations (eta is fixed)
    """
    if results is None or phases is None:
        print("  Skipping fitParam save: no fit results / JCPDS.")
        return
    params = results.get('_lmfit_params')
    if params is None:
        print("  Skipping fitParam save: per-peak fit info not cached "
              "(re-run the fit to enable).")
        return
    sigmaMode = results.get('_sigma_mode', 'scherrer')
    profile = results.get('_profile', 'gaussian')
    f = fmt

    col_headers = ['h', 'k', 'l',
                   '2theta_0', '2theta_fit', 'd_2theta_fit',
                   'd_0(A)', 'd_fit(A)', 'd_d(A)',
                   'Q_0(A^-1)', 'Q_fit(A^-1)', 'd_Q(A^-1)',
                   'Amp', 'd_Amp', 'sig_deg', 'dsig_deg',
                   'FWHM_deg', 'dFWHM_deg', 'I_int', 'dI']

    with open(outPath, 'w') as fh:
        fh.write('# Per-reflection fit parameters\n')
        for ln in stats_header_lines(results, frame, sourceName, WL,
                                     includeDefinitions=False):
            fh.write(ln + '\n')
        fh.write('#\n')

        for phase_idx in phases.get('phases_used', []):
            cs = phase_cell_summary(results, phase_idx, phases)
            if cs is None:
                continue
            UC, err, cov = cs['UC'], cs['err'], cs['cov']
            phase_name = phases.get(f'phase_{phase_idx}_name', f'phase_{phase_idx}')
            crystal_system = phases.get(f'phase_{phase_idx}_crystal_system',
                                        'TRICLINIC')

            fh.write('#\n')
            fh.write(f'# Phase {phase_idx}: {phase_name} ({crystal_system})\n')
            fh.write(
                f'# a={f(UC[0], ".5f")}({f(err[0], ".5f")})  '
                f'b={f(UC[1], ".5f")}({f(err[1], ".5f")})  '
                f'c={f(UC[2], ".5f")}({f(err[2], ".5f")})  '
                f'alpha={f(UC[3], ".4f")}({f(err[3], ".4f")})  '
                f'beta={f(UC[4], ".4f")}({f(err[4], ".4f")})  '
                f'gamma={f(UC[5], ".4f")}({f(err[5], ".4f")})\n')
            fh.write(f'# V_A3 = {f(cs["V"], ".4f")}   dV_A3 = '
                     f'{f(cs["dV"], ".4f")}   P_GPa = {f(cs["P"], ".4f")}'
                     f'   dP_GPa = {f(cs["dP"], ".4f")}'
                     '   (dP from dV only; EoS parameters taken as exact)\n')
            fh.write(f'# EoS = BM3, T = 300 K, '
                     f'V0_uc_Ang3 = {f(cs["V0"], ".4f")}   '
                     f'K0_GPa = {f(cs["K0"], ".3f")}   '
                     f'K0p = {f(cs["K0P"], ".4f")}\n')
            fh.write('# ' + delimiter.join(col_headers) + '\n')

            try:
                _, peak_rows = cryst.reflection_list(
                    list(UC), phases[f'phase_{phase_idx}_HKL'], WL,
                    twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                    crystalSystem=crystal_system)
            except Exception as e:
                print(f"  fitParam: could not compute reflection list "
                      f"for phase {phase_idx}: {e}")
                continue
            if peak_rows is None or len(peak_rows) == 0:
                continue

            UC0 = phases.get(f'phase_{phase_idx}_unit_cell_0', None)
            HKL_full = np.asarray(phases[f'phase_{phase_idx}_HKL'], dtype=float)
            hkl_to_idx = {(int(round(h)), int(round(k)), int(round(l))): i
                          for i, (h, k, l) in enumerate(HKL_full[:, :3])}

            for row in np.asarray(peak_rows, dtype=float):
                h, k, l = (int(round(row[1])), int(round(row[2])),
                           int(round(row[3])))
                hkl_idx = hkl_to_idx.get((h, k, l))
                if hkl_idx is None:
                    continue
                rv = stats.reflection_values_and_errors(UC, cov, h, k, l, WL)
                cen = rv['tt'] if np.isfinite(rv['tt']) else float(row[0])

                if UC0 is not None:
                    d0 = cryst.d_spacing_general(list(UC0), h, k, l)
                    s0 = WL / (2 * d0) if np.isfinite(d0) and d0 > 0 else np.nan
                    tt0 = (float(np.degrees(2 * np.arcsin(s0)))
                           if np.isfinite(s0) and s0 < 1 else np.nan)
                    Q0 = 2 * np.pi / d0 if np.isfinite(d0) and d0 > 0 else np.nan
                else:
                    d0 = tt0 = Q0 = np.nan

                A, sA, s, ss, I_int, sI = peak_width_and_area(
                    params, sigmaMode, profile, phase_idx, hkl_idx, cen)
                FWHM = pawley.FWHM_PER_SIGMA * s
                sFWHM = pawley.FWHM_PER_SIGMA * ss

                fields = ([f'{h:d}', f'{k:d}', f'{l:d}']
                          + [f(x) for x in (tt0, rv['tt'], rv['sig_tt'],
                                            d0, rv['d'], rv['sig_d'],
                                            Q0, rv['Q'], rv['sig_Q'],
                                            A, sA, s, ss, FWHM, sFWHM,
                                            I_int, sI)])
                fh.write(delimiter.join(fields) + '\n')

    print(f"Saved reflection table to: {outPath}")
