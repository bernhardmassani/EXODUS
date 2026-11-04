# statistics_toolbox.py for EXODUS v0.1.4
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
# Fit statistics and error propagation (no Qt):
#
#   * profile agreement indices Rp, Rwp, Rexp, chi^2 of a Pawley fit [1],
#   * 1-sigma standard errors of the lattice parameters from lmfit, with the
#     sqrt(reduced chi^2) fallback when no covariance matrix is available,
#   * first-order (linear) propagation of the cell covariance to the
#     volume, d-spacings, 2theta and Q of the reflections [2].
#
# The unit-cell volume and its error are computed ONCE per fit by
# cell_volume_and_error() and stored with the fit results, so the output
# files, 'Plot Data' and 'EoS Fit' all use the same sigma(V).
#
# References
# ----------
# [1] B. H. Toby, "R factors in Rietveld analysis: How good is good
#     enough?", Powder Diffr. 21, 67-70 (2006).
# [2] JCGM 100:2008, Evaluation of measurement data - Guide to the
#     expression of uncertainty in measurement (GUM), section 5.2.
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import numpy as np

import crystallography_toolbox as cryst


# ============================================================================
#                         PROFILE AGREEMENT INDICES
# ============================================================================
def profile_statistics(yNetObs, yNetCalc, yRaw=None, nParams=0):
    """
    Profile agreement indices for one fitted pattern.

    EXODUS fits background-subtracted data, so the model compared with the
    data is y_calc_total = yNetCalc + BG and the residual is the same
    whether it is formed from the net or the raw intensities:
        r_i = y_net_obs_i - y_net_calc_i = y_raw_i - (y_net_calc_i + BG_i)

    Definitions (sums over all points of the fitted 2theta window):

      Rp          = sum|r_i| / sum y_raw_i                (conventional)
      Rp_bgcorr   = sum|r_i| / sum|y_net_obs_i|           (background-corrected)
      Rwp         = sqrt( sum w_i r_i^2 / sum w_i y_raw_i^2 )        (conventional)
      Rwp_bgcorr  = sqrt( sum w_i r_i^2 / sum w_i y_net_obs_i^2 )    (bg-corrected)
      Rexp        = sqrt( (N - P) / sum w_i y_raw_i^2 )
      chi2        = sum w_i r_i^2
      chi2_red    = chi2 / (N - P)          (= (Rwp/Rexp)^2)

    Weights: Poisson statistics of the RAW (not background-subtracted)
    intensity, sigma_i^2 = y_raw_i, i.e. w_i = 1/y_raw_i for y_raw_i > 0 and
    w_i = 0 otherwise (same convention as GSAS-II for x-y data without
    uncertainties). N counts only points with w_i > 0. Note that the
    variance of the net intensity is Var(yRaw - BG) ~ Var(yRaw) because
    the background is a smooth function fitted through many points.

    Caveat: integrated/normalised detector data are not raw counts, so the
    weights are only correct up to an unknown scale factor. Rp, Rwp and the
    refined parameters are independent of that factor; chi2 and chi2_red are
    not, so chi2_red = 1 is only expected for true counts.

    If yRaw is None, the weighted quantities (and the conventional Rp) are
    returned as NaN and only Rp_bgcorr is computed.
    """
    y_obs = np.asarray(yNetObs, dtype=float)
    y_calc = np.asarray(yNetCalc, dtype=float)
    n = min(len(y_obs), len(y_calc))
    y_obs, y_calc = y_obs[:n], y_calc[:n]
    r = y_obs - y_calc
    nan = float('nan')
    out = {'Rp': nan, 'Rp_bgcorr': nan, 'Rwp': nan, 'Rwp_bgcorr': nan,
           'Rexp': nan, 'chi2': nan, 'chi2_red': nan,
           'N_points': int(n), 'N_params': int(nParams),
           'weighting': 'w=1/I_raw (Poisson)'}

    den = np.sum(np.abs(y_obs))
    if den > 0:
        out['Rp_bgcorr'] = float(np.sum(np.abs(r)) / den)

    if yRaw is None:
        out['weighting'] = 'none (raw intensities unavailable)'
        return out

    yRaw = np.asarray(yRaw, dtype=float)[:n]
    if len(yRaw) != n:
        out['weighting'] = 'none (raw/net length mismatch)'
        return out

    den_raw = np.sum(yRaw)
    if den_raw > 0:
        out['Rp'] = float(np.sum(np.abs(r)) / den_raw)

    pos = yRaw > 0
    w = np.zeros_like(yRaw)
    w[pos] = 1.0 / yRaw[pos]
    n_w = int(np.count_nonzero(pos))
    dof = n_w - int(nParams)
    out['N_points'] = n_w

    chi2 = float(np.sum(w * r * r))
    sw_raw = float(np.sum(w * yRaw * yRaw))     # = sum(y_raw) for y_raw>0
    sw_net = float(np.sum(w * y_obs * y_obs))
    out['chi2'] = chi2
    if sw_raw > 0:
        out['Rwp'] = float(np.sqrt(chi2 / sw_raw))
        if dof > 0:
            out['Rexp'] = float(np.sqrt(dof / sw_raw))
    if sw_net > 0:
        out['Rwp_bgcorr'] = float(np.sqrt(chi2 / sw_net))
    if dof > 0:
        out['chi2_red'] = chi2 / dof
    return out


# Keys written by profile_statistics, in output-file order.
PROFILE_STAT_KEYS = ('Rp', 'Rwp', 'Rp_bgcorr', 'Rwp_bgcorr', 'Rexp',
                     'chi2', 'chi2_red', 'N_points', 'N_params')


def fit_statistics(result, twoTheta, fullFitReport=False, verbose=False):
    """
    Print a summary of the fit and return the global error estimate
    sqrt(reduced chi-square), used as fallback lattice-parameter error
    when no covariance matrix is available (see unit_cell_errors).

    Per-parameter standard errors (when available) live on
    `result.params[name].stderr`; use `unit_cell_errors(result, phase)` to
    pull them out as a 6-element ndarray.
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
            stderr_str = (f"±{par.stderr:.4g}" if par.stderr is not None
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


# ============================================================================
#                          LATTICE-PARAMETER ERRORS
# ============================================================================
CELL_NAMES = ('a', 'b', 'c', 'alp', 'bet', 'gam')


def unit_cell_errors(result, phase, fallback=None):
    """
    Return a 6-element ndarray [da, db, dc, dalpha, dbeta, dgamma] of
    1-sigma standard errors for the unit-cell parameters of `phase`.

    Constraint expressions (e.g. `b = a` in cubic) are followed: a slaved
    parameter inherits its master's stderr. Fixed parameters get error 0.
    A varying parameter that has no stderr (no covariance matrix, e.g.
    because a parameter sits on a bound) gets `fallback` if provided, else
    NaN.

    Returns (errors, usedFallback); usedFallback is True if at least one
    error had to be replaced by `fallback`.
    """
    names = [f'a_{phase}', f'b_{phase}', f'c_{phase}',
             f'alp_{phase}', f'bet_{phase}', f'gam_{phase}']
    errs = np.zeros(6)
    usedFallback = False
    for i, name in enumerate(names):
        par = result.params[name]
        if par.expr is not None:
            # For EXODUS the expressions are simple identifiers like 'a_0';
            # walk one hop. If it ever gets compound (a*b/c), .stderr on
            # the slave comes from lmfit's expr propagation when present.
            if par.stderr is not None:
                stderr = par.stderr
            else:
                master = result.params.get(par.expr.strip())
                stderr = master.stderr if master is not None else None
        elif not par.vary:
            stderr = 0.0
        else:
            stderr = par.stderr
        if stderr is None:
            usedFallback = usedFallback or fallback is not None
            stderr = fallback if fallback is not None else np.nan
        errs[i] = stderr
    return errs, usedFallback


def _param_cov(params, n1, n2):
    """Covariance of two lmfit parameters from their stderr and correl."""
    p1, p2 = params[n1], params[n2]
    if p1.stderr is None or p2.stderr is None:
        return float('nan')
    if n1 == n2:
        return float(p1.stderr) ** 2
    rho = 0.0
    if getattr(p1, 'correl', None) and n2 in p1.correl:
        rho = float(p1.correl[n2])
    elif getattr(p2, 'correl', None) and n1 in p2.correl:
        rho = float(p2.correl[n1])
    return rho * float(p1.stderr) * float(p2.stderr)


def cell_covariance(params, phase):
    """
    6x6 covariance matrix of [a, b, c, alpha, beta, gamma] of one phase,
    built from the fitted lmfit Parameters (stderr + correl, which lmfit
    derives from the covariance matrix, scaled by the reduced chi-square of
    the fit as lmfit does by default).

    Constrained parameters (expr='a_0' etc.) are mapped onto their master
    parameter, so e.g. a cubic cell gets a fully correlated a=b=c block.
    Fixed parameters contribute zero. Returns None if any refined cell
    parameter has no standard error (covariance unavailable).
    """
    if params is None:
        return None
    roots = []
    for nm in CELL_NAMES:
        name = f'{nm}_{phase}'
        if name not in params:
            return None
        p = params[name]
        if p.expr is not None and p.expr.strip():
            root = p.expr.strip()
            if root not in params:
                return None          # compound expression: not supported
            p_root = params[root]
            if p_root.expr is not None and p_root.expr.strip():
                return None
            roots.append(root if p_root.vary else None)
        else:
            roots.append(name if p.vary else None)

    cov = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if roots[i] is None or roots[j] is None:
                continue
            c = _param_cov(params, roots[i], roots[j])
            if not np.isfinite(c):
                return None
            cov[i, j] = c
    return cov


def covariance_ok(result, condLimit=1e10):
    """True if lmfit's covariance is usable: present, every refined
    parameter has a finite positive standard error and the column-scaled
    Jacobian is not (numerically) singular."""
    if getattr(result, 'covar', None) is None:
        return False
    for n in result.var_names:
        s = result.params[n].stderr
        if s is None or not np.isfinite(s) or s <= 0:
            return False
    J = getattr(result, 'jac', None)
    if J is None:
        return True
    J = np.asarray(J, dtype=float)
    norms = np.linalg.norm(J, axis=0)
    if J.size == 0 or np.any(norms == 0):
        return False
    sv = np.linalg.svd(J / norms, compute_uv=False)
    return sv[-1] > 0 and sv[0] / sv[-1] < condLimit


def robust_covariance(result, priority=None, depTol=1e-7):
    """
    Covariance of a least-squares fit whose normal matrix is singular.

    Refined parameters that cannot be determined from the data are left out
    of the covariance (i.e. treated as fixed at their fitted value):
      * 'no effect on the pattern'  : its Jacobian column is zero (e.g. the
                                      width of a reflection with amplitude 0),
      * 'not separable ...'         : its column is a linear combination of
                                      the columns kept before it (relative
                                      residual < depTol).
    Columns are taken in `priority` order (callable name -> sort key, lower
    first), so lattice parameters are kept in preference to amplitudes. The
    covariance of the kept parameters is (J^T J)^-1 * reduced chi^2 (as
    lmfit with scale_covar=True); their stderr and correl are written into
    result.params, and the stderr of constrained (expr) parameters is
    propagated from them.

    Returns {name: reason} of the parameters left out.
    """
    params = result.params
    names = list(result.var_names)
    J = np.asarray(result.jac, dtype=float)
    order = sorted(range(len(names)),
                   key=(lambda i: (priority(names[i]), i)) if priority else None)
    norms = np.linalg.norm(J, axis=0)
    big = float(np.max(norms)) if norms.size else 0.0
    excluded, kept, Q = {}, [], np.zeros((J.shape[0], 0))
    for i in order:
        v = J[:, i]
        if norms[i] <= 1e-12 * big:
            excluded[names[i]] = 'no effect on the pattern'
            continue
        r = v - Q @ (Q.T @ v)
        r = r - Q @ (Q.T @ r)                  # re-orthogonalise
        nr = np.linalg.norm(r)
        if nr <= depTol * norms[i]:
            excluded[names[i]] = 'not separable from other parameters'
            continue
        Q = np.column_stack((Q, r / nr))
        kept.append(i)
    for n in params:
        params[n].stderr = None
        params[n].correl = None
    if not kept:
        return excluded
    Jk = J[:, kept]
    try:
        cov = np.linalg.inv(Jk.T @ Jk) * float(result.redchi)
    except np.linalg.LinAlgError:
        return {n: 'not separable from other parameters' for n in names}
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    kn = [names[i] for i in kept]
    for a, na in enumerate(kn):
        params[na].stderr = float(sd[a])
        params[na].correl = {nb: float(cov[a, b] / (sd[a] * sd[b]))
                             for b, nb in enumerate(kn)
                             if b != a and sd[a] > 0 and sd[b] > 0}
    # Constrained (expr) parameters: linear propagation through their
    # expressions, evaluated by lmfit.
    exprs = [n for n in params if params[n].expr]
    if exprs:
        base = np.array([params[n].value for n in exprs])
        G = np.zeros((len(exprs), len(kn)))
        for b, nb in enumerate(kn):
            v0 = params[nb].value
            h = 1e-6 * max(abs(v0), 1e-6)
            params[nb].value = v0 + h
            params.update_constraints()
            G[:, b] = (np.array([params[n].value for n in exprs]) - base) / h
            params[nb].value = v0
        params.update_constraints()
        Ce = G @ cov
        ve = np.einsum('ij,ij->i', Ce, G)
        for e, ne in enumerate(exprs):
            if not np.any(G[e]):
                continue
            se = float(np.sqrt(max(ve[e], 0.0)))
            params[ne].stderr = se
            params[ne].correl = {nb: float(Ce[e, b] / (se * sd[b]))
                                 for b, nb in enumerate(kn)
                                 if se > 0 and sd[b] > 0}
    return excluded


# ============================================================================
#                             ERROR PROPAGATION
# ============================================================================
def _propagate(func, x, cov, relStep=1e-6):
    """Linear (first-order) error propagation of a scalar function of the
    cell vector x with covariance cov. Central-difference gradient."""
    if cov is None:
        return float('nan')
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(len(x)):
        if cov[i, i] == 0 and not np.any(cov[i]):
            continue
        h = relStep * max(abs(x[i]), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (func(xp) - func(xm)) / (2 * h)
    var = float(g @ cov @ g)
    return float(np.sqrt(var)) if var >= 0 else float('nan')


def cell_volume_and_error(unitCell, cov=None, uncErrors=None,
                          crystalSystem=None):
    """
    Unit-cell volume and its 1-sigma error (A^3) - the ONLY place where
    sigma(V) is computed.

    cov            6x6 covariance of [a, b, c, alpha, beta, gamma] from
                   cell_covariance(); used whenever it is available.
    uncErrors      6 lattice-parameter errors, used only if cov is None
                   (fallback route, e.g. sqrt(chi^2) errors). Only the
                   independent parameters of `crystalSystem` are
                   propagated (tied parameters follow their master) and
                   covariances are neglected.
    Returns (V, sigma_V); V is NaN for an invalid cell, sigma_V is NaN if
    no error information is available.
    """
    nan = float('nan')
    V = cryst.unit_cell_volume(list(unitCell))
    if V is None:
        return nan, nan

    def volume(x):
        v = cryst.unit_cell_volume(list(x))
        return nan if v is None else v

    if cov is not None:
        return V, _propagate(volume, unitCell, cov)
    if uncErrors is None:
        return V, nan

    flags = (cryst.free_lattice_flags(crystalSystem) if crystalSystem
             else (True,) * 6)
    diag = np.zeros(6)
    for i, free in enumerate(flags):
        e = float(uncErrors[i]) if i < len(uncErrors) else nan
        if free and np.isfinite(e) and e > 0:
            diag[i] = e * e
    if not np.any(diag):
        return V, nan
    if crystalSystem:
        return V, _propagate(
            lambda x: volume(cryst.enforce_symmetry(x, crystalSystem)),
            unitCell, np.diag(diag))
    return V, _propagate(volume, unitCell, np.diag(diag))


def reflection_values_and_errors(unitCell, cov, h, k, l, WL):
    """
    For one reflection return a dict with d, 2theta (deg), Q and their
    standard errors propagated from the cell covariance. The wavelength is
    treated as exact (it is not refined).
        sigma_Q      = 2 pi sigma_d / d^2
        sigma_2theta = 2 tan(theta) sigma_d / d       (radians -> degrees)
    """
    d = cryst.d_spacing_general(unitCell, h, k, l)
    sd = _propagate(lambda x: cryst.d_spacing_general(x, h, k, l), unitCell, cov)
    nan = float('nan')
    if not np.isfinite(d) or d <= 0 or WL / (2 * d) >= 1:
        return {'d': d, 'sig_d': sd, 'tt': nan, 'sig_tt': nan,
                'Q': nan, 'sig_Q': nan}
    theta = np.arcsin(WL / (2 * d))
    tt = float(np.degrees(2 * theta))
    Q = 2 * np.pi / d
    s_tt = float(np.degrees(2 * np.tan(theta) * sd / d)) if np.isfinite(sd) else nan
    s_Q = float(2 * np.pi * sd / d ** 2) if np.isfinite(sd) else nan
    return {'d': d, 'sig_d': sd, 'tt': tt, 'sig_tt': s_tt, 'Q': Q, 'sig_Q': s_Q}
