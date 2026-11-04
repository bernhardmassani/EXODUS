# eos_fit_toolbox.py for EXODUS v0.1.4
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
# "EoS Fit" pop-up: fit an isothermal equation of state to the P-V data that
# come out of the Pawley refinement.
#
# Layers
# ------
# 1. EoS MODELS (pure numpy)       -> eos_toolbox (EOS_MODELS, eos_pressure)
# 2. FITTING (lmfit, no Qt)        -> fit_eos()  -> EoSFitResult
# 3. GUI                           -> FitEoSDialog / open_fit_eos_dialog()
#
# The data (P and V per frame, with 1-sigma errors) come from
# plot_data_toolbox.build_dataset(), so this pop-up and "Plot Data" always
# use identical numbers. The main window only passes its snapshot dict
# (MainWindow._fit_data_snapshot).
#
# K0' and K0'': a truncated EoS fixes the next derivative. BM2 implies
# K0' = 4; BM2, BM3, Vinet, natural strain and Murnaghan imply a K0''
# (eos_toolbox.implied_kpp). These values are shown as "implied"; only BM4
# refines K0''. param_tag() is the single place that decides whether a
# parameter is shown as refined (+- error), fixed or implied.
#
# Fitting
# -------
# Residual = P_obs - P_model(V_obs). "Weighted" uses the effective variance
# sigma^2 = sigma_P^2 + (dP/dV * sigma_V)^2 [1], so the errors of BOTH axes
# count (dP/dV is re-evaluated with the current parameters on every
# iteration). Parameter errors are lmfit's 1-sigma values, scaled by
# sqrt(reduced chi^2) (lmfit default, scale_covar=True).
#
# References
# ----------
# [1] J. Orear, "Least squares when both variables have uncertainties",
#     Am. J. Phys. 50, 912-916 (1982).
# EoS models: see eos_toolbox.py.
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import csv
import datetime

import numpy as np
import pyqtgraph as pg
from lmfit import Parameters, minimize
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

RESID_BLUE = (70, 130, 255)   # blue residuals, readable on the dark plot

import eos_toolbox as eos
import plot_data_toolbox as pdt


# =============================================================================
#                               EOS MODELS
# =============================================================================
EOS_MODELS = eos.EOS_MODELS
DEFAULT_MODEL = eos.DEFAULT_MODEL
eos_pressure = eos.eos_pressure
implied_kpp = eos.implied_kpp

PARAM_NAMES = ('V0', 'K0', 'Kp', 'Kpp')
PARAM_LABELS = {'V0': 'V\u2080 (\u00c5\u00b3)', 'K0': 'K\u2080 (GPa)',
                'Kp': "K\u2080'", 'Kpp': "K\u2080'' (1/GPa)"}


def param_tag(model, res, nm, fmt='.4g'):
    """How a fitted parameter is labelled everywhere (error column, stats
    box, legend, CSV): 'implied' (fixed by the truncation of the model),
    'fixed' (fixed by the user) or '+- error'."""
    _fn, uses_kp, uses_kpp = EOS_MODELS[model]
    if (nm == 'Kp' and not uses_kp) or (nm == 'Kpp' and not uses_kpp):
        return "implied"
    if res.fixed.get(nm, False):
        return "fixed"
    e = res.errors.get(nm, np.nan)
    return f"\u00b1 {e:{fmt}}" if np.isfinite(e) else "\u00b1 n/a"


# =============================================================================
#                                 FITTING
# =============================================================================
class EoSFitResult:
    """Plain container for everything the GUI / CSV writer needs."""

    def __init__(self):
        self.model = ''
        self.values = {}       # name -> value
        self.errors = {}       # name -> 1-sigma (nan if n/a)
        self.fixed = {}        # name -> bool
        self.stats = {}        # ordered dict-like of statistics
        self.success = False
        self.message = ''
        self.weighted = False


def fit_eos(model, V, P, sV=None, sP=None, start=None, fix=None,
            weighted=False):
    """Least-squares fit of P(V).

    start : dict V0/K0/Kp/Kpp start values
    fix   : dict name -> bool
    Returns EoSFitResult.
    """
    fn, uses_kp, uses_kpp = EOS_MODELS[model]
    V = np.asarray(V, dtype=float)
    P = np.asarray(P, dtype=float)
    sV = np.zeros_like(V) if sV is None else np.nan_to_num(np.asarray(sV, float), nan=0.0)
    sP = np.zeros_like(P) if sP is None else np.nan_to_num(np.asarray(sP, float), nan=0.0)
    start = dict(start or {})
    fix = dict(fix or {})

    if not uses_kp:            # BM2: K' = 4 by definition
        start['Kp'] = 4.0
        fix['Kp'] = True
    if not uses_kpp:
        start['Kpp'] = start.get('Kpp', 0.0)
        fix['Kpp'] = True

    params = Parameters()
    params.add('V0', value=float(start.get('V0', np.max(V))), min=1e-6,
               vary=not fix.get('V0', False))
    params.add('K0', value=float(start.get('K0', 20.0)), min=1e-6,
               vary=not fix.get('K0', False))
    params.add('Kp', value=float(start.get('Kp', 4.0)),
               vary=not fix.get('Kp', False))
    params.add('Kpp', value=float(start.get('Kpp', 0.0)),
               vary=not fix.get('Kpp', False))

    n_vary = sum(1 for p in params.values() if p.vary)
    res = EoSFitResult()
    res.model = model
    if V.size <= n_vary:
        res.message = (f"Not enough data points ({V.size}) for "
                       f"{n_vary} refined parameter(s).")
        return res
    if n_vary == 0:
        res.message = "All parameters are fixed - nothing to fit."
        return res

    use_w = bool(weighted) and (np.any(sP > 0) or np.any(sV > 0))
    res.weighted = use_w

    def _sigma(pv):
        h = np.maximum(np.abs(V) * 1e-6, 1e-8)
        with np.errstate(all='ignore'):
            dPdV = (fn(V + h, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp']) -
                    fn(V - h, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp'])) / (2 * h)
        s = np.sqrt(sP ** 2 + (dPdV * sV) ** 2)
        good = np.isfinite(s) & (s > 0)
        fallback = np.median(s[good]) if np.any(good) else 1.0
        return np.where(good, s, fallback)

    def _residual(pars):
        pv = pars.valuesdict()
        with np.errstate(all='ignore'):
            r = P - fn(V, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp'])
        if use_w:
            r = r / _sigma(pv)
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

    try:
        out = minimize(_residual, params, method='leastsq')
    except Exception as e:     # pragma: no cover - defensive
        res.message = f"Fit failed: {e}"
        return res

    pv = out.params.valuesdict()
    for nm in PARAM_NAMES:
        par = out.params[nm]
        res.values[nm] = float(par.value)
        res.fixed[nm] = not par.vary
        res.errors[nm] = (float(par.stderr) if (par.vary and par.stderr is not None)
                          else np.nan)
    if not uses_kpp:
        res.values['Kpp'] = implied_kpp(model, pv['K0'], pv['Kp'])
    res.success = bool(out.success)
    res.message = str(out.message)

    Pfit = fn(V, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp'])
    dP = P - Pfit
    ss_res = float(np.sum(dP ** 2))
    ss_tot = float(np.sum((P - np.mean(P)) ** 2))
    corr = np.nan
    try:
        if out.params['K0'].vary and out.params['Kp'].vary:
            corr = float(out.params['K0'].correl.get('Kp', np.nan))
    except Exception:
        pass
    res.stats = {
        'N points': int(V.size),
        'N refined parameters': int(n_vary),
        'Degrees of freedom': int(V.size - n_vary),
        'Weighting': ('effective variance (sigma_P, sigma_V)'
                      if use_w else 'none (unweighted)'),
        'chi^2': float(out.chisqr),
        'reduced chi^2': float(out.redchi),
        'RMS(P - P_fit) [GPa]': float(np.sqrt(ss_res / V.size)),
        'max|P - P_fit| [GPa]': float(np.max(np.abs(dP))),
        'R^2 (P)': (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "corr(K0, K0')": corr,
        'Function evaluations': int(out.nfev),
    }
    return res


# =============================================================================
#                                   GUI
# =============================================================================
class FitEoSDialog(QtWidgets.QDialog):
    """Pop-up for fitting an EoS to the fitted P-V data."""

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.setWindowTitle("EXODUS - EoS Fit")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(1300, 820)
        self.ds = dataset
        self.fit_result = None     # EoSFitResult of the last fit
        self._fit_current = False  # False once anything changed after a fit
        self._loading = False
        self._build_ui()
        self._populate_sources()
        self._on_sources_changed()

    # ---- UI construction ----------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        # a) model
        gb_m = QtWidgets.QGroupBox("EoS model")
        v = QtWidgets.QVBoxLayout(gb_m)
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(list(EOS_MODELS))
        self.cmb_model.setCurrentText(DEFAULT_MODEL)
        v.addWidget(self.cmb_model)
        lv.addWidget(gb_m)

        # b) data sources
        gb_d = QtWidgets.QGroupBox("Data")
        f = QtWidgets.QFormLayout(gb_d)
        self.cmb_P = QtWidgets.QComboBox()
        self.cmb_P.setToolTip("Pressure: P from the JCPDS EoS of a phase "
                              "(e.g. a pressure standard), or the "
                              "pressure-guess table of a phase.")
        self.cmb_V = QtWidgets.QComboBox()
        self.cmb_V.setToolTip("Fitted unit-cell volume used for the EoS fit.")
        f.addRow("Pressure from:", self.cmb_P)
        f.addRow("Volume of:", self.cmb_V)
        rng = QtWidgets.QHBoxLayout()
        self.spn_fmin = QtWidgets.QSpinBox()
        self.spn_fmax = QtWidgets.QSpinBox()
        for s in (self.spn_fmin, self.spn_fmax):
            s.setRange(0, 10_000_000)
        rng.addWidget(self.spn_fmin)
        rng.addWidget(QtWidgets.QLabel("to"))
        rng.addWidget(self.spn_fmax)
        f.addRow("Frames:", rng)
        self.cmb_weight = QtWidgets.QComboBox()
        self.cmb_weight.addItems(["Unweighted", "Weighted (\u03c3P, \u03c3V)"])
        f.addRow("Weighting:", self.cmb_weight)
        self.lbl_npts = QtWidgets.QLabel("")
        self.lbl_npts.setStyleSheet("color: #aaaaaa;")
        f.addRow("", self.lbl_npts)
        lv.addWidget(gb_d)

        # c) parameters
        gb_p = QtWidgets.QGroupBox("Parameters")
        g = QtWidgets.QGridLayout(gb_p)
        for col, txt in enumerate(("", "Value", "Fix", "\u00b1 Error")):
            lab = QtWidgets.QLabel(f"<b>{txt}</b>")
            g.addWidget(lab, 0, col)
        self.spn = {}
        self.chk_fix = {}
        self.lbl_err = {}
        specs = {'V0': (1e-3, 1e6, 4, 1.0), 'K0': (1e-3, 1e4, 3, 1.0),
                 'Kp': (-50.0, 50.0, 4, 0.1), 'Kpp': (-100.0, 100.0, 5, 0.01)}
        for r, nm in enumerate(PARAM_NAMES, start=1):
            lo, hi, dec, step = specs[nm]
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setMinimumWidth(110)
            c = QtWidgets.QCheckBox()
            e = QtWidgets.QLabel("-")
            e.setMinimumWidth(90)
            g.addWidget(QtWidgets.QLabel(PARAM_LABELS[nm]), r, 0)
            g.addWidget(s, r, 1)
            g.addWidget(c, r, 2, alignment=Qt.AlignCenter)
            g.addWidget(e, r, 3)
            self.spn[nm], self.chk_fix[nm], self.lbl_err[nm] = s, c, e
        self.spn['K0'].setValue(20.0)
        self.spn['Kp'].setValue(4.0)
        self.spn['Kpp'].setValue(implied_kpp(DEFAULT_MODEL, 20.0, 4.0))
        pb = QtWidgets.QHBoxLayout()
        self.btn_from_jcpds = QtWidgets.QPushButton("From JCPDS")
        self.btn_from_jcpds.setToolTip("Use V0, K0, K0' of the selected "
                                       "volume phase's JCPDS file as start.")
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setToolTip("V0 = volume of the first frame, "
                                  "K0 = 20 GPa, K0' = 4")
        pb.addWidget(self.btn_from_jcpds)
        pb.addWidget(self.btn_reset)
        g.addLayout(pb, len(PARAM_NAMES) + 1, 0, 1, 4)
        lv.addWidget(gb_p)

        # e) fit button
        self.btn_fit = QtWidgets.QPushButton("Fit EoS")
        self.btn_fit.setMinimumHeight(32)
        f2 = self.btn_fit.font(); f2.setBold(True); self.btn_fit.setFont(f2)
        lv.addWidget(self.btn_fit)

        # statistics
        gb_s = QtWidgets.QGroupBox("Fit statistics")
        v = QtWidgets.QVBoxLayout(gb_s)
        self.txt_stats = QtWidgets.QPlainTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setMinimumHeight(140)
        v.addWidget(self.txt_stats)
        lv.addWidget(gb_s, stretch=1)

        # d) residual toggle + f) save
        self.chk_resid = QtWidgets.QCheckBox("Show residuals")
        self.chk_resid.setChecked(True)
        lv.addWidget(self.chk_resid)
        btns = QtWidgets.QHBoxLayout()
        self.btn_save_data = QtWidgets.QPushButton("Save Data")
        self.btn_save_plot = QtWidgets.QPushButton("Save Plot")
        btns.addWidget(self.btn_save_data)
        btns.addWidget(self.btn_save_plot)
        lv.addLayout(btns)
        self.btn_close = QtWidgets.QPushButton("Close")
        lv.addWidget(self.btn_close)

        # d) plots: V(P) on top (pressure on x, volume on y), pressure
        #    residuals below (x-linked, plotted against P)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(pdt.PLOT_BG)
        self.p_main = self.glw.addPlot(row=0, col=0)
        self.p_res = self.glw.addPlot(row=1, col=0)
        self.p_res.setXLink(self.p_main)
        self.glw.ci.layout.setRowStretchFactor(0, 3)
        self.glw.ci.layout.setRowStretchFactor(1, 1)
        for p in (self.p_main, self.p_res):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.layout.setContentsMargins(10, 10, 10, 10)
            p.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.p_main.setLabel('left', 'V (\u00c5\u00b3)')
        self.p_main.setLabel('bottom', 'P (GPa)')
        self.p_res.setLabel('left', '\u0394P (GPa)')
        self.p_res.setLabel('bottom', 'P (GPa)')
        self.legend = self.p_main.addLegend(offset=(-10, 10))
        self._res_visible = True

        splitter.addWidget(left)
        splitter.addWidget(self.glw)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 930])

        # signals
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.cmb_P.currentIndexChanged.connect(self._on_sources_changed)
        self.cmb_V.currentIndexChanged.connect(self._on_volume_changed)
        self.spn_fmin.valueChanged.connect(self._on_input_changed)
        self.spn_fmax.valueChanged.connect(self._on_input_changed)
        self.cmb_weight.currentIndexChanged.connect(self._on_input_changed)
        for nm in PARAM_NAMES:
            self.spn[nm].valueChanged.connect(self._on_input_changed)
            self.chk_fix[nm].toggled.connect(self._on_input_changed)
        self.btn_from_jcpds.clicked.connect(self._on_from_jcpds)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_fit.clicked.connect(self.run_fit)
        self.chk_resid.toggled.connect(self._on_toggle_residuals)
        self.btn_save_data.clicked.connect(self.save_data)
        self.btn_save_plot.clicked.connect(self.save_plot)
        self.btn_close.clicked.connect(self.close)
        self._apply_model_enable()

    # ---- sources -------------------------------------------------------
    def _populate_sources(self):
        self._loading = True
        try:
            v_phases = self.ds.fitted_phases()
            for p in v_phases:
                self.cmb_V.addItem(pdt.color_icon(self.ds.phase_color(p)),
                                   self.ds.phase_info[p]['name'], ('phase', p, 'V'))
            p_keys = [k for k in self.ds.series
                      if k[0] == 'phase' and k[2] in ('P_JCPDS', 'P_guess')]
            # JCPDS pressures first, then guesses
            p_keys.sort(key=lambda k: (k[2] != 'P_JCPDS', k[1]))
            for k in p_keys:
                s = self.ds.get(k)
                src = 'JCPDS' if k[2] == 'P_JCPDS' else 'pressure guess'
                self.cmb_P.addItem(pdt.color_icon(self.ds.phase_color(k[1])),
                                   f"{s.phase_name} ({src})", k)
            # Default P: a JCPDS pressure of a phase OTHER than the volume
            # phase (typical: pressure standard), else anything available.
            v0 = v_phases[0] if v_phases else None
            best = next((i for i, k in enumerate(p_keys)
                         if k[2] == 'P_JCPDS' and k[1] != v0), 0)
            if p_keys:
                self.cmb_P.setCurrentIndex(best)
        finally:
            self._loading = False
        has_p = self.cmb_P.count() > 0
        self.btn_fit.setEnabled(has_p)
        if not has_p:
            self.txt_stats.setPlainText(
                "No pressure available.\n\nLoad a JCPDS file with K0 > 0 "
                "(e.g. a pressure standard) or fill the pressure-guess "
                "table of a phase, then refit.")

    def _series(self):
        kP, kV = self.cmb_P.currentData(), self.cmb_V.currentData()
        if kP is None or kV is None:
            return None, None
        return self.ds.get(kP), self.ds.get(kV)

    def _all_points(self):
        """(frames, V, sV, P, sP) for all common frames."""
        sP_, sV_ = self._series()
        if sP_ is None:
            e = np.array([])
            return e.astype(int), e, e, e, e
        frames, V, sV, P, sP = pdt.align(sV_, sP_)
        return frames, V, sV, P, sP

    def _points(self):
        """Common frames restricted to the frame range."""
        frames, V, sV, P, sP = self._all_points()
        m = (frames >= self.spn_fmin.value()) & (frames <= self.spn_fmax.value())
        m &= np.isfinite(V) & np.isfinite(P)
        return frames[m], V[m], sV[m], P[m], sP[m]

    def _first_frame_volume(self):
        _sP, sV = self._series()
        if sV is None or not len(sV):
            return None
        return sV.data[int(sV.frames[0])][0]

    # ---- handlers ------------------------------------------------------
    def _mark_stale(self):
        self._fit_current = False
        for nm in PARAM_NAMES:
            self.lbl_err[nm].setText("-")

    def _on_sources_changed(self, *_):
        if self._loading:
            return
        frames, *_ = self._all_points()
        self._loading = True
        try:
            if frames.size:
                for s in (self.spn_fmin, self.spn_fmax):
                    s.setRange(int(frames.min()), int(frames.max()))
                self.spn_fmin.setValue(int(frames.min()))
                self.spn_fmax.setValue(int(frames.max()))
            V1 = self._first_frame_volume()
            if V1 is not None:
                self.spn['V0'].setValue(V1)
        finally:
            self._loading = False
        self.fit_result = None
        self._mark_stale()
        self.redraw()

    def _on_volume_changed(self, *_):
        """If the pressure currently comes from the JCPDS of the phase that
        was just chosen as the volume, switch to another phase's JCPDS
        pressure (fitting a phase against its own EoS is circular). The
        user can still pick it deliberately afterwards."""
        if self._loading:
            return
        kV, kP = self.cmb_V.currentData(), self.cmb_P.currentData()
        if kV is not None and kP is not None and kP[1] == kV[1]:
            for i in range(self.cmb_P.count()):
                k = self.cmb_P.itemData(i)
                if k[1] != kV[1]:
                    self._loading = True
                    self.cmb_P.setCurrentIndex(i)
                    self._loading = False
                    break
        self._on_sources_changed()

    def _on_model_changed(self, *_):
        self._apply_model_enable()
        if 'BM4' in self.cmb_model.currentText():
            self._loading = True
            self.spn['Kpp'].setValue(implied_kpp('BM3', self.spn['K0'].value(),
                                                 self.spn['Kp'].value()))
            self._loading = False
        self._on_input_changed()

    def _apply_model_enable(self):
        _fn, uk, ukk = EOS_MODELS[self.cmb_model.currentText()]
        if not uk:
            self._loading = True
            self.spn['Kp'].setValue(4.0)
            self._loading = False
        for nm, on in (('Kp', uk), ('Kpp', ukk)):
            self.spn[nm].setEnabled(on)
            self.chk_fix[nm].setEnabled(on)
        self.spn['Kp'].setToolTip("" if uk else "Implied: K0' = 4 for BM2.")
        self.spn['Kpp'].setToolTip("" if ukk else
                                   "Implied by the model; refined only for BM4.")

    def _on_input_changed(self, *_):
        if self._loading:
            return
        self._mark_stale()
        self.redraw()

    def _on_from_jcpds(self):
        kV = self.cmb_V.currentData()
        if kV is None:
            return
        info = self.ds.phase_info[kV[1]]
        if not (np.isfinite(info['V0']) and info['K0'] > 0):
            QMessageBox.information(self, "From JCPDS",
                                    "This JCPDS file has no usable EoS (K0 = 0).")
            return
        self._loading = True
        self.spn['V0'].setValue(info['V0'])
        self.spn['K0'].setValue(info['K0'])
        self.spn['Kp'].setValue(info['K0P'])
        self.spn['Kpp'].setValue(implied_kpp('BM3', info['K0'], info['K0P']))
        self._loading = False
        self._apply_model_enable()
        self._on_input_changed()

    def _on_reset(self):
        self._loading = True
        V1 = self._first_frame_volume()
        if V1 is not None:
            self.spn['V0'].setValue(V1)
        self.spn['K0'].setValue(20.0)
        self.spn['Kp'].setValue(4.0)
        self.spn['Kpp'].setValue(implied_kpp('BM3', 20.0, 4.0))
        for c in self.chk_fix.values():
            c.setChecked(False)
        self._loading = False
        self._apply_model_enable()
        self._on_input_changed()

    def _on_toggle_residuals(self, on):
        if on and not self._res_visible:
            self.glw.ci.addItem(self.p_res, row=1, col=0)
            self.glw.ci.layout.setRowStretchFactor(1, 1)
        elif not on and self._res_visible:
            self.glw.ci.removeItem(self.p_res)
        self._res_visible = on
        self.redraw()

    # ---- parameters ----------------------------------------------------
    def _params(self):
        return {nm: self.spn[nm].value() for nm in PARAM_NAMES}

    def _curve_params(self):
        """Parameters for curve, residuals and saved P_fit: the exact fitted
        values after a fit (the spin boxes are rounded), else the start
        values."""
        if self._fit_current and self.fit_result is not None:
            return dict(self.fit_result.values)
        return self._params()

    def _fixes(self):
        return {nm: self.chk_fix[nm].isChecked() for nm in PARAM_NAMES}

    # ---- fit -----------------------------------------------------------
    def run_fit(self):
        frames, V, sV, P, sP = self._points()
        model = self.cmb_model.currentText()
        res = fit_eos(model, V, P, sV=sV, sP=sP, start=self._params(),
                      fix=self._fixes(),
                      weighted=self.cmb_weight.currentIndex() == 1)
        if not res.values:
            QMessageBox.warning(self, "EoS Fit", res.message)
            return
        self.fit_result = res
        self._loading = True
        try:
            for nm in PARAM_NAMES:
                self.spn[nm].setValue(res.values[nm])
        finally:
            self._loading = False
        for nm in PARAM_NAMES:
            self.lbl_err[nm].setText(param_tag(model, res, nm))
        self._fit_current = True
        self._show_stats(res)
        self.redraw()

    def _show_stats(self, res):
        lines = [f"Model: {res.model}",
                 f"Status: {'converged' if res.success else 'NOT converged'}",
                 ""]
        for nm in PARAM_NAMES:
            lines.append(f"{nm:>4} = {res.values[nm]:.6g}  "
                         f"{param_tag(res.model, res, nm)}")
        lines.append("")
        for k, v in res.stats.items():
            if isinstance(v, float):
                v = f"{v:.5g}" if np.isfinite(v) else "n/a"
            lines.append(f"{k}: {v}")
        if not res.success:
            lines += ["", res.message]
        self.txt_stats.setPlainText("\n".join(lines))

    # ---- plotting ------------------------------------------------------
    def redraw(self):
        self.p_main.clear()
        self.p_res.clear()
        self.legend.clear()
        frames, V, sV, P, sP = self._points()
        self.lbl_npts.setText(f"{frames.size} point(s) in range")
        if frames.size == 0:
            return
        kV = self.cmb_V.currentData()
        col = self.ds.phase_color(kV[1]) if kV else '#ffd400'

        # data: pressure on x, volume on y
        eP = np.nan_to_num(sP, nan=0.0)
        eV = np.nan_to_num(sV, nan=0.0)
        if np.any(eP > 0) or np.any(eV > 0):
            self.p_main.addItem(pg.ErrorBarItem(x=P, y=V, left=eP, right=eP,
                                                top=eV, bottom=eV, beam=0,
                                                pen=pg.mkPen(col, width=1)))
        self.p_main.plot(P, V, pen=None, symbol='o', symbolSize=7,
                         symbolBrush=pg.mkBrush(col), symbolPen=pg.mkPen(col),
                         name='Data')

        # curve: fit (solid) or current guess (dashed)
        model = self.cmb_model.currentText()
        pv = self._curve_params()
        vmin = float(np.min(V))
        vmax = float(max(np.max(V), pv['V0']))
        Vc = np.linspace(vmin - 0.02 * (vmax - vmin), vmax, 400)
        Vc = Vc[Vc > 0]
        Pc = eos_pressure(model, Vc, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp'])
        ok = np.isfinite(Pc)
        short = model.split(' ')[0]
        if self._fit_current:
            self.p_main.plot(Pc[ok], Vc[ok], pen=pg.mkPen('r', width=2),
                             name=f'{short} fit')
        else:
            self.p_main.plot(Pc[ok], Vc[ok],
                             pen=pg.mkPen('r', width=1.5, style=Qt.DashLine),
                             name=f'{short} (start values)')

        # residuals (blue) only for a current fit: delta P vs P
        if self._fit_current and self._res_visible:
            Pm = eos_pressure(model, V, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp'])
            dP = P - Pm
            self.p_res.addItem(pg.InfiniteLine(pos=0, angle=0,
                                               pen=pg.mkPen((150, 150, 150),
                                                            style=Qt.DashLine)))
            if np.any(eP > 0):
                self.p_res.addItem(pg.ErrorBarItem(x=P, y=dP, top=eP, bottom=eP,
                                                   beam=0,
                                                   pen=pg.mkPen(RESID_BLUE, width=1)))
            self.p_res.plot(P, dP, pen=None, symbol='o', symbolSize=6,
                            symbolBrush=pg.mkBrush(RESID_BLUE), symbolPen=pg.mkPen(RESID_BLUE))

        # parameter summary in the legend after a fit
        if self._fit_current and self.fit_result is not None:
            r = self.fit_result
            for nm in PARAM_NAMES:
                self.legend.addItem(
                    pg.PlotDataItem(pen=None),
                    f"{PARAM_LABELS[nm]} = {r.values[nm]:.5g} "
                    f"{param_tag(model, r, nm, '.3g')}")
        self.p_main.enableAutoRange()
        self.p_res.enableAutoRange(axis='y')

    # ---- saving --------------------------------------------------------
    def save_data(self):
        frames, V, sV, P, sP = self._points()
        if frames.size == 0:
            QMessageBox.information(self, "Save Data", "No data to save.")
            return
        path = pdt.ask_csv_path(self, "EXODUS_EoS_fit.csv")
        if not path:
            return
        model = self.cmb_model.currentText()
        pv = self._curve_params()
        Pm = eos_pressure(model, V, pv['V0'], pv['K0'], pv['Kp'], pv['Kpp'])
        dP = P - Pm
        r = self.fit_result if self._fit_current else None
        hdr = [f"EXODUS EoS fit - {datetime.datetime.now():%Y-%m-%d %H:%M}",
               f"Model: {model}",
               f"Pressure from: {self.cmb_P.currentText()}",
               f"Volume of: {self.cmb_V.currentText()}",
               f"Frames: {self.spn_fmin.value()} - {self.spn_fmax.value()}",
               f"Weighting: {self.cmb_weight.currentText()}"]
        if r is None:
            hdr.append("STATUS: NOT FITTED - P_fit/residual use the start values")
        for nm in PARAM_NAMES:
            if r is not None:
                tag = param_tag(model, r, nm, '.6g').replace('\u00b1', '+/-')
                hdr.append(f"{nm} = {r.values[nm]:.8g} {tag}")
            else:
                hdr.append(f"{nm} = {pv[nm]:.8g} (start value)")
        if r is not None:
            for k, v in r.stats.items():
                hdr.append(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}")
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                for line in hdr:
                    f.write(f"# {line}\n")
                w = csv.writer(f)
                w.writerow(['Frame', 'P_GPa', 'sigma_P_GPa', 'V_A3', 'sigma_V_A3',
                            'P_fit_GPa', 'residual_P_minus_Pfit_GPa'])
                for i in range(frames.size):
                    w.writerow([int(frames[i]), pdt._fmt(P[i]), pdt._fmt(sP[i]),
                                pdt._fmt(V[i]), pdt._fmt(sV[i]),
                                pdt._fmt(Pm[i]), pdt._fmt(dP[i])])
        except OSError as e:
            QMessageBox.critical(self, "Save Data", f"Could not write file:\n{e}")
            return
        print(f"EoS data saved to: {path}")

    def save_plot(self):
        pdt.export_png(self, self.glw.ci, "EXODUS_EoS_fit.png", self.glw.width())


# =============================================================================
#                          ENTRY POINT FOR THE MAIN GUI
# =============================================================================
def open_fit_eos_dialog(parent, snapshot):
    """Called by the main window's 'EoS Fit' button."""
    if not pdt.has_fit_data(snapshot):
        QMessageBox.information(parent, "EoS Fit", "Please fit data first")
        return None
    ds = pdt.build_dataset(snapshot)
    if ds.is_empty():
        QMessageBox.information(parent, "EoS Fit", "Please fit data first")
        return None
    dlg = FitEoSDialog(parent, ds)
    dlg.setAttribute(Qt.WA_DeleteOnClose)
    dlg.show()
    return dlg
