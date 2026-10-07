# plot_data_toolbox.py for EXODUS v0.1.4
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
# "Plot Data" pop-up: plot any fitted / guessed quantity against any other.
#
# Two layers, kept separate on purpose:
#
# 1. DATA LAYER (no Qt)  ->  FitDataset / build_dataset()
#    Turns the raw fit containers of the main window into a flat collection of
#    per-frame series (value + 1-sigma error). eos_fit_toolbox reuses this
#    layer, so both pop-ups always agree on what "V" or "P" means.
#
# 2. GUI LAYER  ->  PlotDataDialog / open_plot_data_dialog()
#
# The main window only has to hand over a *snapshot* dict
# (MainWindow._fit_data_snapshot):
#
#     {
#       'phases':         self.phases,                 # JCPDS phases dict
#       'sequential':     self.sequentialFitResults,   # {'frame_N': {...}}
#       'single':         self.results,                # single-pattern fit
#       'single_frame':   self.results_frame_index,
#       'pressure_guess': {phase_index: [(frame, P, fix, ov), ...]},
#       'colors':         {phase_index: '#rrggbb' / pg colour},
#     }
#
# Available series
# ----------------
# Frame-wide:  Frame, Rp, Rwp, chi^2, reduced chi^2 (w = 1/I_raw)
# Per phase:   V, V/V0, the INDEPENDENT lattice parameters of the crystal system
#              (e.g. only a for cubic) and their x/x0 ratios,
#              P (JCPDS)  - BM3 pressure from the fitted V and the EoS constants
#                           of the phase's own JCPDS file (needs K0 > 0),
#              P (guess)  - pressure-guess table of that phase, linearly
#                           interpolated (= exactly what the 'Pressure Guess'
#                           fitting mode used as starting pressure).
#
# Errors: lattice errors are the lmfit 1-sigma values, scaled by
# sqrt(reduced chi^2) (lmfit default, scale_covar=True), or the sqrt(chi^2)
# fallback if no covariance matrix was available. sigma(V) is the value
# stored with the fit (statistics_toolbox.cell_volume_and_error, full
# covariance), so it is identical to the one in the output files;
# sigma(P) = |dP/dV| sigma(V).
# ============================================================================
#                              Imported libraries
# ============================================================================

import csv
import os
import re

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import crystallography_toolbox as cryst
import eos_toolbox as eos
import statistics_toolbox as stats


# =============================================================================
#                                CONSTANTS
# =============================================================================
LATTICE_KEYS = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
LATTICE_LABELS = {'a': 'a', 'b': 'b', 'c': 'c',
                  'alpha': '\u03b1', 'beta': '\u03b2', 'gamma': '\u03b3'}
LATTICE_UNITS = {'a': '\u00c5', 'b': '\u00c5', 'c': '\u00c5',
                 'alpha': '\u00b0', 'beta': '\u00b0', 'gamma': '\u00b0'}

# Frame-wide statistics: key -> (label, unit)
FRAME_VARS = {
    'frame':  ('Frame', ''),
    'Rp':       ('Rp', ''),
    'Rwp':      ('Rwp', ''),
    'chi2':     ('\u03c7\u00b2', ''),
    'chi2_red': ('Reduced \u03c7\u00b2', ''),
    # 'nfev' (number of residual evaluations of the optimiser) removed
    # from the plot options: a solver diagnostic, not needed for analysis.
    # 'nfev':   ('Function evaluations', ''),
}

# Colours for multi-variable plots (bright enough for the dark background)
SERIES_PALETTE = ['#ffd400', '#00d0ff', '#ff5ad1', '#6cff6c', '#ff7043',
                  '#b388ff', '#ffffff', '#4dd0e1', '#ffab40', '#c6ff00']
PHASE_SYMBOLS = ['o', 's', 't', 'd', 'star', 'h', 'p', '+']

_PHASE_KEY_RE = re.compile(r'^phase_(\d+)(?:_unit_cell_fit)?$')

PLOT_BG = (61, 61, 61)   # same as the main window


# =============================================================================
#                               DATA LAYER
# =============================================================================
class Series:
    """One plottable quantity: values (and errors) per frame."""

    def __init__(self, key, label, unit, phase=None, phaseName=None,
                 name=None):
        self.key = key                # unique, e.g. ('phase', 0, 'V')
        self.label = label            # e.g. 'V'
        self.unit = unit              # e.g. 'A^3'
        self.phase = phase            # phase index or None (frame-wide)
        self.phase_name = phaseName
        self.name = name or label     # variable name without phase, e.g. 'V'
        self.data = {}                # frame -> (value, error)

    # -- convenience -----------------------------------------------------
    def add(self, frame, value, error=np.nan):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if not np.isfinite(v):
            return
        try:
            e = float(error)
        except (TypeError, ValueError):
            e = np.nan
        self.data[int(frame)] = (v, e)

    @property
    def frames(self):
        return np.array(sorted(self.data), dtype=int)

    def arrays(self, frames=None):
        """(frames, values, errors) - optionally restricted to `frames`."""
        fr = self.frames if frames is None else np.asarray(frames, dtype=int)
        vals = np.array([self.data[f][0] if f in self.data else np.nan
                         for f in fr], dtype=float)
        errs = np.array([self.data[f][1] if f in self.data else np.nan
                         for f in fr], dtype=float)
        return fr, vals, errs

    def axis_label(self):
        return f"{self.label} ({self.unit})" if self.unit else self.label

    def display(self):
        """Label used in combo boxes / legends / CSV headers."""
        base = self.axis_label()
        return f"{base} - {self.phase_name}" if self.phase_name else base

    def __len__(self):
        return len(self.data)


class FitDataset:
    """All series extracted from one snapshot of the main window."""

    def __init__(self):
        self.series = {}          # key -> Series (insertion-ordered)
        self.phase_info = {}      # phase index -> dict(name, color, cs, ...)

    def add(self, s):
        if len(s):
            self.series[s.key] = s

    def get(self, key):
        return self.series.get(key)

    def fitted_phases(self):
        """Phase indices that have at least one fitted volume."""
        return [p for p in sorted(self.phase_info)
                if ('phase', p, 'V') in self.series]

    def keys_for(self, phaseIndices, includeFrameVars=True):
        """Ordered keys of all series belonging to the given phases."""
        out = []
        if includeFrameVars:
            out += [k for k in self.series if k[0] == 'frame']
        for p in phaseIndices:
            out += [k for k in self.series if k[0] == 'phase' and k[1] == p]
        return out

    def phase_color(self, p):
        return self.phase_info.get(p, {}).get('color', '#ffd400')

    def is_empty(self):
        return not self.fitted_phases()


# ---- helpers -----------------------------------------------------------------
def _frame_number(key):
    try:
        return int(str(key).split('_')[1])
    except (IndexError, ValueError):
        return None


def _frame_results(snapshot):
    """{frame: frame_dict} merging the sequential cache and the single fit.
    The single-pattern fit wins for its own frame (same priority as the
    main window's _fitted_uc_for)."""
    out = {}
    for k, v in (snapshot.get('sequential') or {}).items():
        n = _frame_number(k)
        if n is not None and isinstance(v, dict):
            out[n] = v
    single = snapshot.get('single')
    sf = snapshot.get('single_frame')
    if isinstance(single, dict) and sf is not None:
        out[int(sf)] = single
    return out


def _phases_in_frame(frameDict):
    """{phase_index: (unit_cell, unit_cell_error or None, V, sigma_V)}
    V / sigma_V are the values stored by the fit (NaN if absent)."""
    found = {}
    for key in frameDict:
        m = _PHASE_KEY_RE.match(str(key))
        if m:
            found.setdefault(int(m.group(1)), None)
    out = {}
    for p in found:
        uc = frameDict.get(f'phase_{p}_unit_cell_fit',
                            frameDict.get(f'phase_{p}'))
        if uc is None:
            continue
        try:
            uc = np.asarray(uc, dtype=float).reshape(-1)[:6]
        except (TypeError, ValueError):
            continue
        if uc.size != 6:
            continue
        err = frameDict.get(f'phase_{p}_unit_cell_error',
                             frameDict.get(f'phase_{p}_error'))
        try:
            err = None if err is None else np.asarray(err, dtype=float).reshape(-1)[:6]
        except (TypeError, ValueError):
            err = None
        V = frameDict.get(f'phase_{p}_V_fit', np.nan)
        sV = frameDict.get(f'phase_{p}_V_error', np.nan)
        out[p] = (uc, err, V, sV)
    return out


def volume_with_error(uc, ucErr, crystalSystem, V=np.nan, sV=np.nan):
    """Cell volume and its 1-sigma error. Uses the values stored with the
    fit when present; for results without them (e.g. older sessions) the
    same statistics_toolbox function as the fit is used on the lattice
    errors (covariance not available here)."""
    if V is not None and np.isfinite(V):
        return float(V), (float(sV) if sV is not None else np.nan)
    return stats.cell_volume_and_error(uc, None, uncErrors=ucErr,
                                       crystalSystem=crystalSystem)


def _interp_guess(rows, frame):
    """Pressure-guess table value at `frame` (same rule as the fit)."""
    if not rows:
        return np.nan
    if len(rows) == 1:
        return float(rows[0][1])
    rows = sorted(rows, key=lambda r: float(r[0]))
    f = np.array([float(r[0]) for r in rows])
    P = np.array([float(r[1]) for r in rows])
    return float(np.interp(float(frame), f, P))


def build_dataset(snapshot):
    """Build a FitDataset from a main-window snapshot (see module doc)."""
    ds = FitDataset()
    phases = snapshot.get('phases') or {}
    colors = snapshot.get('colors') or {}
    n_phases = int(phases.get('phases_Number', 0) or 0)

    for p in range(n_phases):
        cs = cryst.normalise_crystal_system(phases.get(f'phase_{p}_crystal_system', 'CUBIC'))
        comp = np.asarray(phases.get(f'phase_{p}_compression_constants',
                                     [np.nan, 0, 4, 0, 0, 0]), dtype=float)
        uc0 = np.asarray(phases.get(f'phase_{p}_unit_cell_0',
                                    [np.nan] * 6), dtype=float)
        V0 = float(comp[0]) if np.isfinite(comp[0]) and comp[0] > 0 \
            else (cryst.unit_cell_volume(uc0) or np.nan)
        ds.phase_info[p] = {
            'name': str(phases.get(f'phase_{p}_name', f'phase_{p}')),
            'color': colors.get(p, SERIES_PALETTE[p % len(SERIES_PALETTE)]),
            'crystal_system': cs,
            'uc0': uc0,
            'V0': V0,
            'K0': float(comp[1]) if comp.size > 1 else 0.0,
            'K0P': float(comp[2]) if comp.size > 2 else 4.0,
        }

    frames = _frame_results(snapshot)

    # ---- frame-wide series ----------------------------------------------
    fseries = {k: Series(('frame', None, k), lab, unit, name=lab)
               for k, (lab, unit) in FRAME_VARS.items()}
    for fr, fd in frames.items():
        if not _phases_in_frame(fd):
            continue
        fseries['frame'].add(fr, fr, np.nan)
        # 'nfev' removed from the plot options (see FRAME_VARS)
        for k in ('Rp', 'Rwp', 'chi2', 'chi2_red'):
            if k in fd:
                fseries[k].add(fr, fd[k])
    for s in fseries.values():
        ds.add(s)
    fitted_frames = sorted(fseries['frame'].data)

    # ---- per-phase series ----------------------------------------------
    pg_rows = snapshot.get('pressure_guess') or {}
    for p, info in ds.phase_info.items():
        nm, cs = info['name'], info['crystal_system']
        flags = cryst.free_lattice_flags(cs)

        def mk(name, label, unit):
            return Series(('phase', p, name), label, unit, phase=p,
                          phaseName=nm, name=name)

        sV = mk('V', 'V', '\u00c5\u00b3')
        sVV0 = mk('V_V0', 'V/V\u2080', '')
        sP = mk('P_JCPDS', 'P [JCPDS]', 'GPa')
        lat = {}
        rat = {}
        for i, lk in enumerate(LATTICE_KEYS):
            if flags[i]:
                lat[lk] = mk(lk, LATTICE_LABELS[lk], LATTICE_UNITS[lk])
                if i < 3:
                    rat[lk] = mk(f'{lk}_{lk}0', f'{lk}/{lk}\u2080', '')

        has_eos = (np.isfinite(info['V0']) and info['V0'] > 0
                   and np.isfinite(info['K0']) and info['K0'] > 0)

        for fr, fd in frames.items():
            pin = _phases_in_frame(fd)
            if p not in pin:
                continue
            uc, err, Vst, sVst = pin[p]
            V, sigV = volume_with_error(uc, err, cs, Vst, sVst)
            sV.add(fr, V, sigV)
            if np.isfinite(info['V0']) and info['V0'] > 0:
                sVV0.add(fr, V / info['V0'], sigV / info['V0'])
            if has_eos and np.isfinite(V):
                P, sigP = eos.bm3_pressure_and_error(V, sigV, info['V0'],
                                                  info['K0'], info['K0P'])
                sP.add(fr, P, sigP)
            for i, lk in enumerate(LATTICE_KEYS):
                if lk in lat:
                    e = err[i] if err is not None else np.nan
                    lat[lk].add(fr, uc[i], e)
                    if lk in rat and np.isfinite(info['uc0'][i]) and info['uc0'][i] > 0:
                        rat[lk].add(fr, uc[i] / info['uc0'][i], e / info['uc0'][i])

        for s in (sV, sVV0, *lat.values(), *rat.values(), sP):
            ds.add(s)

        rows = pg_rows.get(p) or []
        if rows:
            sG = mk('P_guess', 'P [guess]', 'GPa')
            for fr in fitted_frames:
                sG.add(fr, _interp_guess(rows, fr))
            ds.add(sG)

    return ds


def has_fit_data(snapshot):
    """True if the snapshot contains at least one fitted unit cell."""
    return any(_phases_in_frame(fd) for fd in _frame_results(snapshot).values())


def align(xSeries, ySeries):
    """Frames present in both series -> (frames, x, sx, y, sy)."""
    common = sorted(set(xSeries.data) & set(ySeries.data))
    _, x, sx = xSeries.arrays(common)
    _, y, sy = ySeries.arrays(common)
    return np.array(common, dtype=int), x, sx, y, sy


# =============================================================================
#                          SHARED PLOT / IO HELPERS
# =============================================================================
def make_plot_widget():
    """pg.PlotWidget styled like the plots of the main window."""
    w = pg.PlotWidget(background=PLOT_BG)
    pi = w.getPlotItem()
    pi.getViewBox().setDefaultPadding(0.05)
    pi.layout.setContentsMargins(10, 10, 10, 10)
    pi.showGrid(x=True, y=True, alpha=0.15)
    try:
        pi.getViewBox().setMouseMode(pg.ViewBox.RectMode)   # rect zoom
    except Exception:
        pass
    return w


def export_png(parent, item, defaultName, onScreenWidth=800):
    """Save a pyqtgraph item (PlotItem / GraphicsLayout) as PNG."""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Save plot", defaultName, "PNG Image (*.png);;All Files (*)")
    if not path:
        return None
    if not path.lower().endswith('.png'):
        path += '.png'
    try:
        from pyqtgraph.exporters import ImageExporter
        ex = ImageExporter(item)
        ex.parameters()['width'] = max(400, int(onScreenWidth) * 2)
        ex.export(path)
    except Exception as e:
        QMessageBox.critical(parent, "Save plot", f"Could not save the plot:\n{e}")
        return None
    print(f"Plot saved to: {path}")
    return path


def ask_csv_path(parent, defaultName):
    path, _ = QFileDialog.getSaveFileName(
        parent, "Save data", defaultName, "CSV (*.csv);;All Files (*)")
    if not path:
        return None
    if not path.lower().endswith('.csv'):
        path += '.csv'
    return path


def color_icon(color, size=12):
    """Small square swatch used to mark a phase's colour in lists."""
    from PyQt5.QtGui import QIcon, QPixmap
    pm = QPixmap(size, size)
    pm.fill(pg.mkColor(color))
    return QIcon(pm)


def find_data(combo, key):
    """Index of `key` in a combo's item data (QComboBox's own lookup
    cannot match Python tuples, so compare in Python)."""
    for i in range(combo.count()):
        if combo.itemData(i) == key:
            return i
    return -1


def _fmt(v):
    return '' if v is None or not np.isfinite(v) else f'{v:.8g}'


# =============================================================================
#                               GUI LAYER
# =============================================================================
class PlotDataDialog(QtWidgets.QDialog):
    """Pop-up for plotting fitted / guessed quantities."""

    def __init__(self, parent, dataset):
        super().__init__(parent)
        self.setWindowTitle("EXODUS - Plot Data")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.resize(1250, 1250)
        self.ds = dataset
        self._updating = False
        self._build_ui()
        self._populate_phases()
        self._refresh_variable_lists()
        self.redraw()

    # ---- UI -----------------------------------------------------------
    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 4, 0)

        gb_ph = QtWidgets.QGroupBox("Phases")
        v = QtWidgets.QVBoxLayout(gb_ph)
        self.list_phases = QtWidgets.QListWidget()
        self.list_phases.setMaximumHeight(130)
        v.addWidget(self.list_phases)
        lv.addWidget(gb_ph)

        gb_x = QtWidgets.QGroupBox("X axis")
        v = QtWidgets.QVBoxLayout(gb_x)
        self.cmb_x = QtWidgets.QComboBox()
        v.addWidget(self.cmb_x)
        lv.addWidget(gb_x)

        gb_y = QtWidgets.QGroupBox("Y axis (tick one or more)")
        v = QtWidgets.QVBoxLayout(gb_y)
        self.list_y = QtWidgets.QListWidget()
        v.addWidget(self.list_y)
        lv.addWidget(gb_y, stretch=1)

        gb_opt = QtWidgets.QGroupBox("Display")
        g = QtWidgets.QVBoxLayout(gb_opt)
        self.chk_err = QtWidgets.QCheckBox("Error bars (1\u03c3)")
        self.chk_err.setChecked(True)
        self.chk_lines = QtWidgets.QCheckBox("Connect points")
        self.chk_lines.setChecked(False)
        g.addWidget(self.chk_err)
        g.addWidget(self.chk_lines)
        lv.addWidget(gb_opt)

        btns = QtWidgets.QHBoxLayout()
        self.btn_save_data = QtWidgets.QPushButton("Save Data")
        self.btn_save_plot = QtWidgets.QPushButton("Save Plot")
        btns.addWidget(self.btn_save_data)
        btns.addWidget(self.btn_save_plot)
        lv.addLayout(btns)
        self.btn_close = QtWidgets.QPushButton("Close")
        lv.addWidget(self.btn_close)

        self.plot = make_plot_widget()
        self.legend = self.plot.getPlotItem().addLegend(offset=(10, 10))

        splitter.addWidget(left)
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 930])

        self.list_phases.itemChanged.connect(self._on_phases_changed)
        self.cmb_x.currentIndexChanged.connect(self.redraw)
        self.list_y.itemChanged.connect(self._on_y_changed)
        self.chk_err.toggled.connect(self.redraw)
        self.chk_lines.toggled.connect(self.redraw)
        self.btn_save_data.clicked.connect(self.save_data)
        self.btn_save_plot.clicked.connect(self.save_plot)
        self.btn_close.clicked.connect(self.close)

    # ---- population ----------------------------------------------------
    def _populate_phases(self):
        self._updating = True
        try:
            for i, p in enumerate(self.ds.fitted_phases()):
                it = QtWidgets.QListWidgetItem(self.ds.phase_info[p]['name'])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                # Default: first fitted phase only
                it.setCheckState(Qt.Checked if i == 0 else Qt.Unchecked)
                it.setData(Qt.UserRole, p)
                it.setIcon(color_icon(self.ds.phase_color(p)))
                self.list_phases.addItem(it)
        finally:
            self._updating = False

    def selected_phases(self):
        out = []
        for i in range(self.list_phases.count()):
            it = self.list_phases.item(i)
            if it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _checked_y_keys(self):
        return [self.list_y.item(i).data(Qt.UserRole)
                for i in range(self.list_y.count())
                if self.list_y.item(i).checkState() == Qt.Checked]

    def _refresh_variable_lists(self):
        """Rebuild the X combo and the Y list for the selected phases,
        keeping the user's choices where they still exist."""
        phases = self.selected_phases()
        keys = self.ds.keys_for(phases)
        old_x = self.cmb_x.currentData()
        old_y = set(self._checked_y_keys())

        self._updating = True
        try:
            self.cmb_x.blockSignals(True)
            self.cmb_x.clear()
            for k in keys:
                self.cmb_x.addItem(self.ds.get(k).display(), k)
            idx = find_data(self.cmb_x, old_x) if old_x is not None else -1
            if idx < 0:
                idx = find_data(self.cmb_x, ('frame', None, 'frame'))
            self.cmb_x.setCurrentIndex(max(idx, 0))
            self.cmb_x.blockSignals(False)

            self.list_y.clear()
            ykeys = [k for k in keys if k != ('frame', None, 'frame')]
            any_checked = False
            for k in ykeys:
                s = self.ds.get(k)
                it = QtWidgets.QListWidgetItem(s.display())
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                chk = k in old_y
                any_checked |= chk
                it.setCheckState(Qt.Checked if chk else Qt.Unchecked)
                it.setData(Qt.UserRole, k)
                if s.phase is not None:
                    it.setIcon(color_icon(self.ds.phase_color(s.phase)))
                self.list_y.addItem(it)
            # Default: V of the first selected phase
            if not any_checked and phases:
                for i in range(self.list_y.count()):
                    if self.list_y.item(i).data(Qt.UserRole) == ('phase', phases[0], 'V'):
                        self.list_y.item(i).setCheckState(Qt.Checked)
                        break
        finally:
            self._updating = False

    def _on_phases_changed(self, _item):
        if self._updating:
            return
        self._refresh_variable_lists()
        self.redraw()

    def _on_y_changed(self, _item):
        if not self._updating:
            self.redraw()

    # ---- plotting -----------------------------------------------------
    def _style_for(self, ykeys):
        """{key: (colour, symbol)}. One variable -> phase colours; several
        variables -> one colour per variable, one symbol per phase."""
        names = []
        for k in ykeys:
            if k[2] not in names:
                names.append(k[2])
        phases = []
        for k in ykeys:
            if k[1] not in phases:
                phases.append(k[1])
        out = {}
        for k in ykeys:
            sym = PHASE_SYMBOLS[phases.index(k[1]) % len(PHASE_SYMBOLS)]
            if len(names) == 1 and k[1] is not None:
                col = self.ds.phase_color(k[1])
                sym = 'o'
            else:
                col = SERIES_PALETTE[names.index(k[2]) % len(SERIES_PALETTE)]
            out[k] = (col, sym)
        return out

    def redraw(self):
        if self._updating:
            return
        pi = self.plot.getPlotItem()
        pi.clear()
        self.legend.clear()
        xkey = self.cmb_x.currentData()
        ykeys = self._checked_y_keys()
        if xkey is None or not ykeys:
            return
        xs = self.ds.get(xkey)
        styles = self._style_for(ykeys)
        for k in ykeys:
            ys = self.ds.get(k)
            frames, x, sx, y, sy = align(xs, ys)
            if frames.size == 0:
                continue
            col, sym = styles[k]
            pen = pg.mkPen(col, width=1.2) if self.chk_lines.isChecked() else None
            pi.plot(x, y, pen=pen, symbol=sym, symbolSize=7,
                    symbolBrush=pg.mkBrush(col), symbolPen=pg.mkPen(col),
                    name=ys.display())
            if self.chk_err.isChecked():
                ey = np.nan_to_num(sy, nan=0.0)
                ex = np.nan_to_num(sx, nan=0.0)
                if np.any(ey > 0) or np.any(ex > 0):
                    pi.addItem(pg.ErrorBarItem(x=x, y=y, top=ey, bottom=ey,
                                               left=ex, right=ex, beam=0,
                                               pen=pg.mkPen(col, width=1)))
        pi.setLabel('bottom', xs.display() if xs.phase is not None else xs.axis_label())
        ylabels = {self.ds.get(k).axis_label() for k in ykeys}
        pi.setLabel('left', ylabels.pop() if len(ylabels) == 1 else 'Value')
        pi.enableAutoRange()

    # ---- saving --------------------------------------------------------
    def save_data(self):
        xkey = self.cmb_x.currentData()
        ykeys = self._checked_y_keys()
        if xkey is None or not ykeys:
            QMessageBox.information(self, "Save Data", "Nothing to save - "
                                    "choose at least one Y variable.")
            return
        path = ask_csv_path(self, "EXODUS_plot_data.csv")
        if not path:
            return
        cols = [k for k in [xkey] + ykeys if k != ('frame', None, 'frame')]
        frames = sorted(set().union(*[self.ds.get(k).data for k in [xkey] + ykeys]))
        header = ['Frame']
        for k in cols:
            s = self.ds.get(k)
            header += [s.display(), f'sigma {s.display()}']
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(header)
                for fr in frames:
                    row = [fr]
                    for k in cols:
                        v, e = self.ds.get(k).data.get(fr, (np.nan, np.nan))
                        row += [_fmt(v), _fmt(e)]
                    w.writerow(row)
        except OSError as e:
            QMessageBox.critical(self, "Save Data", f"Could not write file:\n{e}")
            return
        print(f"Plot data saved to: {path}")

    def save_plot(self):
        export_png(self, self.plot.getPlotItem(), "EXODUS_plot.png",
                   self.plot.width())


# =============================================================================
#                          ENTRY POINT FOR THE MAIN GUI
# =============================================================================
def open_plot_data_dialog(parent, snapshot):
    """Called by the main window's 'Plot Data' button. Returns the dialog
    (the caller keeps a reference so it stays alive) or None."""
    if not has_fit_data(snapshot):
        QMessageBox.information(parent, "Plot Data", "Please fit data first")
        return None
    ds = build_dataset(snapshot)
    if ds.is_empty():
        QMessageBox.information(parent, "Plot Data", "Please fit data first")
        return None
    dlg = PlotDataDialog(parent, ds)
    dlg.setAttribute(Qt.WA_DeleteOnClose)
    dlg.show()
    return dlg
