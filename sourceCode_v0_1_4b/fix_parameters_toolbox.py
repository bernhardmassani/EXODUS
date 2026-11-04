# fix_parameters_toolbox.py for EXODUS v0.1.4
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
#
# fix_parameters_toolbox.py  (EXODUS v0.1.4)
#
# Per-frame parameter overrides for the "Pressure Guess" fitting mode.
#
# Every row in a pressure-guess tab ([Frame, Pressure, Fix, Edit]) can carry
# an *override* dictionary, edited through FrameParameterDialog (opened by the
# row's "Edit" button). Overrides apply ONLY to the frame whose number matches
# the row's "Frame" value, and ONLY to the phase the tab belongs to -
# exactly like the existing "Fix" checkbox.

# Override dictionary layout (all keys optional; an empty dict = no override)
# ---------------------------------------------------------------------------
#     {
#       'lattice': {                     # starting values / fixes
#           'a':   {'value': 3.912, 'fix': True},
#           'c':   {'value': 6.020, 'fix': False},   # custom start, refined
#           ...                          # names: a b c alp bet gam
#       },
#       'fit': {                         # fit inputs for this frame
#           'amp_prefactor': 0.8,        # per phase
#           'amp_bounds':    5.0,        # per phase (upper-bound multiplier)
#           'max_shift':     0.02,       # per phase (lattice bounds, 0 = fix)
#           'sig_guess':     0.015,      # whole frame (sigma is shared)
#           'sig_bounds':    3.0,        # whole frame
#       },
#     }

# Precedence rules (applied in EXODUS_main._process_next_frame)
# -------------------------------------------------------------
# * If ANY lattice parameter is fixed in the override, the row's "Fix"
#   (= hold the whole lattice at the guessed pressure) is IGNORED for this
#   frame: only the explicitly fixed parameters are held, everything else
#   refines.
# * Lattice parameters tied by symmetry (e.g. b = a for tetragonal) follow
#   their parent automatically and cannot be set on their own.
# * Sigma inputs are global to a frame (sigma parameters are shared between
#   phases in the 'fixed' and 'scherrer' modes), so if several phases
#   override them for the same frame, the first phase (lowest index) wins.
#   In the 'caglioti' mode the widths come from U, V, W and the sigma
#   inputs have no effect.
#
# Overrides are kept in memory only (they live on the table rows). They are
# NOT written to disk - a session file may be added later.
#
#
#
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import copy

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

import crystallography_toolbox as cryst
import eos_toolbox as eos


# ============================================================================
#                                 CONSTANTS
# ============================================================================
LATTICE_NAMES = cryst.LATTICE_NAMES
LATTICE_LABELS = {'a': 'a', 'b': 'b', 'c': 'c',
                  'alp': '\u03b1', 'bet': '\u03b2', 'gam': '\u03b3'}
LATTICE_UNITS = {'a': ' \u00c5', 'b': ' \u00c5', 'c': ' \u00c5',
                 'alp': ' \u00b0', 'bet': ' \u00b0', 'gam': ' \u00b0'}

# Fit-input keys: (label, tooltip, scope)
FIT_INPUTS = (
    ('amp_prefactor', 'Amplitude',
     'Amplitude pre-factor for THIS phase (start = max(data) x value).',
     'phase'),
    ('amp_bounds', 'Amplitude Bounds',
     'Upper bound multiplier for the amplitudes of THIS phase '
     '(max = start amplitude x value).',
     'phase'),
    ('max_shift', 'Lattice Bounds (0=fix)',
     'Fractional lattice bounds for THIS phase (0.05 = \u00b15 %). '
     '0 fixes every lattice parameter of this phase.',
     'phase'),
    ('sig_guess', 'Sigma',
     'Starting peak width. Applies to the WHOLE frame, because sigma is '
     'shared between phases in most sigma modes.',
     'frame'),
    ('sig_bounds', 'Sigma Bounds',
     'Upper sigma bound multiplier. Applies to the WHOLE frame.',
     'frame'),
)

# Keys this module stamps onto the phases dict for one frame. The GUI
# clears them before every fit so nothing leaks into the next frame.
PHASE_OVERRIDE_SUFFIXES = ('_fix_lattice', '_fix_params',
                           '_max_shift', '_amp_scale', '_amp_bounds')


# ============================================================================
#                              OVERRIDE HELPERS
# ============================================================================
def empty_overrides():
    return {'lattice': {}, 'fit': {}}


def copy_overrides(ov):
    """Deep copy (None-safe) so table snapshots don't share state."""
    if not ov:
        return empty_overrides()
    out = copy.deepcopy(ov)
    out.setdefault('lattice', {})
    out.setdefault('fit', {})
    return out


def has_overrides(ov):
    return bool(ov) and (bool(ov.get('lattice')) or bool(ov.get('fit')))


def fixed_lattice_names(ov):
    """Names of lattice parameters that are FIXED by the override."""
    if not ov:
        return []
    return [nm for nm in LATTICE_NAMES
            if ov.get('lattice', {}).get(nm, {}).get('fix', False)]


def apply_lattice_overrides(uc, crystalSystem, ov):
    """Return the starting unit cell with every overridden lattice value
    placed in, then symmetry ties re-applied. `uc` is not modified."""
    out = [float(x) for x in uc]
    if not ov:
        return np.array(out, dtype=float)
    flags = cryst.free_lattice_flags(crystalSystem)
    for i, nm in enumerate(LATTICE_NAMES):
        entry = ov.get('lattice', {}).get(nm)
        if entry is None or not flags[i]:
            continue
        out[i] = float(entry['value'])
    return np.array(cryst.enforce_symmetry(out, crystalSystem), dtype=float)


def summarise(ov):
    """One-line human-readable summary (used for tooltips / console)."""
    if not has_overrides(ov):
        return "No overrides"
    parts = []
    for nm in LATTICE_NAMES:
        e = ov.get('lattice', {}).get(nm)
        if e is None:
            continue
        tag = 'fixed' if e.get('fix') else 'start'
        parts.append(f"{LATTICE_LABELS[nm]}={e['value']:.5g} ({tag})")
    for key, label, _tip, _scope in FIT_INPUTS:
        if key in ov.get('fit', {}):
            parts.append(f"{label}={ov['fit'][key]:.4g}")
    return "; ".join(parts)


def clear_phase_override_keys(phases, phaseIndices):
    """Remove every per-frame override key from `phases` for the given
    phase indices (None-safe). Call before every fit."""
    if phases is None:
        return
    for pi in phaseIndices:
        for suf in PHASE_OVERRIDE_SUFFIXES:
            phases.pop(f'phase_{pi}{suf}', None)


def shift_frame_rows(rows, removedFrame):
    """Helper for 'Remove pattern': given a list of pressure-guess row
    tuples (frame, P, fix, overrides), drop rows that sit exactly on the
    removed frame and shift every later frame down by one."""
    out = []
    for r in rows:
        fr = int(r[0])
        if fr == removedFrame:
            continue
        if fr > removedFrame:
            fr -= 1
        out.append((fr,) + tuple(r[1:]))
    return out




# ============================================================================
#                                 THE DIALOG
# ============================================================================
class FrameParameterDialog(QtWidgets.QDialog):
    """Popup for ONE pressure-guess row (one phase, one frame).

    Shows all crystallographic information of the phase (symmetry, ambient
    cell, EoS, EoS cell at the row pressure, fitted cell if available,
    reflections) and lets the user set / fix single lattice parameters and
    override fit inputs for this frame only.

    Usage:
        dlg = FrameParameterDialog(parent, phase, frame, pressure, fix,
                                   overrides, WL=..., fitted_uc=...,
                                   global_inputs={...})
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            dlg.result_pressure, dlg.result_fix, dlg.result_overrides
    """

    _COL_PARAM, _COL_AMB, _COL_EOS, _COL_FIT, _COL_VAL, _COL_SET, _COL_FIX = range(7)

    def __init__(self, parent, phase, frame, pressure, fixPressure,
                 overrides, WL=None, twoThetaMin=None, twoThetaMax=None,
                 fittedUc=None, globalInputs=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Frame parameters - {phase.get('name', '')} - frame {frame}")
        self.setModal(True)
        # Fit the screen: full height if there is room, otherwise the body
        # scrolls (see _build_ui).
        scr = (parent.screen() if parent is not None
               else QtWidgets.QApplication.primaryScreen())
        avail = scr.availableGeometry()
        self.resize(1280, max(500, min(1100, avail.height() - 60)))

        self._phase = phase
        self._sym = cryst.normalise_crystal_system(phase.get('crystal_system', 'CUBIC'))
        self._uc0 = np.array(phase.get('unit_cell_0', [1, 1, 1, 90, 90, 90]),
                             dtype=float)
        cc = np.array(phase.get('compression_constants', [0, 0, 4, 0, 0, 0]),
                      dtype=float)
        self._V0, self._K0, self._K0P = (float(cc[0]), float(cc[1]),
                                         float(cc[2]))
        self._HKL = np.array(phase.get('HKL', []), dtype=float).reshape(-1, 4)
        self._frame = int(frame)
        self._WL = WL
        self._ttmin = twoThetaMin
        self._ttmax = twoThetaMax
        self._fitted_uc = (None if fittedUc is None
                           else np.array(fittedUc, dtype=float))
        self._global = dict(globalInputs or {})
        self._ov_in = copy_overrides(overrides)
        self._flags = cryst.free_lattice_flags(self._sym)
        self._busy = False

        # outputs
        self.result_pressure = float(pressure)
        self.result_fix = bool(fixPressure)
        self.result_overrides = copy_overrides(overrides)

        self._build_ui(float(pressure), bool(fixPressure))
        self._load_overrides(self._ov_in)
        self._refresh_all()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self, pressure, fixPressure):
        # The body scrolls so the dialog also fits on small / laptop
        # screens; the summary line and OK/Cancel always stay visible.
        root = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(body)
        outer.setContentsMargins(0, 0, 6, 0)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        # --- header: crystallographic information ---------------------
        gb_info = QtWidgets.QGroupBox("Phase")
        form = QtWidgets.QGridLayout(gb_info)
        uc0 = self._uc0
        info = [
            ("Phase:", self._phase.get('name', '')),
            ("Frame:", str(self._frame)),
            ("Symmetry:", self._sym),
            ("Ambient cell:",
             f"a={uc0[0]:.5f}  b={uc0[1]:.5f}  c={uc0[2]:.5f} \u00c5   "
             f"\u03b1={uc0[3]:.3f}  \u03b2={uc0[4]:.3f}  \u03b3={uc0[5]:.3f}\u00b0"),
            ("EoS (BM3):",
             f"V0={self._V0:.4f} \u00c5\u00b3   K0={self._K0:.3f} GPa   "
             f"K0'={self._K0P:.3f}"),
            ("Wavelength:", "-" if not self._WL else f"{float(self._WL):.5f} \u00c5"),
        ]
        for r, (k, v) in enumerate(info):
            lk = QtWidgets.QLabel(k)
            lv = QtWidgets.QLabel(v)
            lv.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addWidget(lk, r, 0)
            form.addWidget(lv, r, 1)
        outer.addWidget(gb_info)

        # --- pressure ---------------------------------------------------
        gb_p = QtWidgets.QGroupBox("Pressure (this row of the pressure-guess table)")
        gp = QtWidgets.QGridLayout(gb_p)
        self.spn_P = QtWidgets.QDoubleSpinBox()
        self.spn_P.setRange(-50.0, 1000.0)
        self.spn_P.setDecimals(3)
        self.spn_P.setSingleStep(0.1)
        self.spn_P.setSuffix(" GPa")
        self.spn_P.setValue(pressure)
        self.chk_fixP = QtWidgets.QCheckBox(
            "Fix whole lattice at this pressure  (= 'Fix' column)")
        self.chk_fixP.setChecked(fixPressure)
        self.lbl_VP = QtWidgets.QLabel("V(P) = -")
        self.lbl_fix_note = QtWidgets.QLabel("")
        self.lbl_fix_note.setWordWrap(True)
        self.lbl_fix_note.setStyleSheet("color: #f0b429;")
        gp.addWidget(QtWidgets.QLabel("Pressure:"), 0, 0)
        gp.addWidget(self.spn_P, 0, 1)
        gp.addWidget(self.lbl_VP, 0, 2)
        gp.addWidget(self.chk_fixP, 1, 0, 1, 3)
        gp.addWidget(self.lbl_fix_note, 2, 0, 1, 3)
        outer.addWidget(gb_p)

        # --- lattice parameters ------------------------------------------
        gb_lat = QtWidgets.QGroupBox(
            "Lattice parameters for this frame  "
            "(Set = custom starting value, Fix = hold during the fit)")
        vl = QtWidgets.QVBoxLayout(gb_lat)
        self.tbl_lat = QtWidgets.QTableWidget(6, 7)
        self.tbl_lat.setHorizontalHeaderLabels(
            ["", "Ambient", "EoS @ P", "Fitted", "Value", "Set", "Fix"])
        self.tbl_lat.verticalHeader().setVisible(False)
        self.tbl_lat.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.tbl_lat.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        hh = self.tbl_lat.horizontalHeader()
        for col, w in ((0, 34), (1, 95), (2, 95), (3, 95), (4, 140), (5, 42), (6, 42)):
            self.tbl_lat.setColumnWidth(col, w)
        hh.setStretchLastSection(False)

        self._val_spins, self._set_chks, self._fix_chks = {}, {}, {}
        for r, nm in enumerate(LATTICE_NAMES):
            is_len = r < 3
            it = QtWidgets.QTableWidgetItem(LATTICE_LABELS[nm])
            it.setTextAlignment(Qt.AlignCenter)
            self.tbl_lat.setItem(r, self._COL_PARAM, it)
            for col in (self._COL_AMB, self._COL_EOS, self._COL_FIT):
                cell = QtWidgets.QTableWidgetItem("-")
                cell.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_lat.setItem(r, col, cell)

            sp = QtWidgets.QDoubleSpinBox()
            if is_len:
                sp.setRange(0.001, 1000.0); sp.setDecimals(5); sp.setSingleStep(0.001)
            else:
                sp.setRange(1.0, 179.999); sp.setDecimals(4); sp.setSingleStep(0.01)
            sp.setSuffix(LATTICE_UNITS[nm])
            self.tbl_lat.setCellWidget(r, self._COL_VAL, sp)

            cs, cs_box = self._centered_checkbox()
            cf, cf_box = self._centered_checkbox()
            self.tbl_lat.setCellWidget(r, self._COL_SET, cs_box)
            self.tbl_lat.setCellWidget(r, self._COL_FIX, cf_box)

            free = self._flags[r]
            for w in (sp, cs, cf):
                w.setEnabled(free)
            if not free:
                tip = "Tied by symmetry - follows its parent parameter."
                for w in (sp, cs, cf):
                    w.setToolTip(tip)
                self.tbl_lat.item(r, self._COL_PARAM).setToolTip(tip)

            sp.valueChanged.connect(lambda _v, n=nm: self._on_value_edited(n))
            cs.toggled.connect(lambda on, n=nm: self._on_set_toggled(n, on))
            cf.toggled.connect(lambda on, n=nm: self._on_fix_toggled(n, on))
            self._val_spins[nm], self._set_chks[nm], self._fix_chks[nm] = sp, cs, cf

        # Rows must fit the spinbox cell widgets; then give the table the
        # exact height for header + 6 rows so nothing is clipped (the
        # header height is 0 before the dialog is shown -> use sizeHint).
        row_h = max(self.tbl_lat.verticalHeader().defaultSectionSize(),
                    self._val_spins['a'].sizeHint().height() + 4)
        vh = self.tbl_lat.verticalHeader()
        vh.setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        vh.setDefaultSectionSize(row_h)
        for r in range(6):
            self.tbl_lat.setRowHeight(r, row_h)
        self.tbl_lat.setFixedHeight(
            self.tbl_lat.horizontalHeader().sizeHint().height()
            + self.tbl_lat.verticalHeader().length()
            + 2 * self.tbl_lat.frameWidth() + 2)
        self.tbl_lat.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        vl.addWidget(self.tbl_lat)

        brow = QtWidgets.QHBoxLayout()
        self.btn_from_eos = QtWidgets.QPushButton("Values \u2190 EoS @ P")
        self.btn_from_fit = QtWidgets.QPushButton("Values \u2190 Fitted")
        self.btn_from_fit.setEnabled(self._fitted_uc is not None)
        if self._fitted_uc is None:
            self.btn_from_fit.setToolTip("This frame has not been fitted yet.")
        self.btn_clear_lat = QtWidgets.QPushButton("Clear lattice overrides")
        brow.addWidget(self.btn_from_eos)
        brow.addWidget(self.btn_from_fit)
        brow.addStretch(1)
        brow.addWidget(self.btn_clear_lat)
        vl.addLayout(brow)
        outer.addWidget(gb_lat)

        # --- fit inputs -----------------------------------------------------
        gb_fit = QtWidgets.QGroupBox("Fit inputs for this frame  (unticked = use the main window value)")
        gf = QtWidgets.QGridLayout(gb_fit)
        self._fit_chks, self._fit_spins = {}, {}
        ranges = {
            'amp_prefactor': (0.001, 100.0, 3, 0.1),
            'amp_bounds':    (0.01, 1000.0, 3, 0.5),
            'max_shift':     (0.0, 1.0, 3, 0.01),
            'sig_guess':     (0.0001, 10.0, 4, 0.001),
            'sig_bounds':    (0.01, 100.0, 2, 0.1),
        }
        for r, (key, label, tip, scope) in enumerate(FIT_INPUTS):
            chk = QtWidgets.QCheckBox(label)
            chk.setToolTip(tip)
            lo, hi, dec, step = ranges[key]
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setDecimals(dec); sp.setSingleStep(step)
            g = self._global.get(key)
            if g is not None:
                sp.setValue(float(g))
            sp.setEnabled(False)
            sp.setToolTip(tip)
            glob = QtWidgets.QLabel(
                f"main: {g:.4g}" if g is not None else "main: -")
            glob.setStyleSheet("color: #aaaaaa;")
            sc = QtWidgets.QLabel("(whole frame)" if scope == 'frame' else "(this phase)")
            sc.setStyleSheet("color: #aaaaaa;")
            chk.toggled.connect(sp.setEnabled)
            chk.toggled.connect(lambda _on: self._refresh_summary())
            sp.valueChanged.connect(lambda _v: self._refresh_summary())
            gf.addWidget(chk, r, 0)
            gf.addWidget(sp, r, 1)
            gf.addWidget(glob, r, 2)
            gf.addWidget(sc, r, 3)
            self._fit_chks[key], self._fit_spins[key] = chk, sp
        outer.addWidget(gb_fit)

        # --- reflections (read-only) ------------------------------------
        gb_ref = QtWidgets.QGroupBox("Reflections at the starting cell of this frame")
        vr = QtWidgets.QVBoxLayout(gb_ref)
        self.tbl_ref = QtWidgets.QTableWidget(0, 6)
        self.tbl_ref.setHorizontalHeaderLabels(
            ["H", "K", "L", "Int.", "d (\u00c5)", "2\u03b8 (\u00b0)"])
        self.tbl_ref.verticalHeader().setVisible(False)
        self.tbl_ref.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for col, w in ((0, 36), (1, 36), (2, 36), (3, 64), (4, 90)):
            self.tbl_ref.setColumnWidth(col, w)
        self.tbl_ref.horizontalHeader().setStretchLastSection(True)
        self.tbl_ref.setMinimumHeight(150)
        vr.addWidget(self.tbl_ref)
        outer.addWidget(gb_ref, stretch=1)

        # --- summary + buttons -----------------------------------------
        self.lbl_summary = QtWidgets.QLabel("")
        self.lbl_summary.setWordWrap(True)
        root.addWidget(self.lbl_summary)

        btnrow = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset all overrides")
        btnrow.addWidget(self.btn_reset)
        btnrow.addStretch(1)
        self.bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btnrow.addWidget(self.bbox)
        root.addLayout(btnrow)

        # signals
        self.spn_P.valueChanged.connect(lambda _v: self._refresh_all())
        self.chk_fixP.toggled.connect(lambda _on: self._refresh_summary())
        self.btn_from_eos.clicked.connect(self._values_from_eos)
        self.btn_from_fit.clicked.connect(self._values_from_fit)
        self.btn_clear_lat.clicked.connect(self._clear_lattice)
        self.btn_reset.clicked.connect(self._reset_all)
        self.bbox.accepted.connect(self._on_accept)
        self.bbox.rejected.connect(self.reject)

    @staticmethod
    def _centered_checkbox():
        """A QCheckBox centred in a container widget (for table cells).
        Returns (checkbox, container). The caller must keep / reparent the
        container (setCellWidget) - otherwise Python would garbage-collect
        it together with the checkbox."""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setAlignment(Qt.AlignCenter)
        chk = QtWidgets.QCheckBox(w)
        lay.addWidget(chk)
        return chk, w

    # ------------------------------------------------------------------
    # state <-> widgets
    # ------------------------------------------------------------------
    def _eos_uc(self):
        return eos.safe_uc_at_p(self._uc0, self.spn_P.value(), self._K0, self._K0P)

    def _load_overrides(self, ov):
        self._busy = True
        try:
            eosCell = self._eos_uc()
            for i, nm in enumerate(LATTICE_NAMES):
                e = ov.get('lattice', {}).get(nm)
                if e is not None and self._flags[i]:
                    self._val_spins[nm].setValue(float(e['value']))
                    self._set_chks[nm].setChecked(True)
                    self._fix_chks[nm].setChecked(bool(e.get('fix', False)))
                else:
                    self._val_spins[nm].setValue(float(eosCell[i]))
                    self._set_chks[nm].setChecked(False)
                    self._fix_chks[nm].setChecked(False)
            for key, *_ in FIT_INPUTS:
                if key in ov.get('fit', {}):
                    self._fit_chks[key].setChecked(True)
                    self._fit_spins[key].setValue(float(ov['fit'][key]))
                else:
                    self._fit_chks[key].setChecked(False)
                    g = self._global.get(key)
                    if g is not None:
                        self._fit_spins[key].setValue(float(g))
        finally:
            self._busy = False

    def _collect_overrides(self):
        ov = empty_overrides()
        for i, nm in enumerate(LATTICE_NAMES):
            if not self._flags[i]:
                continue
            if self._set_chks[nm].isChecked() or self._fix_chks[nm].isChecked():
                ov['lattice'][nm] = {'value': float(self._val_spins[nm].value()),
                                     'fix': bool(self._fix_chks[nm].isChecked())}
        for key, *_ in FIT_INPUTS:
            if self._fit_chks[key].isChecked():
                ov['fit'][key] = float(self._fit_spins[key].value())
        return ov

    def _start_uc(self):
        """Starting cell for this frame = EoS cell at P with the set
        values placed in (and symmetry ties applied)."""
        return apply_lattice_overrides(self._eos_uc(), self._sym,
                                       self._collect_overrides())

    # ------------------------------------------------------------------
    # handlers
    # ------------------------------------------------------------------
    def _on_value_edited(self, nm):
        if self._busy:
            return
        # Editing a value means the user wants it as the starting value.
        if not self._set_chks[nm].isChecked():
            self._set_chks[nm].setChecked(True)
        self._refresh_symmetry_followers()
        self._refresh_reflections()
        self._refresh_summary()

    def _on_set_toggled(self, nm, on):
        if self._busy:
            return
        if not on:
            # Un-setting also un-fixes and restores the EoS value
            self._busy = True
            try:
                self._fix_chks[nm].setChecked(False)
                i = LATTICE_NAMES.index(nm)
                self._val_spins[nm].setValue(float(self._eos_uc()[i]))
            finally:
                self._busy = False
        self._refresh_symmetry_followers()
        self._refresh_reflections()
        self._refresh_summary()

    def _on_fix_toggled(self, nm, on):
        if self._busy:
            return
        if on and not self._set_chks[nm].isChecked():
            self._busy = True
            try:
                self._set_chks[nm].setChecked(True)
            finally:
                self._busy = False
        self._refresh_summary()

    def _values_from_eos(self):
        eosCell = self._eos_uc()
        self._busy = True
        try:
            for i, nm in enumerate(LATTICE_NAMES):
                if self._flags[i] and self._set_chks[nm].isChecked():
                    self._val_spins[nm].setValue(float(eosCell[i]))
        finally:
            self._busy = False
        self._refresh_all()

    def _values_from_fit(self):
        if self._fitted_uc is None:
            return
        self._busy = True
        try:
            for i, nm in enumerate(LATTICE_NAMES):
                if self._flags[i]:
                    self._val_spins[nm].setValue(float(self._fitted_uc[i]))
                    self._set_chks[nm].setChecked(True)
        finally:
            self._busy = False
        self._refresh_all()

    def _clear_lattice(self):
        ov = self._collect_overrides()
        ov['lattice'] = {}
        self._load_overrides(ov)
        self._refresh_all()

    def _reset_all(self):
        self._load_overrides(empty_overrides())
        self._refresh_all()

    def _on_accept(self):
        self.result_pressure = float(self.spn_P.value())
        self.result_fix = bool(self.chk_fixP.isChecked())
        self.result_overrides = self._collect_overrides()
        self.accept()

    # ------------------------------------------------------------------
    # refresh helpers
    # ------------------------------------------------------------------
    def _refresh_all(self):
        self._refresh_cell_columns()
        self._refresh_symmetry_followers()
        self._refresh_reflections()
        self._refresh_summary()

    def _refresh_cell_columns(self):
        eosCell = self._eos_uc()
        try:
            VP = cryst.unit_cell_volume(eosCell)
            VP = float('nan') if VP is None else VP
            self.lbl_VP.setText(f"V(P) = {VP:.4f} \u00c5\u00b3   "
                                f"(V/V0 = {VP / self._V0:.4f})"
                                if self._V0 else f"V(P) = {VP:.4f} \u00c5\u00b3")
        except Exception:
            self.lbl_VP.setText("V(P) = -")
        self._busy = True
        try:
            for i, nm in enumerate(LATTICE_NAMES):
                fmt = "{:.5f}" if i < 3 else "{:.4f}"
                self.tbl_lat.item(i, self._COL_AMB).setText(fmt.format(self._uc0[i]))
                self.tbl_lat.item(i, self._COL_EOS).setText(fmt.format(eosCell[i]))
                self.tbl_lat.item(i, self._COL_FIT).setText(
                    "-" if self._fitted_uc is None else fmt.format(self._fitted_uc[i]))
                # Rows not "Set" track the EoS value as P changes
                if self._flags[i] and not self._set_chks[nm].isChecked():
                    self._val_spins[nm].setValue(float(eosCell[i]))
        finally:
            self._busy = False

    def _refresh_symmetry_followers(self):
        """Dependent (tied) rows show the value they will actually get."""
        uc = self._start_uc()
        self._busy = True
        try:
            for i, nm in enumerate(LATTICE_NAMES):
                if not self._flags[i]:
                    self._val_spins[nm].setValue(float(uc[i]))
        finally:
            self._busy = False

    def _refresh_reflections(self):
        uc = self._start_uc()
        self.tbl_ref.setRowCount(0)
        for row in self._HKL:
            h, k, l, inten = (float(x) for x in row[:4])
            d, tt = cryst.d_and_twotheta(uc, self._sym, h, k, l, self._WL)
            r = self.tbl_ref.rowCount()
            self.tbl_ref.insertRow(r)
            vals = [f"{int(round(h))}", f"{int(round(k))}", f"{int(round(l))}",
                    f"{inten:.1f}",
                    "-" if d is None else f"{d:.5f}",
                    "-" if tt is None else f"{tt:.4f}"]
            outside = (tt is not None and self._ttmin is not None
                       and self._ttmax is not None
                       and not (self._ttmin <= tt <= self._ttmax))
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignCenter)
                if outside:
                    it.setForeground(Qt.gray)
                    it.setToolTip("Outside the current 2\u03b8 fit window")
                self.tbl_ref.setItem(r, c, it)

    def _refresh_summary(self):
        if self._busy:
            return
        ov = self._collect_overrides()
        fixed = fixed_lattice_names(ov)
        if fixed and self.chk_fixP.isChecked():
            self.lbl_fix_note.setText(
                "Individual parameters are fixed below, so the whole-lattice "
                "'Fix' is IGNORED for this frame: only "
                + ", ".join(LATTICE_LABELS[n] for n in fixed)
                + " will be held, everything else refines.")
        else:
            self.lbl_fix_note.setText("")
        self.lbl_summary.setText("Overrides for this frame: " + summarise(ov))
