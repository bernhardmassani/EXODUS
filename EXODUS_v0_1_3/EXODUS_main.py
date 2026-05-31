# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 23:03:56 2026

@author: Dr Bernhard Massani
"""

import sys
import os
import time
import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QColorDialog, QPushButton, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QRectF
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon, QColor
import pyqtgraph as pg
from pyqtgraph import ColorMap
from pyqtgraph import ImageView, AxisItem

import EXODUS_core as exc
from EXODUS_GUI_ui import Ui_MainWindow

from functools import partial

import EoS_toolbox as EoS
import BatchFit_toolbox as tb
import CIF_to_JCPDS_toolbox as cif2jcpds

# Force PyInstaller to bundle pymatgen even though the real import is lazy
# inside CIF_to_JCPDS_toolbox._import_pymatgen(). Do not remove.
import pymatgen
import pymatgen.core
import pymatgen.analysis.diffraction.xrd  # noqa: F401


# ============================================================================
#                       USER-TWEAKABLE SETTINGS
# ============================================================================
# Live fit-plot updates during batch fitting. Set to N>0 to redraw the 1D
# fit plot every N frames (1 = every frame, the smoothest visual feedback).
# Set to 0 to disable live updates entirely (matches the old behaviour;
# fastest batch throughput at the cost of no live preview).
FIT_PLOT_LIVE_EVERY_N = 0

# Easter egg: comment out the next line (or set False) to disable the
# RSW.png popup when the user clicks the log-scale button on the 2D plot.
ENABLE_LOG_EASTER_EGG = False

# Default intensity scaling on the 2D plot. 'log' is usually the right
# default for diffraction data because the dynamic range is huge and the
# LUT histogram bunches everything into a tiny strip otherwise. Acceptable
# values: 'linear', 'log', 'sqrt'.
DEFAULT_2D_SCALE = 'linear'

# Color-map cycle for the 2D plot (see pyqtgraph gradient presets).
COLORMAP_CYCLE = ['grey', 'thermal', 'flame', 'inferno', 'viridis', 
                  'plasma', 'magma',  'cyclic']

# Parallelism for the background-subtraction pass on the 2D-stack build.
# 'serial'    - single-threaded (the old behaviour; safest, no overhead)
# 'threads'   - ThreadPoolExecutor; the default. Real-time per-file
#               progress reporting, no pickling overhead, no pool-
#               startup cost, works reliably on Windows / frozen builds.
#               ALS releases the GIL during its SciPy linalg calls so
#               threads still scale well across cores.
# 'processes' - ProcessPoolExecutor; theoretically fastest on big
#               batches but in practice the per-file progress callback
#               is bursty (results come back in waves of N_workers at
#               a time so the bar can appear to jump) and process
#               startup / pickling adds noticeable latency on Windows
#               and frozen builds. Switch to this only if you really
#               need the last bit of speed and don't mind a chunky
#               progress bar.
BG_PARALLEL_MODE = 'threads'

# Don't bother with a parallel pool unless we're processing at least this
# many files. Below this, the pool overhead exceeds the speedup.
BG_PARALLEL_MIN_FILES = 50


# ============================================================================
#                          JCPDS EDIT DIALOG
# ============================================================================
# Dioptas-inspired editor for a single JCPDS phase. Shows lattice parameters,
# EoS constants, and the (h k l, intensity, d) reflection table; edits are
# previewed live in the dialog and only written to disk on Save / Save As /
# OK. Cancel discards everything (the in-memory phases dict is left
# untouched).
# ============================================================================
class JCPDSEditDialog(QtWidgets.QDialog):
    """Editor for a single JCPDS phase, modelled on the Dioptas JCPDS editor.

    Use as:
        dlg = JCPDSEditDialog(parent, phase_dict)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            # dlg.result_phase, dlg.result_path, dlg.dirty are now populated
            ...

    Where ``phase_dict`` is a dict with the following keys (all required):
        name, file_path, crystal_system, unit_cell_0,
        compression_constants ([V0, K0, K0P, alphaT, DK0DT, DK0PDT]),
        HKL  (Nx4 numpy array of [h, k, l, intensity])
    """

    SYMMETRIES = [
        "CUBIC", "TETRAGONAL", "HEXAGONAL", "RHOMBOHEDRAL",
        "ORTHOROMBIC", "MONOCLINIC", "TRICLINIC",
    ]

    # Which lattice fields the user may edit for each symmetry. The others
    # are kept in sync automatically (or hard-coded - e.g. alpha=gamma=90 for
    # monoclinic). Order matches [a, b, c, alpha, beta, gamma].
    _ENABLED_LATTICE = {
        "CUBIC":         (True,  False, False, False, False, False),
        "TETRAGONAL":    (True,  False, True,  False, False, False),
        "HEXAGONAL":     (True,  False, True,  False, False, False),
        "RHOMBOHEDRAL":  (True,  False, False, True,  False, False),
        "ORTHOROMBIC":   (True,  True,  True,  False, False, False),
        "MONOCLINIC":    (True,  True,  True,  False, True,  False),
        "TRICLINIC":     (True,  True,  True,  True,  True,  True),
    }

    def __init__(self, parent, phase):
        super().__init__(parent)
        self.setWindowTitle(f"JCPDS Editor - {phase.get('name', '')}")
        self.setModal(True)
        self.resize(620, 720)

        # Cache the original phase data for the Reload button.
        self._original = self._copy_phase(phase)
        self._phase = self._copy_phase(phase)
        self._suppress_updates = False  # guard against signal storms

        # Outputs the caller will read:
        self.dirty = False              # True if anything changed
        self.result_phase = None        # populated on Accept
        self.result_path = None         # path the file was saved to (or None)

        self._build_ui()
        self._populate_from_phase(self._phase)
        self._connect_signals()
        self._refresh_lattice_enable()
        self._refresh_volume()
        self._refresh_reflection_table_d()

    # ---- internal helpers -------------------------------------------------
    @staticmethod
    def _copy_phase(phase):
        """Deep-ish copy of the phase dict so edits don't leak before Save."""
        return {
            "name":                  phase.get("name", ""),
            "file_path":             phase.get("file_path", ""),
            "crystal_system":        str(phase.get("crystal_system", "CUBIC")),
            "unit_cell_0":           np.array(phase.get("unit_cell_0",
                                       [1.0, 1.0, 1.0, 90.0, 90.0, 90.0]),
                                              dtype=float),
            "compression_constants": np.array(phase.get("compression_constants",
                                       [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),
                                              dtype=float),
            "HKL":                   np.array(phase.get("HKL", []), dtype=float).reshape(-1, 4),
        }

    # ---- UI construction --------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)

        # --- Header: filename + comment ---
        hdr = QtWidgets.QFormLayout()
        self.lbl_filename = QtWidgets.QLabel("")
        self.lbl_filename.setStyleSheet("color: #666;")
        hdr.addRow("File:", self.lbl_filename)
        self.cmb_symmetry = QtWidgets.QComboBox()
        self.cmb_symmetry.addItems(self.SYMMETRIES)
        hdr.addRow("Symmetry:", self.cmb_symmetry)
        outer.addLayout(hdr)

        # --- Lattice parameters group ---
        gb_lat = QtWidgets.QGroupBox("Lattice parameters (ambient, P=0, T=0)")
        grid = QtWidgets.QGridLayout(gb_lat)
        self.spn_a = self._make_lat_spin(0.001, 1000.0, 5, " A")
        self.spn_b = self._make_lat_spin(0.001, 1000.0, 5, " A")
        self.spn_c = self._make_lat_spin(0.001, 1000.0, 5, " A")
        self.spn_alpha = self._make_lat_spin(0.001, 179.999, 4, " deg")
        self.spn_beta = self._make_lat_spin(0.001, 179.999, 4, " deg")
        self.spn_gamma = self._make_lat_spin(0.001, 179.999, 4, " deg")
        grid.addWidget(QtWidgets.QLabel("a0"), 0, 0); grid.addWidget(self.spn_a, 0, 1)
        grid.addWidget(QtWidgets.QLabel("b0"), 0, 2); grid.addWidget(self.spn_b, 0, 3)
        grid.addWidget(QtWidgets.QLabel("c0"), 0, 4); grid.addWidget(self.spn_c, 0, 5)
        grid.addWidget(QtWidgets.QLabel("alpha0"), 1, 0); grid.addWidget(self.spn_alpha, 1, 1)
        grid.addWidget(QtWidgets.QLabel("beta0"),  1, 2); grid.addWidget(self.spn_beta,  1, 3)
        grid.addWidget(QtWidgets.QLabel("gamma0"), 1, 4); grid.addWidget(self.spn_gamma, 1, 5)
        # Volume readout
        self.lbl_volume = QtWidgets.QLabel("V0 = -")
        f = self.lbl_volume.font(); f.setBold(True); self.lbl_volume.setFont(f)
        grid.addWidget(self.lbl_volume, 2, 0, 1, 6)
        outer.addWidget(gb_lat)

        # --- EoS group ---
        gb_eos = QtWidgets.QGroupBox("Equation of state")
        eos_grid = QtWidgets.QGridLayout(gb_eos)
        self.spn_K0    = self._make_eos_spin(0.0, 1e6, 4, " GPa")
        self.spn_K0P   = self._make_eos_spin(-100.0, 100.0, 4, "")
        self.spn_alphaT = self._make_eos_spin(-1.0, 1.0, 8, " 1/K")
        self.spn_dK0dT  = self._make_eos_spin(-100.0, 100.0, 6, " GPa/K")
        self.spn_dK0PdT = self._make_eos_spin(-100.0, 100.0, 8, " 1/K")
        eos_grid.addWidget(QtWidgets.QLabel("K0 (B0)"), 0, 0); eos_grid.addWidget(self.spn_K0, 0, 1)
        eos_grid.addWidget(QtWidgets.QLabel("K0P"),     0, 2); eos_grid.addWidget(self.spn_K0P, 0, 3)
        eos_grid.addWidget(QtWidgets.QLabel("alphaT"),  1, 0); eos_grid.addWidget(self.spn_alphaT, 1, 1)
        eos_grid.addWidget(QtWidgets.QLabel("dK0/dT"),  1, 2); eos_grid.addWidget(self.spn_dK0dT, 1, 3)
        eos_grid.addWidget(QtWidgets.QLabel("dK0P/dT"), 1, 4); eos_grid.addWidget(self.spn_dK0PdT, 1, 5)
        outer.addWidget(gb_eos)

        # --- Reflection table ---
        gb_ref = QtWidgets.QGroupBox("Reflections (h, k, l, intensity editable; d auto)")
        v = QtWidgets.QVBoxLayout(gb_ref)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["h", "k", "l", "I", "d (A)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        # Make the d column read-only by intercepting edits in itemChanged.
        v.addWidget(self.table)
        # Row add / remove
        rowbtns = QtWidgets.QHBoxLayout()
        self.btn_add_row = QtWidgets.QPushButton("Add reflection")
        self.btn_del_row = QtWidgets.QPushButton("Remove selected")
        rowbtns.addWidget(self.btn_add_row); rowbtns.addWidget(self.btn_del_row)
        rowbtns.addStretch(1)
        v.addLayout(rowbtns)
        outer.addWidget(gb_ref, stretch=1)

        # --- Bottom buttons ---
        btnrow = QtWidgets.QHBoxLayout()
        self.btn_reload  = QtWidgets.QPushButton("Reload File")
        self.btn_save    = QtWidgets.QPushButton("Save")
        self.btn_save_as = QtWidgets.QPushButton("Save As...")
        btnrow.addWidget(self.btn_reload)
        btnrow.addStretch(1)
        btnrow.addWidget(self.btn_save)
        btnrow.addWidget(self.btn_save_as)
        # OK / Cancel
        self.bbox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btnrow.addWidget(self.bbox)
        outer.addLayout(btnrow)

    @staticmethod
    def _make_lat_spin(lo, hi, decimals, suffix):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setSingleStep(0.001)
        s.setMinimumWidth(120)
        return s

    @staticmethod
    def _make_eos_spin(lo, hi, decimals, suffix):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(decimals)
        s.setSuffix(suffix)
        s.setMinimumWidth(140)
        return s

    # ---- populate / read --------------------------------------------------
    def _populate_from_phase(self, phase):
        self._suppress_updates = True
        try:
            self.lbl_filename.setText(phase.get("file_path") or phase.get("name", ""))
            sym = phase["crystal_system"]
            idx = self.cmb_symmetry.findText(sym)
            if idx < 0:
                idx = 0  # fallback
            self.cmb_symmetry.setCurrentIndex(idx)

            uc = phase["unit_cell_0"]
            self.spn_a.setValue(float(uc[0]))
            self.spn_b.setValue(float(uc[1]))
            self.spn_c.setValue(float(uc[2]))
            self.spn_alpha.setValue(float(uc[3]))
            self.spn_beta.setValue(float(uc[4]))
            self.spn_gamma.setValue(float(uc[5]))

            cc = phase["compression_constants"]
            # cc layout: [V0, K0, K0P, alphaT, DK0DT, DK0PDT]
            self.spn_K0.setValue(float(cc[1]))
            self.spn_K0P.setValue(float(cc[2]))
            self.spn_alphaT.setValue(float(cc[3]))
            self.spn_dK0dT.setValue(float(cc[4]))
            self.spn_dK0PdT.setValue(float(cc[5]))

            self._populate_reflection_table(phase["HKL"])
        finally:
            self._suppress_updates = False

    def _populate_reflection_table(self, hkl):
        self.table.setRowCount(0)
        if hkl is None:
            return
        for row in hkl:
            self._append_reflection_row(row[0], row[1], row[2], row[3])

    def _append_reflection_row(self, h, k, l, inten, d=None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for col, val in enumerate([h, k, l, inten]):
            it = QtWidgets.QTableWidgetItem(f"{int(round(float(val)))}" if col < 3
                                            else f"{float(val):.3f}")
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.table.setItem(r, col, it)
        # d cell: read-only
        d_item = QtWidgets.QTableWidgetItem("" if d is None else f"{d:.5f}")
        d_item.setFlags(d_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(r, 4, d_item)

    def _read_unit_cell(self):
        return np.array([
            self.spn_a.value(), self.spn_b.value(), self.spn_c.value(),
            self.spn_alpha.value(), self.spn_beta.value(), self.spn_gamma.value(),
        ], dtype=float)

    def _read_eos(self):
        # Returns (K0, K0P, alphaT, dK0dT, dK0PdT)
        return (self.spn_K0.value(), self.spn_K0P.value(),
                self.spn_alphaT.value(), self.spn_dK0dT.value(),
                self.spn_dK0PdT.value())

    def _read_HKL(self):
        n = self.table.rowCount()
        out = np.zeros((n, 4), dtype=float)
        for r in range(n):
            for c in range(4):
                it = self.table.item(r, c)
                try:
                    out[r, c] = float(it.text()) if it is not None else 0.0
                except ValueError:
                    out[r, c] = 0.0
        return out

    # ---- signals ----------------------------------------------------------
    def _connect_signals(self):
        self.cmb_symmetry.currentIndexChanged.connect(self._on_symmetry_changed)
        for spin in (self.spn_a, self.spn_b, self.spn_c,
                     self.spn_alpha, self.spn_beta, self.spn_gamma):
            spin.valueChanged.connect(self._on_lattice_changed)
        self.table.itemChanged.connect(self._on_table_item_changed)

        self.btn_add_row.clicked.connect(self._on_add_row)
        self.btn_del_row.clicked.connect(self._on_remove_row)

        self.btn_reload.clicked.connect(self._on_reload)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as.clicked.connect(self._on_save_as)
        self.bbox.accepted.connect(self._on_accept)
        self.bbox.rejected.connect(self.reject)

    # --- handlers ---
    def _on_symmetry_changed(self, _idx):
        if self._suppress_updates:
            return
        self.dirty = True
        self._refresh_lattice_enable()
        # When switching, force constrained values into a consistent state
        # (e.g. moving to CUBIC sets b=c=a, alpha=beta=gamma=90).
        self._enforce_symmetry_constraints()
        self._refresh_volume()
        self._refresh_reflection_table_d()

    def _on_lattice_changed(self, *_):
        if self._suppress_updates:
            return
        self.dirty = True
        self._enforce_symmetry_constraints()
        self._refresh_volume()
        self._refresh_reflection_table_d()

    def _on_table_item_changed(self, item):
        if self._suppress_updates:
            return
        self.dirty = True
        # Recompute d-spacing for the changed row if h/k/l changed.
        if item.column() in (0, 1, 2):
            self._refresh_reflection_row_d(item.row())

    def _on_add_row(self):
        self._suppress_updates = True
        try:
            self._append_reflection_row(0, 0, 1, 100.0)
        finally:
            self._suppress_updates = False
        self._refresh_reflection_row_d(self.table.rowCount() - 1)
        self.dirty = True

    def _on_remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        if not rows:
            return
        for r in rows:
            self.table.removeRow(r)
        self.dirty = True

    def _on_reload(self):
        if QMessageBox.question(self, "Reload",
                                "Discard all unsaved changes and reload from disk?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        path = self._original.get("file_path")
        if path and os.path.isfile(path):
            try:
                fresh = exc.load_JCPDS([path])
                self._original = self._phase_dict_from_loaded(fresh, 0, path)
                self._phase = self._copy_phase(self._original)
            except Exception as e:
                QMessageBox.warning(self, "Reload failed", str(e))
                return
        else:
            # Fall back to the snapshot we already have
            self._phase = self._copy_phase(self._original)
        self._populate_from_phase(self._phase)
        self._refresh_lattice_enable()
        self._refresh_volume()
        self._refresh_reflection_table_d()
        self.dirty = False

    def _on_save(self):
        path = self._original.get("file_path")
        if not path:
            return self._on_save_as()
        return self._save_to(path)

    def _on_save_as(self):
        start = self._original.get("file_path") or ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JCPDS as", start, "JCPDS files (*.jcpds);;All files (*)")
        if not path:
            return False
        if not path.lower().endswith(".jcpds"):
            path = path + ".jcpds"
        return self._save_to(path)

    def _save_to(self, path):
        try:
            cif2jcpds.write_jcpds(
                jcpds_path=path,
                unit_cell=self._read_unit_cell(),
                HKL=self._read_HKL(),
                symmetry=self.cmb_symmetry.currentText(),
                K0=self.spn_K0.value(),
                K0P=self.spn_K0P.value(),
                alphaT=self.spn_alphaT.value(),
                DK0DT=self.spn_dK0dT.value(),
                DK0PDT=self.spn_dK0PdT.value(),
                comment=f"Edited in EXODUS JCPDS editor",
            )
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return False
        self.result_path = path
        self.dirty = False
        # Refresh _original so subsequent Reload reverts to this saved state.
        self._original = self._copy_phase({
            "name": os.path.splitext(os.path.basename(path))[0],
            "file_path": path,
            "crystal_system": self.cmb_symmetry.currentText(),
            "unit_cell_0": self._read_unit_cell(),
            "compression_constants": np.array([
                EoS.unitCellVolume(self._read_unit_cell()),
                self.spn_K0.value(), self.spn_K0P.value(),
                self.spn_alphaT.value(), self.spn_dK0dT.value(),
                self.spn_dK0PdT.value(),
            ]),
            "HKL": self._read_HKL(),
        })
        return True

    def _on_accept(self):
        # If anything was changed and the file hasn't yet been saved, prompt.
        if self.dirty:
            ret = QMessageBox.question(
                self, "Save changes",
                "There are unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save)
            if ret == QMessageBox.Cancel:
                return
            if ret == QMessageBox.Save:
                if not self._on_save():
                    return  # save was cancelled / failed; keep dialog open
        # Build the result phase dict the caller will use to update its state.
        uc = self._read_unit_cell()
        cc = np.array([
            EoS.unitCellVolume(uc),
            self.spn_K0.value(), self.spn_K0P.value(),
            self.spn_alphaT.value(), self.spn_dK0dT.value(),
            self.spn_dK0PdT.value(),
        ], dtype=float)
        self.result_phase = {
            "name": self._original.get("name", ""),
            "file_path": self.result_path or self._original.get("file_path", ""),
            "crystal_system": self.cmb_symmetry.currentText(),
            "unit_cell_0": uc,
            "compression_constants": cc,
            "HKL": self._read_HKL(),
        }
        self.accept()

    # ---- live-update helpers ----------------------------------------------
    def _refresh_lattice_enable(self):
        sym = self.cmb_symmetry.currentText()
        flags = self._ENABLED_LATTICE.get(sym, (True,) * 6)
        for spin, en in zip(
            (self.spn_a, self.spn_b, self.spn_c,
             self.spn_alpha, self.spn_beta, self.spn_gamma), flags):
            spin.setEnabled(en)

    def _enforce_symmetry_constraints(self):
        """Force fields locked by the chosen symmetry to their dependent
        values (e.g. b=a for tetragonal). Runs without re-emitting signals."""
        sym = self.cmb_symmetry.currentText()
        a = self.spn_a.value()
        c_val = self.spn_c.value()
        alpha = self.spn_alpha.value()
        self._suppress_updates = True
        try:
            if sym == "CUBIC":
                self.spn_b.setValue(a); self.spn_c.setValue(a)
                self.spn_alpha.setValue(90.0); self.spn_beta.setValue(90.0); self.spn_gamma.setValue(90.0)
            elif sym in ("TETRAGONAL", "HEXAGONAL"):
                self.spn_b.setValue(a)
                self.spn_alpha.setValue(90.0)
                self.spn_beta.setValue(90.0)
                self.spn_gamma.setValue(120.0 if sym == "HEXAGONAL" else 90.0)
            elif sym == "RHOMBOHEDRAL":
                self.spn_b.setValue(a); self.spn_c.setValue(a)
                self.spn_beta.setValue(alpha); self.spn_gamma.setValue(alpha)
            elif sym == "ORTHOROMBIC":
                self.spn_alpha.setValue(90.0); self.spn_beta.setValue(90.0); self.spn_gamma.setValue(90.0)
            elif sym == "MONOCLINIC":
                self.spn_alpha.setValue(90.0); self.spn_gamma.setValue(90.0)
            # TRICLINIC: no constraints
        finally:
            self._suppress_updates = False

    def _refresh_volume(self):
        try:
            uc = self._read_unit_cell()
            vol = EoS.unitCellVolume(uc)
            self.lbl_volume.setText(f"V0 = {vol:.4f} A^3")
        except Exception:
            self.lbl_volume.setText("V0 = -")

    def _refresh_reflection_table_d(self):
        for r in range(self.table.rowCount()):
            self._refresh_reflection_row_d(r)

    def _refresh_reflection_row_d(self, row):
        if row < 0 or row >= self.table.rowCount():
            return
        try:
            h = float(self.table.item(row, 0).text())
            k = float(self.table.item(row, 1).text())
            l = float(self.table.item(row, 2).text())
        except (ValueError, AttributeError):
            return
        a, b, c = self.spn_a.value(), self.spn_b.value(), self.spn_c.value()
        alpha = self.spn_alpha.value(); beta = self.spn_beta.value(); gamma = self.spn_gamma.value()
        sym = self.cmb_symmetry.currentText()
        d = cif2jcpds._d_spacing(a, b, c, alpha, beta, gamma, h, k, l, sym)
        self._suppress_updates = True
        try:
            d_item = self.table.item(row, 4)
            if d_item is None:
                d_item = QtWidgets.QTableWidgetItem()
                d_item.setFlags(d_item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, 4, d_item)
            d_item.setText("-" if (d is None or not np.isfinite(d)) else f"{d:.5f}")
        finally:
            self._suppress_updates = False

    @staticmethod
    def _phase_dict_from_loaded(phases, idx, file_path):
        """Convert the dict structure returned by load_JCPDS into the flat
        single-phase layout the dialog uses internally."""
        return {
            "name": phases[f"phase_{idx}_name"],
            "file_path": file_path,
            "crystal_system": phases[f"phase_{idx}_crystal_system"],
            "unit_cell_0": np.array(phases[f"phase_{idx}_unit_cell_0"], dtype=float),
            "compression_constants": np.array(
                phases[f"phase_{idx}_compression_constants"], dtype=float),
            "HKL": np.array(phases[f"phase_{idx}_HKL"], dtype=float).reshape(-1, 4),
        }


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Initialise Data Manager
        self.data_manager = DataManager()
        
        # General settings for pg
        pg.setConfigOption('background', (61, 61, 61)) # global
        pg.setConfigOption('foreground', 'w') # global
        
        # Display raw data
        self.plot_raw = pg.PlotWidget()
        self.ui.verticalLayout_rawPattern.addWidget(self.plot_raw)
        self.plot_raw.getPlotItem().getViewBox().setDefaultPadding(0.1)
        self.plot_raw.getPlotItem().layout.setContentsMargins(10, 10, 10, 10)
        self.plot_raw.setLabel('left', 'Intensity')
        self.plot_raw.setLabel('bottom', '2-Theta (deg)')
        twoThetaMin = self.ui.doubleSpinBox_twoThetaMin.value()
        twoThetaMax = self.ui.doubleSpinBox_twoThetaMax.value()
        self.plot_raw.setXRange(twoThetaMin, twoThetaMax, padding=0)
        self.plot_raw.enableAutoRange(axis='y')
        self.plot_raw.disableAutoRange(axis='x')
        
        # Display processed data (second plot)
        self.plot_processed = pg.PlotWidget()
        self.ui.verticalLayout_processedPattern.addWidget(self.plot_processed)
        self.plot_processed.getPlotItem().getViewBox().setDefaultPadding(0.1)
        self.plot_processed.getPlotItem().layout.setContentsMargins(10, 10, 10, 10)
        self.plot_processed.setLabel('left', 'Intensity')
        self.plot_processed.setLabel('bottom', '2-Theta (deg)')
        twoThetaMin = self.ui.doubleSpinBox_twoThetaMin.value()
        twoThetaMax = self.ui.doubleSpinBox_twoThetaMax.value()
        self.plot_processed.setXRange(twoThetaMin, twoThetaMax, padding=0)
        self.plot_processed.enableAutoRange(axis='y')
        self.plot_processed.disableAutoRange(axis='x')
        
        # Poni button
        self.poni_path = None  # Empty variable
        self.ui.pushButton_loadPoni.clicked.connect(self.load_poni_path)

        # ---------------------------------------------------------------
        # ITEM 4: Enable draw-a-rectangle-to-zoom on all plots.
        # Left-click drag = rectangle zoom.  Right-click still opens the
        # context menu; double-click / right-drag auto-range as usual.
        # ---------------------------------------------------------------
        self._enable_rect_zoom(self.plot_raw)
        self._enable_rect_zoom(self.plot_processed)
        self._enable_rect_zoom(self.ui.plotWidget_2D)
        
        # Data button and spinBox
        self.file_paths = []
        self.ui.pushButton_loadFiles.clicked.connect(self.load_data_paths)
        self.ui.listWidget_fileList.currentRowChanged.connect(self.selection_changed)
        self.ui.spinBox_PatternNumber.valueChanged.connect(self.spinbox_changed)
        self.ui.spinBox_PatternNumber.valueChanged.connect(self.selection_changed)
        
        # Connects Fitting buttons
        self.ui.pushButton_fitJCPDS_single.clicked.connect(self.LB_fit_single)
        self.ui.pushButton_save_single.clicked.connect(self.save_single_data)

        # Save the 2D plot as TIFF/PNG
        self.ui.pushButton_save2D.clicked.connect(self.save_2D_plot)

        # Sequetial
        self.ui.pushButton_fitAll.clicked.connect(self.run_fit_all)
        # Cancel-fitting is handled by the progress-dialog's own button.
        self.ui.pushButton_saveAll.clicked.connect(self.save_all_results)  # CONNECTED
        self.fitModeGroup = QtWidgets.QButtonGroup(self)
        self.fitModeGroup.setExclusive(True)
        self.fitModeGroup.addButton(self.ui.checkBox_sequential)
        self.fitModeGroup.addButton(self.ui.checkBox_pressureWalk)
        self.fitModeGroup.addButton(self.ui.checkBox_poressureGuess)
        # self.fitModeGroup.addButton(self.ui.checkBox_sequentialplus)

        # Sigma mode checkboxes - mutually exclusive group
        # Maps to sigma_mode strings: fixed, caglioti, separate, per_phase
        self.sigmaModeGroup = QtWidgets.QButtonGroup(self)
        self.sigmaModeGroup.setExclusive(True)
        self.sigmaModeGroup.addButton(self.ui.checkBox_fixSigma_2)  # "Fix Sigma"       -> 'fixed'
        self.sigmaModeGroup.addButton(self.ui.checkBox)              # "Fix sigma 1/cos" -> 'caglioti'
        self.sigmaModeGroup.addButton(self.ui.checkBox_2)            # "Vary Sigma"      -> 'separate'
        self.sigmaModeGroup.addButton(self.ui.checkBox_3)            # "Vary Sigma/phase"-> 'per_phase'
        self.ui.checkBox.setChecked(True)   # caglioti on by default

        # Set sensible defaults for fitting parameter spinboxes
        # 'Amplitude' row (self.doubleSpinBox) = ampPrefactor:
        #     ampGuess = max(data) * prefactor. 1.0 = 100% of data peak.
        # 'Amp. Bounds' row (self.doubleSpinBox_fitAmpGuess) = upper-bound
        #     multiplier on amp_init (mirrors Sigma Bounds).
        self.ui.doubleSpinBox.setDecimals(3)
        self.ui.doubleSpinBox.setSingleStep(0.1)
        self.ui.doubleSpinBox.setValue(1.0)               # Amplitude pre-factor: 1.0 = 100% of max(data)
        self.ui.doubleSpinBox_fitAmpGuess.setValue(3.0)   # Amp. Bounds: x3 (matches old hard-coded default)
        self.ui.doubleSpinBox_maxShiftFit.setValue(0.5)   # Lattice Bounds: ±50% (0=fix lattice)
        self.ui.doubleSpinBox_3.setValue(2.0)             # Sigma Bounds: ×2

        # Wire difference plot checkbox
        self.ui.checkBox_differencePlot.stateChanged.connect(self.update_processed_plot)


        # Connects BG buttons
        # NOTE: twoThetaMin/Max are DELIBERATELY NOT wired to run_background
        # or update_plot here. Cropping triggers a full BG recompute over
        # every loaded pattern, which is expensive on large series. The
        # user now commits a new range explicitly via the "Recalculate BG"
        # button (wired further down).
        self.ui.spinBox_PatternNumber.valueChanged.connect(self.run_background)
        self.ui.doubleSpinBox_BG_Prominence.valueChanged.connect(self.run_background)
        self.ui.doubleSpinBox_BG_Height.valueChanged.connect(self.run_background)
        self.ui.spinBox_BG_Order.valueChanged.connect(self.run_background)
        self.ui.doubleSpinBox_BG_width.valueChanged.connect(self.run_background)
        self.ui.doubleSpinBox_ALS_lam.valueChanged.connect(self.run_background)
        self.ui.doubleSpinBox_ALS_p.valueChanged.connect(self.run_background)
        self.ui.doubleSpinBox_ALS_niter.valueChanged.connect(self.run_background)
        self.ui.checkBox_ALSfit.stateChanged.connect(self.run_background)
        self.ui.checkBox_polyFit.stateChanged.connect(self.run_background)
        self.ui.checkBox_subtractBG.stateChanged.connect(self.run_background)

        # Everything that changes the view triggers update_plot
        # (twoThetaMin/Max deliberately omitted - see note above)
        self.ui.doubleSpinBox_BG_Prominence.valueChanged.connect(self.update_plot)
        self.ui.doubleSpinBox_BG_Height.valueChanged.connect(self.update_plot)
        self.ui.spinBox_BG_Order.valueChanged.connect(self.update_plot)

        # --- "Recalculate BG" button ---
        # Sole entry point for committing a new 2-theta range. Re-runs BG
        # on the current pattern (1D plots) AND rebuilds the full 2D stack
        # with progress popups so the user can see what's going on.
        self.ui.pushButton_recalculateBG.clicked.connect(self.recalculate_bg)
        
        # --- Pre-check ALS ---
        self.ui.checkBox_ALSfit.setChecked(True)      # ALS ticked by default
        self.ui.checkBox_polyFit.setChecked(False)    # Poly unticked
        # --- Connect signals AFTER setting initial state ---
        self.ui.checkBox_ALSfit.stateChanged.connect(self.BGfit_checkbox_changed)
        self.ui.checkBox_polyFit.stateChanged.connect(self.BGfit_checkbox_changed)
        
        # 2D Plot
        # Assign UI widget to self
        self.plotWidget_2D = self.ui.plotWidget_2D

        # Initialise empty containers - 2D stack will be built when data is loaded
        self._peak_overlay_items = []
        # (Old call to self.plot2D_from_raw() removed - legacy function created
        # its own ImageItem and detached the LUT; update_2D_plot() is used now.)

        # ---------------------------------------------------------------
        # 2D plot toolbar: intensity scaling (Lin / Log / Sqrt) and a
        # cycle-colormap button. Inserted ABOVE the LUT inside the
        # existing verticalLayout_8 so the user gets a small row of
        # buttons followed by the LUT, followed by the heatmap.
        # ---------------------------------------------------------------
        self._intensity_scale = DEFAULT_2D_SCALE      # 'linear' / 'log' / 'sqrt'
        self._cmap_idx        = 0
        self._setup_2D_toolbar()

        # Connect your "Load JCPDS" button
        self.ui.pushButton_loadJCPDS.clicked.connect(self.load_jcpds_files)
        self.phases_list = []
        
        # Sequential Fitting
        self.sequentialFitResults = {}
        self.sequentialFitResultsPeaks = {}
        self.all_patterns_BGsub = []  # populated as patterns are loaded
        self._seq_current_frame = None
        self._seq_end_frame = None
        self._seq_timer = QTimer()
        self._seq_timer.timeout.connect(self._process_next_frame)
        self._seq_mode = 'sequential'  # default mode

        # ---------------------------------------------------------------
        # ITEM 6: Green "+" button next to the Pressure Guess table so
        # the user can dynamically add [frame, P] rows.
        # ---------------------------------------------------------------
        self._setup_pressure_guess_table()

        # Right-click context menu on the JCPDS table (item 3): allows the
        # user to delete a phase entirely (with confirmation).
        self._setup_jcpds_context_menu()

    # =====================================================================
    # PRESSURE-GUESS TAB WIDGET
    # =====================================================================
    # The previous design used a single global QTableWidget
    # (tableWidget_PressureGuessTable). It has been replaced by a
    # QTabWidget (tabWidget) with one tab per phase that
    # is currently ticked "Use" in the JCPDS table. Each tab holds its
    # own QTableWidget with three columns:
    #     col 0 : Frame
    #     col 1 : Pressure (GPa)
    #     col 2 : Fix      (QCheckBox - if ticked, the lattice
    #                       parameters of THIS phase are held fixed
    #                       for the frame matched to this row)
    # =====================================================================

    def _setup_pressure_guess_table(self):
        """Wire up the three buttons that sit to the LEFT of the pressure
        guess tab widget:
            green '+' (pushButton_2): append a new [frame, P, fix] row
                                       to the *currently active* tab
            red   '-' (pushButton)  : delete the currently selected row
                                       in the active tab
            red   'x' (pushButton_3): clear all rows in the active tab

        The tab widget itself starts empty; tabs are added/removed
        dynamically by _rebuild_pressure_guess_tabs() in response to
        changes in the JCPDS-table 'Use' checkboxes.
        """
        # Per-phase storage: phase_index -> QTableWidget owned by a tab.
        # Tabs are rebuilt whenever the set of "Use" phases changes; the
        # underlying [frame, P, fix] data is preserved across rebuilds
        # via _phase_pg_data (see _rebuild_pressure_guess_tabs).
        self._phase_pg_tables = {}        # phase_index -> QTableWidget
        self._phase_pg_data   = {}        # phase_index -> [(frame, P, fix), ...]

        def _style_btn(btn, bg_hex, tooltip):
            if btn is None:
                return
            btn.setToolTip(tooltip)
            btn.setFixedSize(28, 24)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {bg_hex}; color: white; "
                f"font-weight: bold; border-radius: 4px; }} "
                f"QPushButton:hover {{ background-color: black; }}")

        self._btn_add_pressure_row    = getattr(self.ui, 'pushButton_2', None)
        self._btn_remove_pressure_row = getattr(self.ui, 'pushButton',   None)
        self._btn_clear_pressure_rows = getattr(self.ui, 'pushButton_3', None)

        _style_btn(self._btn_add_pressure_row,    '#2ecc71',
                   "Add a new [frame, pressure, fix] row to the active tab")
        _style_btn(self._btn_remove_pressure_row, '#e74c3c',
                   "Delete the selected row in the active tab")
        _style_btn(self._btn_clear_pressure_rows, '#c0392b',
                   "Delete ALL rows in the active tab")

        if self._btn_remove_pressure_row is not None:
            self._btn_remove_pressure_row.setText('\u2212')   # minus
        if self._btn_clear_pressure_rows is not None:
            self._btn_clear_pressure_rows.setText('\u00d7')   # multiplication

        for btn, slot in (
                (self._btn_add_pressure_row,    self._add_pressure_guess_row),
                (self._btn_remove_pressure_row, self._remove_pressure_guess_row),
                (self._btn_clear_pressure_rows, self._clear_pressure_guess_rows)):
            if btn is None:
                continue
            try:
                btn.clicked.disconnect()
            except Exception:
                pass
            btn.clicked.connect(slot)

    # ---------------------------------------------------------------------
    # Active-tab helper
    # ---------------------------------------------------------------------
    def _active_pressure_guess_table(self):
        """Return the QTableWidget held by the currently-selected tab in
        tabWidget, or None if there are no tabs yet.
        Looks up the phase index that the tab belongs to via the
        'phase_index' Qt property the table carries.
        """
        tabw = getattr(self.ui, 'tabWidget', None)
        if tabw is None or tabw.count() == 0:
            return None
        idx = tabw.currentIndex()
        if idx < 0:
            return None
        page = tabw.widget(idx)
        if page is None:
            return None
        # The page is a small container holding the QTableWidget as its
        # only child layout item; pull it out.
        for child in page.findChildren(QtWidgets.QTableWidget):
            return child
        return None

    def _phase_index_for_active_tab(self):
        """Return the phase index that the currently-active pressure
        guess tab is bound to, or None."""
        table = self._active_pressure_guess_table()
        if table is None:
            return None
        pi = table.property('phase_index')
        if pi is None:
            return None
        try:
            return int(pi)
        except (TypeError, ValueError):
            return None

    # ---------------------------------------------------------------------
    # Tab (re)build
    # ---------------------------------------------------------------------
    def _rebuild_pressure_guess_tabs(self, *args, **kwargs):
        """Rebuild the pressure-guess tab widget so that exactly one tab
        exists for every phase whose 'Use' checkbox is ticked in the
        JCPDS table.

        Per-tab table contents are preserved across rebuilds: before
        tearing down the existing tabs we snapshot every table into
        self._phase_pg_data, then restore from that snapshot when we
        rebuild. This means ticking 'Use' off and back on does NOT
        wipe the user's [frame, P, fix] rows for that phase.

        The *args/**kwargs swallow any positional state value Qt
        passes through stateChanged so this is safe to use as a slot
        for both the JCPDS Use checkbox and direct internal calls.
        """
        tabw = getattr(self.ui, 'tabWidget', None)
        if tabw is None:
            return

        # Make sure the tab widget actually has room to render. Without
        # an explicit minimum size + expanding policy, a freshly-empty
        # QTabWidget inside a horizontal layout can collapse to ~zero
        # width and the first new tab is invisible.
        if tabw.minimumWidth() < 200:
            tabw.setMinimumSize(200, 150)
        tabw.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)

        # Snapshot existing per-phase tables -> _phase_pg_data
        for phase_index, tbl in list(self._phase_pg_tables.items()):
            self._phase_pg_data[phase_index] = self._snapshot_table_rows(tbl)

        # Tear down existing tabs (including the static "template" tab
        # placed in EXODUS_GUI_ui.py to give the widget an initial size)
        tabw.blockSignals(True)
        while tabw.count():
            page = tabw.widget(0)
            tabw.removeTab(0)
            if page is not None:
                page.deleteLater()
        self._phase_pg_tables.clear()

        # Discover phases that are "in use" right now
        used = []
        if hasattr(self, 'phases') and self.phases is not None:
            for row in range(self.ui.tableWidget_JCPDSTable.rowCount()):
                chk_use = self._get_table_checkbox(
                    self.ui.tableWidget_JCPDSTable, row, 2)
                if chk_use is not None and chk_use.isChecked():
                    used.append(row)

        for phase_index in used:
            tab_label = self._phase_tab_label(phase_index)
            page = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)

            tbl = QtWidgets.QTableWidget(page)
            tbl.setColumnCount(3)
            tbl.setHorizontalHeaderLabels(['Frame', 'Pressure', 'Fix'])
            tbl.setRowCount(0)
            tbl.setProperty('phase_index', int(phase_index))
            try:
                tbl.horizontalHeader().setStretchLastSection(True)
            except Exception:
                pass
            font = QtGui.QFont()
            font.setPointSize(8)
            tbl.setFont(font)

            lay.addWidget(tbl)
            tabw.addTab(page, tab_label)
            self._phase_pg_tables[phase_index] = tbl

            # Restore any prior rows for this phase
            for row in self._phase_pg_data.get(phase_index, []):
                self._append_pressure_guess_row(tbl, *row)

        tabw.blockSignals(False)

    def _phase_tab_label(self, phase_index):
        """Pretty label for a tab: the phase name if known, else its index."""
        try:
            name = self.phases.get(f'phase_{phase_index}_name', '')
            if name:
                return f"{name}"
        except Exception:
            pass
        return f"Phase {phase_index}"

    @staticmethod
    def _snapshot_table_rows(tbl):
        """Return a list of (frame, pressure, fix) tuples extracted from
        a per-phase pressure-guess QTableWidget. Robust against missing
        items / empty cells / unparseable text."""
        out = []
        if tbl is None:
            return out
        for r in range(tbl.rowCount()):
            frame_item = tbl.item(r, 0)
            pres_item  = tbl.item(r, 1)
            fix_widget = tbl.cellWidget(r, 2)
            try:
                if frame_item is None or pres_item is None:
                    continue
                if not frame_item.text().strip() or not pres_item.text().strip():
                    continue
                frame = int(float(frame_item.text()))
                pres  = float(pres_item.text())
            except (ValueError, AttributeError):
                continue
            fix = False
            if isinstance(fix_widget, QtWidgets.QCheckBox):
                fix = bool(fix_widget.isChecked())
            else:
                # Wrapped (centred) checkbox: drill into children.
                inner = getattr(fix_widget, '_checkbox', None)
                if inner is None and fix_widget is not None:
                    inner = fix_widget.findChild(QtWidgets.QCheckBox)
                if isinstance(inner, QtWidgets.QCheckBox):
                    fix = bool(inner.isChecked())
            out.append((frame, pres, fix))
        return out

    # -------------------------------------------------------------------
    # Centred-checkbox cell widgets
    # -------------------------------------------------------------------
    # The old approach used a stylesheet trick
    # (``margin-left:50%; margin-right:50%``) which only centres badly
    # and breaks visually once the column becomes narrow (which it now
    # is, since the Show/Use/Fix columns have been shrunk).
    #
    # The new approach wraps each QCheckBox in a tiny QWidget with a
    # zero-margin, centred QHBoxLayout — so the checkbox always sits in
    # the geometric middle of the cell regardless of column width.
    #
    # NB. setCellWidget() now stores the WRAPPER, not the checkbox
    # itself, so every site that previously called
    # ``tableWidget.cellWidget(row, col).isChecked()`` must go via
    # ``self._get_table_checkbox(tbl, row, col)`` instead. That helper
    # is also tolerant of cells that still hold a bare QCheckBox (so
    # nothing breaks if future code goes back to the old style).
    @staticmethod
    def _centered_checkbox(checked=False, tooltip=None, on_state_changed=None):
        """Create a (wrapper, checkbox) pair where the checkbox is
        horizontally centred inside the wrapper widget. Returns the
        WRAPPER -- pass it to ``setCellWidget``. The checkbox itself is
        attached as ``wrapper._checkbox`` for retrieval.

        ``on_state_changed`` is connected to ``stateChanged`` if given.
        """
        wrapper = QtWidgets.QWidget()
        wrapper.setStyleSheet("background: transparent;")
        layout = QtWidgets.QHBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)
        chk = QtWidgets.QCheckBox()
        chk.setChecked(bool(checked))
        if tooltip:
            chk.setToolTip(tooltip)
        if on_state_changed is not None:
            chk.stateChanged.connect(on_state_changed)
        layout.addWidget(chk)
        wrapper._checkbox = chk
        return wrapper

    @staticmethod
    def _get_table_checkbox(tbl, row, col):
        """Return the QCheckBox stored at (row, col) in ``tbl``, whether
        it was added bare or wrapped in a centring container. Returns
        ``None`` if nothing checkbox-like is there."""
        cw = tbl.cellWidget(row, col)
        if cw is None:
            return None
        if isinstance(cw, QtWidgets.QCheckBox):
            return cw
        # Look for the back-reference set by _centered_checkbox(); fall
        # back to a child lookup if necessary.
        inner = getattr(cw, '_checkbox', None)
        if isinstance(inner, QtWidgets.QCheckBox):
            return inner
        return cw.findChild(QtWidgets.QCheckBox)

    def _append_pressure_guess_row(self, tbl, frame=0, pressure=0.0, fix=False):
        """Append a [frame, pressure, fix] row to a given per-phase
        pressure-guess QTableWidget.  Used by both the user-driven '+'
        button and by _rebuild_pressure_guess_tabs (data restore)."""
        if tbl is None:
            return
        row = tbl.rowCount()
        tbl.insertRow(row)

        frame_item = QtWidgets.QTableWidgetItem(str(int(frame)))
        pres_item  = QtWidgets.QTableWidgetItem(f"{float(pressure):g}")
        frame_item.setTextAlignment(Qt.AlignCenter)
        pres_item.setTextAlignment(Qt.AlignCenter)
        tbl.setItem(row, 0, frame_item)
        tbl.setItem(row, 1, pres_item)

        # Properly centred Fix checkbox (used to rely on a percentage
        # stylesheet that didn't centre at narrow column widths).
        wrapper = self._centered_checkbox(
            checked=bool(fix),
            tooltip=("If ticked, the lattice parameters of this phase "
                     "are held fixed for the frame matched to this row."))
        tbl.setCellWidget(row, 2, wrapper)
        return row

    # -------------------------------------------------------------------
    # 2D-plot toolbar: intensity scale (Lin / Log / Sqrt) and color cycle
    # -------------------------------------------------------------------
    # =====================================================================
    # JCPDS CONTEXT MENU (item 3): right-click -> delete phase
    # =====================================================================
    def _setup_jcpds_context_menu(self):
        """Enable a right-click context menu on the JCPDS table that
        offers, at minimum, 'Delete phase'. The deletion is confirmed
        via a QMessageBox before anything is removed.
        """
        tbl = self.ui.tableWidget_JCPDSTable
        tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        try:
            tbl.customContextMenuRequested.disconnect()
        except Exception:
            pass
        tbl.customContextMenuRequested.connect(self._on_jcpds_context_menu)

    def _on_jcpds_context_menu(self, pos):
        """Build and pop up the JCPDS row context menu at pos (in the
        table's viewport coordinates).
        """
        tbl = self.ui.tableWidget_JCPDSTable
        item = tbl.itemAt(pos)
        # Allow context menu on either an item OR an empty cell that
        # happens to host a cellWidget (most of our columns do).
        row = tbl.rowAt(pos.y())
        if row < 0:
            return

        try:
            name = self.phases.get(f'phase_{row}_name', f'phase_{row}')
        except Exception:
            name = f'phase_{row}'

        menu = QtWidgets.QMenu(tbl)
        act_delete = menu.addAction(f"Delete phase '{name}' ...")
        chosen = menu.exec_(tbl.viewport().mapToGlobal(pos))
        if chosen == act_delete:
            self._delete_jcpds_phase(row)

    def _delete_jcpds_phase(self, row):
        """Remove a JCPDS phase entirely: drop its row, drop its
        per-phase tab (if any), drop its keys from self.phases, and
        renumber every phase whose index sits ABOVE the deleted one
        so that the assumption 'JCPDS-table row index == phase index'
        (used throughout this codebase) is preserved.

        A confirmation dialog is shown first.
        """
        if not hasattr(self, 'phases') or self.phases is None:
            return
        n = int(self.phases.get('phases_Number', 0))
        if not (0 <= row < n):
            return

        try:
            name = self.phases.get(f'phase_{row}_name', f'phase_{row}')
        except Exception:
            name = f'phase_{row}'

        reply = QtWidgets.QMessageBox.question(
            self, "Delete phase",
            f"Delete phase '{name}' from the JCPDS table?\n\n"
            "This removes its row, its pressure-guess tab (if any), "
            "and any cached fit results for it. This cannot be undone.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return

        # ---- 1. Drop UI row -------------------------------------------------
        self.ui.tableWidget_JCPDSTable.removeRow(row)

        # ---- 2. Renumber self.phases ---------------------------------------
        # All phase_{i}_* keys with i > row are shifted DOWN by one. Keys
        # belonging to the deleted phase are dropped.
        per_phase_suffixes = (
            '_name', '_crystal_system', '_unit_cell', '_unit_cell_0',
            '_HKL', '_compression_constants', '_file_path',
            '_fix_lattice')

        # Drop the deleted phase's keys
        for suf in per_phase_suffixes:
            self.phases.pop(f'phase_{row}{suf}', None)

        # Shift down everything above
        for i in range(row + 1, n):
            for suf in per_phase_suffixes:
                src = f'phase_{i}{suf}'
                if src in self.phases:
                    self.phases[f'phase_{i - 1}{suf}'] = self.phases.pop(src)

        new_n = n - 1
        self.phases['phases_Number'] = new_n
        self.phases['phases_Number_array'] = np.arange(new_n)
        # phases_used: drop the deleted index, shift larger indices down
        old_used = list(self.phases.get('phases_used', np.arange(new_n + 1)))
        new_used = [(p if p < row else p - 1)
                    for p in old_used if p != row]
        self.phases['phases_used'] = np.array(new_used, dtype=int)

        # If we just deleted the last phase, drop the dict altogether so
        # downstream code's "no phases loaded" guards trigger cleanly.
        if new_n == 0:
            self.phases = None

        # ---- 3. Renumber cached fit results --------------------------------
        # sequentialFitResults / sequentialFitResultsPeaks are keyed by
        # 'frame_X' -> {'phase_Y': UC_fit, ...}. Same renumber rule.
        for cache in (getattr(self, 'sequentialFitResults', {}),
                      getattr(self, 'sequentialFitResultsPeaks', {})):
            for fkey, per_phase in list(cache.items()):
                if not isinstance(per_phase, dict):
                    continue
                # Drop the deleted phase, shift others down
                for k in list(per_phase.keys()):
                    if not k.startswith('phase_'):
                        continue
                    # Match 'phase_<int>' and 'phase_<int>_error'
                    parts = k.split('_')
                    try:
                        idx = int(parts[1])
                    except (IndexError, ValueError):
                        continue
                    if idx == row:
                        per_phase.pop(k, None)
                    elif idx > row:
                        new_key = '_'.join(['phase', str(idx - 1)] + parts[2:])
                        per_phase[new_key] = per_phase.pop(k)

        # ---- 4. Renumber the per-phase pressure-guess data store ----------
        # _phase_pg_data keys are phase indices. Drop the deleted one,
        # shift larger indices down. The tabs themselves get rebuilt
        # from this data immediately below.
        if hasattr(self, '_phase_pg_data'):
            new_pg_data = {}
            for pi, rows in self._phase_pg_data.items():
                if pi == row:
                    continue
                new_pg_data[pi if pi < row else pi - 1] = rows
            self._phase_pg_data = new_pg_data

        # ---- 5. Re-bind row-keyed callbacks --------------------------------
        # The colour button, edit button and pressure spinbox were each
        # bound with `partial(callback, row)` in load_jcpds_files. After
        # we delete row `row`, every row whose new index is `i` was
        # previously `i+1`, so its callbacks still send the OLD index.
        # Re-bind them all so they send the post-deletion row.
        self._rebind_jcpds_row_callbacks()

        # ---- 6. Rebuild the pressure-guess tabs and refresh plots ----------
        self._rebuild_pressure_guess_tabs()
        try:
            self.update_processed_plot()
        except Exception:
            pass

    def _rebind_jcpds_row_callbacks(self):
        """Re-bind every row-keyed callback in the JCPDS table so the
        partial(...) objects send the CURRENT row index. Needed after
        we delete a row, because rows above the deletion point shift
        down by one but their cellWidgets keep their original
        partial(slot, old_row) connections.
        """
        tbl = self.ui.tableWidget_JCPDSTable
        for row in range(tbl.rowCount()):
            spin_p = tbl.cellWidget(row, 3)
            if isinstance(spin_p, QtWidgets.QDoubleSpinBox):
                try:
                    spin_p.valueChanged.disconnect()
                except Exception:
                    pass
                spin_p.valueChanged.connect(partial(self.on_pressure_changed, row))

            colour_btn = tbl.cellWidget(row, 6)
            if isinstance(colour_btn, QtWidgets.QPushButton):
                try:
                    colour_btn.clicked.disconnect()
                except Exception:
                    pass
                colour_btn.clicked.connect(
                    partial(self._on_colour_button_clicked, row))

            edit_btn = tbl.cellWidget(row, 7)
            if isinstance(edit_btn, QtWidgets.QPushButton):
                try:
                    edit_btn.clicked.disconnect()
                except Exception:
                    pass
                edit_btn.clicked.connect(
                    partial(self._on_edit_button_clicked, row))

    # =====================================================================
    # END JCPDS CONTEXT MENU
    # =====================================================================

    def _setup_2D_toolbar(self):
        """Wire up the intensity-scale and colormap buttons that live in
        the new ``2D Plot and Fits`` panel of the UI:
            pushButton_4 -> Lin
            pushButton_5 -> Log
            pushButton_6 -> Sqrt
            pushButton_7 -> Colour (cycle colormaps)

        These used to be a dynamic toolbar inserted above the LUT widget;
        they are now static UI buttons (defined in EXODUS_GUI_ui.py) and
        this method only attaches behaviour and styling. The Lin/Log/Sqrt
        trio is exclusive (mutually-checkable). Robust against any of the
        buttons being absent (e.g. if the UI is rolled back).
        """
        # --- common styling ---------------------------------------------
        scale_btn_qss = (
            "QPushButton { background-color: #4a4a4a; color: white; "
            "border-radius: 3px; padding: 2px 8px; } "
            "QPushButton:checked { background-color: #2980b9; "
            "font-weight: bold; } "
            "QPushButton:hover:!checked { background-color: #5a5a5a; }")
        cmap_btn_qss = (
            "QPushButton { background-color: #4a4a4a; color: white; "
            "border-radius: 3px; padding: 2px 8px; } "
            "QPushButton:hover { background-color: #5a5a5a; }")

        def _config_scale_btn(btn, tip, cb):
            if btn is None:
                return None
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setStyleSheet(scale_btn_qss)
            try:
                btn.clicked.disconnect()
            except Exception:
                pass
            btn.clicked.connect(cb)
            return btn

        # --- bind the four UI buttons -----------------------------------
        self._btn_2D_lin = _config_scale_btn(
            getattr(self.ui, 'pushButton_4', None),
            "Linear intensity scale",
            lambda: self._set_2D_scale('linear'))
        self._btn_2D_log = _config_scale_btn(
            getattr(self.ui, 'pushButton_5', None),
            "Logarithmic intensity scale (usually best for diffraction data)",
            lambda: self._set_2D_scale('log'))
        self._btn_2D_sqrt = _config_scale_btn(
            getattr(self.ui, 'pushButton_6', None),
            "Square-root intensity scale",
            lambda: self._set_2D_scale('sqrt'))

        self._btn_2D_cmap = getattr(self.ui, 'pushButton_7', None)
        if self._btn_2D_cmap is not None:
            self._btn_2D_cmap.setToolTip("Cycle through colour maps")
            self._btn_2D_cmap.setCheckable(False)
            self._btn_2D_cmap.setStyleSheet(cmap_btn_qss)
            try:
                self._btn_2D_cmap.clicked.disconnect()
            except Exception:
                pass
            self._btn_2D_cmap.clicked.connect(self._cycle_colormap)

        # Initial check state mirrors the chosen default
        self._refresh_2D_scale_buttons()

    def _refresh_2D_scale_buttons(self):
        """Make exactly one of the Lin/Log/Sqrt buttons appear pressed."""
        mode = getattr(self, '_intensity_scale', 'linear')
        for m, btn in (('linear', getattr(self, '_btn_2D_lin', None)),
                       ('log',    getattr(self, '_btn_2D_log', None)),
                       ('sqrt',   getattr(self, '_btn_2D_sqrt', None))):
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(m == mode)
            btn.blockSignals(False)

    def _set_2D_scale(self, mode):
        """Switch the 2D heatmap intensity scaling. mode in
        {'linear', 'log', 'sqrt'}.  Triggers an Easter-egg popup the
        first time the user picks 'log' in a session (if enabled)."""
        prev_mode = getattr(self, '_intensity_scale', 'linear')
        self._intensity_scale = mode
        self._refresh_2D_scale_buttons()
        # Re-render with the new transform; the LUT histogram will then
        # span the transformed data range, which is much more uniform.
        try:
            self.update_2D_plot()
        except Exception as e:
            print(f"2D plot redraw failed after scale change: {e}")

        # Easter egg: pop up RSW.png the first time the user clicks Log.
        # Comment out ENABLE_LOG_EASTER_EGG at top of file to disable.
        if (mode == 'log'
                and prev_mode != 'log'   # only on the transition INTO log
                and globals().get('ENABLE_LOG_EASTER_EGG', False)):
            try:
                self._show_log_easter_egg()
            except Exception as e:
                print(f"Easter egg failed: {e}")

    def _cycle_colormap(self):
        """Step through COLORMAP_CYCLE on the LUT widget."""
        if not hasattr(self.ui, 'lut_widget'):
            return
        cmaps = COLORMAP_CYCLE if COLORMAP_CYCLE else ['thermal']
        self._cmap_idx = (getattr(self, '_cmap_idx', 0) + 1) % len(cmaps)
        try:
            self.ui.lut_widget.gradient.loadPreset(cmaps[self._cmap_idx])
        except Exception as e:
            print(f"Could not switch colormap to '{cmaps[self._cmap_idx]}': {e}")

    def _apply_2D_intensity_scaling(self, arr):
        """Apply self._intensity_scale to a 2D intensity array. Returns
        a new ndarray suitable for ImageItem.setImage; the LUT will then
        operate on transformed values, which spreads the histogram more
        uniformly than raw data does."""
        mode = getattr(self, '_intensity_scale', 'linear')
        a = np.nan_to_num(arr, nan=0.0)
        if mode == 'log':
            # Shift so the minimum is 0, then log1p for numerical safety.
            # log1p(0) == 0, so empty regions stay at the floor.
            shifted = a - np.nanmin(a)
            return np.log1p(np.clip(shifted, 0, None))
        if mode == 'sqrt':
            shifted = a - np.nanmin(a)
            return np.sqrt(np.clip(shifted, 0, None))
        return a    # 'linear' and any unknown value

    def _show_log_easter_egg(self):
        """Pop up RSW.png from the same directory as this script. Silent
        no-op if the file isn't there. Toggle off by commenting out
        ENABLE_LOG_EASTER_EGG at the top of this file."""
        img_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'RSW.png')
        if not os.path.exists(img_path):
            print(f"  (Easter egg image not found at {img_path}; skipping.)")
            return

        from PyQt5.QtGui import QPixmap
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("...")
        dlg.setModal(False)   # non-modal so it doesn't block the GUI
        lbl = QtWidgets.QLabel(dlg)
        pix = QPixmap(img_path)
        # Scale down if the image is huge - keep aspect ratio
        max_w, max_h = 800, 600
        if pix.width() > max_w or pix.height() > max_h:
            pix = pix.scaled(max_w, max_h,
                             Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lbl.setPixmap(pix)
        v = QtWidgets.QVBoxLayout(dlg)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(lbl)
        # Keep a reference so the GC doesn't close it instantly
        self._easter_egg_dlg = dlg
        dlg.show()

    def _add_pressure_guess_row(self):
        """Append a new empty [frame, pressure, fix] row to the table in
        the currently-active pressure-guess tab. The user then clicks a
        cell to edit it.
        """
        tbl = self._active_pressure_guess_table()
        if tbl is None:
            QtWidgets.QMessageBox.information(
                self, "No phase selected",
                "There are no pressure-guess tabs yet. Tick the 'Use' "
                "checkbox on a phase in the JCPDS table first.")
            return

        # Pre-fill 'frame' with a sensible default: either 0, or one past
        # the largest frame already present in this tab.
        default_frame = 0
        for r in range(tbl.rowCount()):
            itm = tbl.item(r, 0)
            if itm is not None and itm.text().strip():
                try:
                    default_frame = max(default_frame, int(float(itm.text())) + 1)
                except ValueError:
                    pass

        row = self._append_pressure_guess_row(tbl, default_frame, 0.0, False)
        # Jump straight into edit mode on the pressure cell
        if row is not None:
            tbl.editItem(tbl.item(row, 1))

    def _remove_pressure_guess_row(self):
        """Delete the currently-selected row(s) in the active tab's table."""
        tbl = self._active_pressure_guess_table()
        if tbl is None:
            return
        rows = sorted({idx.row() for idx in tbl.selectedIndexes()}, reverse=True)
        if not rows:
            cur = tbl.currentRow()
            if cur >= 0:
                rows = [cur]
        for r in rows:
            tbl.removeRow(r)

    def _clear_pressure_guess_rows(self):
        """Delete all rows in the active tab's table, after confirmation."""
        tbl = self._active_pressure_guess_table()
        if tbl is None or tbl.rowCount() == 0:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Clear pressure table",
            "Delete all [frame, P, fix] entries in this tab?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            tbl.setRowCount(0)

    def run_fit_all(self):
        # Logic for sequential buttons
        if self.ui.checkBox_sequential.isChecked():
            print("Running Sequential LB")
            self.LB_fit_sequential()
    
        elif self.ui.checkBox_pressureWalk.isChecked():
            print("Running Pressure Walk")
            self.pressureWalk_fit()
    
        elif self.ui.checkBox_poressureGuess.isChecked():
            print("Running Pressure Guess")
            self.pressureGuess_fit()
    
        # elif self.ui.checkBox_sequentialplus.isChecked():
        #     print("Running Sequential Plus")
        #     self.sequentialPlus_fit()
    
        else:
            print("No fit mode selected.")    
        
    def load_poni_path(self):
        # Loads the poni path. Not the poni. Only the file name
        poni_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a file",
            "",
            "Poni Files (*.poni);;All Files (*)")
        if poni_path:
            self.poni_path = poni_path
            self.ui.lineEdit_poniList.setText(poni_path)
            #print("Loaded:", self.poni_path)
    
            # Load wavelength from the PONI file
            self.WL = exc.load_WL(self.poni_path)
            #print("Wavelength:", self.WL)
            
                
    def load_data_paths(self):
        # Loads the data paths. Not the data. Only the file names
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select data files",
            "",
            "All Supported (*.fxye *.xy *.dat *.chi);;"
            "GSAS-II Files (*.fxye);;"
            "XY Files (*.xy);;"
            "DAT Files (*.dat);;"
            "CHI Files (*.chi);;"
            "All Files (*)"
        )

        if not file_paths:
            return   # user cancelled - keep existing state

        # ---------------------------------------------------------------
        # ITEM 1: Reset all plots and stored results when new data loaded
        # ---------------------------------------------------------------
        self._reset_all_plots_and_results()

        # Store new list
        self.file_paths = file_paths
        self.ui.listWidget_fileList.clear()
        for i, path in enumerate(file_paths):
            item = QtWidgets.QListWidgetItem(os.path.basename(path))
            item.setData(Qt.UserRole, path)
            self.ui.listWidget_fileList.addItem(item)

        self.ui.spinBox_PatternNumber.setMinimum(0)
        self.ui.spinBox_PatternNumber.setMaximum(len(file_paths) - 1)
        self._max_frame = len(file_paths) - 1
        self._update_table_frame_ranges()

        # ---------------------------------------------------------------
        # ITEM 2: Build 2D stack for ALL files immediately so the 2D plot
        # is populated as soon as data is loaded (not lazily per-pattern).
        # ---------------------------------------------------------------
        self._build_full_2D_stack()

        # Auto-load first pattern
        self.ui.spinBox_PatternNumber.blockSignals(True)
        self.ui.spinBox_PatternNumber.setValue(0)
        self.ui.spinBox_PatternNumber.blockSignals(False)
        self.selection_changed(0)

        # Auto-click first file
        if self.ui.listWidget_fileList.count() > 0:
            first_item = self.ui.listWidget_fileList.item(0)
            self.ui.listWidget_fileList.setCurrentItem(first_item)
            self.file_selected(first_item)

        # Render the freshly-built 2D stack
        self.update_2D_plot()

    # -------------------------------------------------------------------
    # Item 1 helper: hard reset of plots / cached fit results
    # -------------------------------------------------------------------
    def _reset_all_plots_and_results(self):
        """Clear every plot, stored fit result, and cached background
        so loading a fresh set of data files starts from a clean state."""
        # Clear plot widgets
        try:
            self.plot_raw.clear()
        except Exception:
            pass
        try:
            self.plot_processed.clear()
        except Exception:
            pass
        try:
            # Preserve the persistent ImageItem (LUT is bound to it);
            # just blank its data rather than removing it.
            if hasattr(self.ui, 'img_item'):
                self.ui.img_item.clear()
        except Exception:
            pass

        # Also clear any lingering overlays (e.g. fitted-peak scatter dots)
        self._clear_2D_overlays()

        # Forget any stored per-pattern state
        self.all_patterns_BGsub = []
        self.sequentialFitResults = {}
        self.sequentialFitResultsPeaks = {}
        self.results = None
        self.results_frame_index = None

        # Drop raw-2D cache (will be rebuilt by _build_full_2D_stack)
        if hasattr(self, 'plot2DrawData'):
            del self.plot2DrawData
        if hasattr(self, 'twoTheta'):
            del self.twoTheta

        # Drop both caches so fresh files don't collide with stale keys.
        # raw_cache holds full untruncated patterns; bg_cache holds BG-
        # subtracted variants keyed by (path, method, params).
        if hasattr(self, 'data_manager'):
            self.data_manager.bg_cache = {}
            self.data_manager.raw_cache = {}

    # -------------------------------------------------------------------
    # Item 2 helper: compute BG-subtracted stack for every loaded file
    # -------------------------------------------------------------------
    def _build_full_2D_stack(self, show_progress_dialog=True):
        """Populate self.all_patterns_BGsub and self.current_twoTheta
        so update_2D_plot() can render everything without needing the
        user to click each pattern first.

        If show_progress_dialog is True (default), shows a cancellable
        progress dialog while iterating files. Callers that fire during
        interactive use (e.g. debounced 2theta-range changes) should pass
        False so the dialog doesn't flash up mid-interaction."""
        if not self.file_paths:
            return

        ttmin = self.ui.doubleSpinBox_twoThetaMin.value()
        ttmax = self.ui.doubleSpinBox_twoThetaMax.value()
        if ttmax <= ttmin:
            print(f"  2D rebuild skipped: twoThetaMin ({ttmin}) >= twoThetaMax ({ttmax})")
            return

        method = "ALS" if self.ui.checkBox_ALSfit.isChecked() else "POLY"
        params = {
            "twoThetaMin": ttmin,
            "twoThetaMax": ttmax,
            "lam":        self.ui.doubleSpinBox_ALS_lam.value(),
            "p":          self.ui.doubleSpinBox_ALS_p.value(),
            "niter":      self.ui.doubleSpinBox_ALS_niter.value(),
            "prominence": self.ui.doubleSpinBox_BG_Prominence.value(),
            "height":     self.ui.doubleSpinBox_BG_Height.value(),
            "order":      self.ui.spinBox_BG_Order.value(),
            "peakwidth":  self.ui.doubleSpinBox_BG_width.value(),
        }

        n_files = len(self.file_paths)

        # Only show the dialog if the caller asked for it AND we have enough
        # files to justify the flash.
        show_dialog = show_progress_dialog and n_files > 4
        load_cancelled = [False]    # mutable flag captured by callback

        if show_dialog:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle("Loading patterns")
            dlg.setModal(True)
            dlg.setMinimumWidth(380)
            dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)
            lay = QtWidgets.QVBoxLayout(dlg)
            lbl = QtWidgets.QLabel(f"Loading {n_files} patterns...")
            lay.addWidget(lbl)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, n_files)
            bar.setValue(0)
            lay.addWidget(bar)
            eta_lbl = QtWidgets.QLabel(
                f"File 0 of {n_files}   |   elapsed 0 s   |   ETA -")
            lay.addWidget(eta_lbl)

            btn_row = QtWidgets.QHBoxLayout()
            btn_row.addStretch(1)
            stop_btn = QtWidgets.QPushButton("Stop")
            stop_btn.setStyleSheet(
                "QPushButton { background-color: #e74c3c; color: white; "
                "font-weight: bold; padding: 6px 18px; border-radius: 4px; } "
                "QPushButton:hover { background-color: #c0392b; }")
            def _on_stop():
                load_cancelled[0] = True
            stop_btn.clicked.connect(_on_stop)
            btn_row.addWidget(stop_btn)
            lay.addLayout(btn_row)
            dlg.show()
            QtWidgets.QApplication.processEvents()

        start_time = time.monotonic()

        # ---------- Phase 1: parallel raw load (cheap if cached) ----------
        # Disk I/O releases the GIL, so threads scale almost linearly with
        # the number of files until the disk saturates. Already-cached
        # paths are skipped inside prefetch_raw.
        def _on_phase1_progress(done, total):
            if not show_dialog:
                return
            elapsed = time.monotonic() - start_time
            if done > 0:
                eta = (elapsed / done) * (total - done)
                eta_s = self._format_seconds(eta)
            else:
                eta_s = "-"
            bar.setValue(done)
            lbl.setText(f"Loading {total} patterns... (reading from disk)")
            eta_lbl.setText(
                f"File {done} of {total}   |   elapsed "
                f"{self._format_seconds(elapsed)}   |   ETA {eta_s}")
            QtWidgets.QApplication.processEvents()

        try:
            self.data_manager.prefetch_raw(
                self.file_paths,
                progress_callback=_on_phase1_progress if show_dialog else None,
                cancel_flag=load_cancelled,
            )
        except Exception as e:
            print(f"  prefetch_raw failed: {e}")

        # ---------- Phase 2: BG subtraction (now reads from raw cache) ----
        # Mode is set via the BG_PARALLEL_MODE setting at the top of this
        # file. Threads are the safe default; processes give a bigger
        # speedup on large batches but add ~1 s pool-startup overhead.
        if show_dialog:
            mode_label = {'serial': 'serial',
                          'threads': 'parallel - threads',
                          'processes': 'parallel - processes'}.get(
                              BG_PARALLEL_MODE, BG_PARALLEL_MODE)
            lbl.setText(f"Subtracting background ({method}, {mode_label})...")
            bar.setValue(0)
            QtWidgets.QApplication.processEvents()
        phase2_start = time.monotonic()

        def _on_phase2_progress(done, total):
            if not show_dialog:
                return
            elapsed = time.monotonic() - phase2_start
            if done > 0:
                eta = (elapsed / done) * (total - done)
                eta_s = self._format_seconds(eta)
            else:
                eta_s = "-"
            bar.setValue(done)
            eta_lbl.setText(
                f"File {done} of {total}   |   elapsed "
                f"{self._format_seconds(elapsed)}   |   ETA {eta_s}")
            QtWidgets.QApplication.processEvents()

        try:
            bg_results = self.data_manager.compute_backgrounds_parallel(
                self.file_paths, method, params,
                mode=BG_PARALLEL_MODE,
                progress_callback=_on_phase2_progress if show_dialog else None,
                cancel_flag=load_cancelled,
            )
        except Exception as e:
            print(f"  compute_backgrounds_parallel failed: {e}; "
                  f"falling back to serial.")
            if show_dialog:
                lbl.setText(f"Subtracting background ({method}, serial fallback)...")
                bar.setValue(0)
                QtWidgets.QApplication.processEvents()
            bg_results = []
            n_total = len(self.file_paths)
            for done_idx, path in enumerate(self.file_paths, start=1):
                if load_cancelled[0]:
                    break
                bg_results.append(
                    self.data_manager.get_background(path, method, params))
                if show_dialog:
                    _on_phase2_progress(done_idx, n_total)

        # ----------------------------------------------------------------
        # Defensive last-resort fallback: if every entry in bg_results is
        # None (i.e. the parallel pool started but every worker failed,
        # which can happen on Windows / frozen builds in edge cases), do
        # one serial pass with the synchronous DataManager.get_background.
        # This is the same code path the per-pattern 1D plot uses, so if
        # 1D plotting works at all, this will succeed. Report progress
        # through the dialog so the bar isn't stuck at 100% for the
        # duration of the fallback (that was the "jumps to 100% then
        # hangs for 20s" symptom).
        if bg_results and all(r is None for r in bg_results) and not load_cancelled[0]:
            print("  All parallel BG fits returned None - falling back to "
                  "serial loop.")
            if show_dialog:
                lbl.setText(f"Subtracting background ({method}, serial fallback)...")
                bar.setValue(0)
                QtWidgets.QApplication.processEvents()
            bg_results = []
            n_total = len(self.file_paths)
            for done_idx, path in enumerate(self.file_paths, start=1):
                if load_cancelled[0]:
                    break
                bg_results.append(
                    self.data_manager.get_background(path, method, params))
                if show_dialog:
                    _on_phase2_progress(done_idx, n_total)

        # ----------------------------------------------------------------
        # Assemble the stack ALIGNED to self.file_paths so pattern indices
        # match. Failed entries become a row of NaN; previously they were
        # silently skipped, which compressed the stack and (if the very
        # first file failed) also prevented twoTheta_ref from being set,
        # which in turn caused _build_full_2D_stack to bail at the
        # "if not stack: return" guard - leaving all_patterns_BGsub
        # untouched. That was the root of the "2D plot doesn't show after
        # load; only patterns I clicked appear" bug.
        n_files = len(self.file_paths)
        twoTheta_ref = None
        ref_len = None
        for res in bg_results:
            if res is None:
                continue
            tt, _, _, vBGsub = res
            twoTheta_ref = tt
            ref_len = len(vBGsub)
            break

        if show_dialog:
            try:
                dlg.close()
                dlg.deleteLater()
            except Exception:
                pass

        if twoTheta_ref is None or ref_len is None:
            # Genuinely nothing worked - leave 2D state untouched so we
            # don't clobber whatever was on screen before.
            print("  _build_full_2D_stack: every BG fit failed; "
                  "2D state unchanged.")
            return

        # Build a rectangular array with one row per file. Missing rows
        # stay as NaN, so update_2D_plot's nan_to_num turns them to zero
        # (visually the same colour as a low-intensity row).
        stack_padded = np.full((n_files, ref_len), np.nan)
        for i, res in enumerate(bg_results[:n_files]):
            if res is None:
                continue
            _, _, _, vBGsub = res
            n = min(len(vBGsub), ref_len)
            stack_padded[i, :n] = vBGsub[:n]

        self.all_patterns_BGsub = stack_padded
        self.current_twoTheta = twoTheta_ref
   
    def file_selected(self, item):
        # Selcts file by clicking
        row = self.ui.listWidget_fileList.row(item)  # index number
        full_path = item.data(Qt.UserRole)
    
        # store current pattern
        self.current_file_index = row
        self.current_file_path = full_path
    
        # trigger plotting/loading
        #self.load_pattern(full_path, row)
    
        # Update spinbox (this will NOT cause recursion unless connected incorrectly)
        self.ui.spinBox_PatternNumber.setValue(row)        
    
    # def load_single_pattern(self, file_path, index):
    #     """
    #     Load a single pattern into memory and update the current plot.
    #     """
    #     # Read the data from file
    #     twoTheta, valueIntBGsub = self.read_pattern_data(file_path)  # you should already have read_pattern_data()
    
    #     # Store in the GUI state
    #     self.current_twoTheta = twoTheta
    #     self.current_valueIntBGsub = valueIntBGsub
    #     self.current_file_index = index
    #     self.current_file_path = file_path
    
    #     # Update plot
    #     self.update_plot()
    
    def selection_changed(self, row):
        if row < 0:
            return
    
        # Prevent double-calling
        self.ui.spinBox_PatternNumber.blockSignals(True)
        self.ui.spinBox_PatternNumber.setValue(row)
        self.ui.spinBox_PatternNumber.blockSignals(False)
    
        self.update_plot()
        
    def BGfit_checkbox_changed(self, state):
        """Ensure only one checkbox is checked and update the plot."""
    
        sender = self.sender()
    
        if sender == self.ui.checkBox_ALSfit and state == Qt.Checked:
            self.ui.checkBox_polyFit.blockSignals(True)
            self.ui.checkBox_polyFit.setChecked(False)
            self.ui.checkBox_polyFit.blockSignals(False)
    
        elif sender == self.ui.checkBox_polyFit and state == Qt.Checked:
            self.ui.checkBox_ALSfit.blockSignals(True)
            self.ui.checkBox_ALSfit.setChecked(False)
            self.ui.checkBox_ALSfit.blockSignals(False)
    
        # Optional: If both unchecked, you can default to one
        if not (self.ui.checkBox_ALSfit.isChecked() or self.ui.checkBox_polyFit.isChecked()):
            # default to ALS
            self.ui.checkBox_ALSfit.setChecked(True)
    
        # Trigger replot
        self.update_plot()
    
    def update_plot(self):
        self.run_background()

    def recalculate_bg(self):
        """Slot for the "Recalculate BG" button.

        Behaviour: reloads ALL the patterns from disk and recomputes
        background subtraction over the *current* 2-theta range (i.e.
        whatever twoThetaMin/Max are sitting at right now). This is
        essentially the same work load_data_paths does on initial load
        - the user can dial in a new 2-theta crop without the program
        freezing, and only commits the work when they click this button.

        What this does NOT do:
          - It does not reopen the file dialog (file list is unchanged).
          - It does not jump back to pattern 0; the currently-selected
            pattern stays selected.
          - It does not wipe JCPDS phases or sequential / single-fit
            results - those aren't tied to the BG crop.

        Steps, in order:
          1. Push the new 2-theta range to the visible x-axis of the
             1D plots (without this the data is recomputed but the
             plot window stays frozen at the previous range).
          2. Drop the data caches so the rebuild actually re-loads
             from disk and recomputes BG over the new crop, instead
             of returning whatever's in cache for the old crop.
          3. Blank the visible plots so the user sees that something
             is happening - they will be repopulated immediately.
          4. Rebuild the full 2D stack (with progress popup).
          5. Re-render the 1D plot for the currently-selected pattern.
          6. Re-render the 2D heatmap.

        Safe to call with no files loaded - just no-ops everything.
        """
        if not getattr(self, 'file_paths', None):
            return

        ttmin = self.ui.doubleSpinBox_twoThetaMin.value()
        ttmax = self.ui.doubleSpinBox_twoThetaMax.value()
        if ttmax <= ttmin:
            print(f"  Recalculate BG skipped: twoThetaMin ({ttmin}) "
                  f">= twoThetaMax ({ttmax})")
            return

        # 1. Re-apply the visible x-range on the 1D plots.
        try:
            self.plot_raw.setXRange(ttmin, ttmax, padding=0)
            self.plot_processed.setXRange(ttmin, ttmax, padding=0)
        except Exception as e:
            print(f"  Recalculate BG: x-range update failed: {e}")

        # 2. Drop both data-manager caches. Without this, get_background
        #    can return a stale crop if (path, method, params) happens
        #    to map to a previously-cached entry. We also want the raw
        #    cache repopulated under the new crop window so prefetch_raw
        #    behaves the same way as it does on initial load.
        if hasattr(self, 'data_manager'):
            try:
                self.data_manager.bg_cache = {}
                self.data_manager.raw_cache = {}
            except Exception as e:
                print(f"  Recalculate BG: cache reset failed: {e}")

        # 3. Blank the visible plots. We deliberately DON'T touch
        #    sequentialFitResults / phase data / etc. - those aren't
        #    tied to the BG crop and the user expects them to survive
        #    a recalculate.
        try:
            self.plot_raw.clear()
        except Exception:
            pass
        try:
            self.plot_processed.clear()
        except Exception:
            pass
        try:
            if hasattr(self.ui, 'img_item'):
                self.ui.img_item.clear()
        except Exception:
            pass
        # The 2D plot's per-frame peak overlays are tied to fit results
        # that survived the reset, but their pixel positions depend on
        # the new image geometry; clearing now lets _replot_2D_peak_overlays
        # redraw them at the right spot when update_2D_plot runs.
        self._clear_2D_overlays()

        # Force the in-memory 2D stack to be re-allocated under the new
        # crop. Leaving the old (n_files, old_n_pts) array in place would
        # mean run_background's "len(...) > 0" check skips re-init and
        # we get a shape mismatch on row assignment.
        self.all_patterns_BGsub = []

        # 4. Rebuild the full 2D stack with the progress popup.
        try:
            self._build_full_2D_stack(show_progress_dialog=True)
        except Exception as e:
            print(f"  Recalculate BG: 2D rebuild failed: {e}")
            return

        # 5. Re-render the 1D plot for whichever pattern is selected
        #    (preserve the user's choice; don't snap back to pattern 0).
        self.run_background()

        # 6. Re-render the 2D heatmap.
        self.update_2D_plot()

    def _rebuild_2D_on_range_change(self):
        """Legacy helper - kept for backwards compatibility but no longer
        wired to anything. Use recalculate_bg() instead."""
        if not self.file_paths:
            return
        try:
            self._build_full_2D_stack(show_progress_dialog=False)
            self.update_2D_plot()
        except Exception as e:
            print(f"  2D rebuild failed: {e}")

    def _update_wavelength_label(self):
        """Item 5: draw/update a small 'λ = ... Å' label in the top-left of
        the processed plot. Uses a TextItem pinned to the viewbox corner so
        it tracks zoom/pan.

        If the currently-displayed frame has been fitted (either a single
        Le Bail fit or a cached sequential-fit result), append the fit's Rw
        on the next line so the user can see the fit quality at a glance.
        """
        WL = getattr(self, 'WL', None)

        # Create label once and reuse
        if not hasattr(self, '_wl_label') or self._wl_label is None:
            self._wl_label = pg.TextItem(anchor=(0, 0), color='w')
            # pin to top-left of viewbox in pixel coordinates
            try:
                vb = self.plot_processed.getViewBox()
                self._wl_label.setParentItem(vb)
                self._wl_label.setPos(6, 4)
            except Exception:
                self.plot_processed.addItem(self._wl_label)

        # Wavelength portion
        if WL is None:
            wl_str = 'λ = (no PONI loaded)'
        else:
            try:
                wl_val = float(WL)
                wl_str = f'λ = {wl_val:.4f} Å'
            except (TypeError, ValueError):
                wl_str = 'λ = (invalid)'

        # Rw portion: pull from whichever fit corresponds to the displayed frame
        rw_str = ''
        try:
            current_frame = self.ui.spinBox_PatternNumber.value()
            rw_value = self._get_rw_for_frame(current_frame)
            if rw_value is not None:
                rw_str = f'\nRw = {rw_value:.4f}'
        except Exception:
            pass

        text = wl_str + rw_str
        if WL is None:
            self._wl_label.setText(text, color=(200, 200, 200))
        else:
            self._wl_label.setText(text)
        font = self._wl_label.textItem.font()
        font.setPointSize(9)
        self._wl_label.textItem.setFont(font)

    def _get_rw_for_frame(self, frame):
        """Return the Rw value for the given frame index, or None if the
        frame hasn't been fitted yet. Single-fit results take priority over
        sequential-fit results since the user has just run them."""
        # Single-fit overlay (live results from "Fit Le Bail" button)
        fit_frame = getattr(self, 'results_frame_index', None)
        single_results = getattr(self, 'results', None)
        if (single_results is not None
                and fit_frame is not None
                and fit_frame == frame
                and 'Rw' in single_results):
            return single_results['Rw']

        # Cached sequential-fit results
        seq = getattr(self, 'sequentialFitResults', None)
        if seq is not None:
            frame_res = seq.get(f'frame_{frame}')
            if frame_res is not None and 'Rw' in frame_res:
                return frame_res['Rw']
        return None

    # -------------------------------------------------------------------
    # Item 4 helper: make a pyqtgraph plot widget use rectangle-zoom
    # (drag a box with the left mouse button to magnify that area).
    # Right-click context menu and double-click-to-auto-range still work.
    # -------------------------------------------------------------------
    def _enable_rect_zoom(self, plot_widget):
        try:
            vb = plot_widget.getViewBox()
            vb.setMouseMode(pg.ViewBox.RectMode)
        except Exception:
            pass
    
    def spinbox_changed(self, index):
        if not self.file_paths:
            return
    
        # Prevent recursion
        self.ui.listWidget_fileList.blockSignals(True)
        self.ui.listWidget_fileList.setCurrentRow(index)
        self.ui.listWidget_fileList.blockSignals(False)
    
        self.update_plot()
    
    def get_current_file(self):
        # Gets the currently selected file
        current_item = self.ui.listWidget_fileList.currentItem()
    
        if current_item is None:
            return None
    
        return current_item.data(Qt.UserRole)
    
    def run_background(self):
        self.plot_raw.clear()
    
        data_path = self.get_current_file()
        if data_path is None:
            return
    
        method = "ALS" if self.ui.checkBox_ALSfit.isChecked() else "POLY"
    
        params = {
            "twoThetaMin": self.ui.doubleSpinBox_twoThetaMin.value(),
            "twoThetaMax": self.ui.doubleSpinBox_twoThetaMax.value(),
            "lam": self.ui.doubleSpinBox_ALS_lam.value(),
            "p": self.ui.doubleSpinBox_ALS_p.value(),
            "niter": self.ui.doubleSpinBox_ALS_niter.value(),
            "prominence": self.ui.doubleSpinBox_BG_Prominence.value(),
            "height": self.ui.doubleSpinBox_BG_Height.value(),
            "order": self.ui.spinBox_BG_Order.value(),
            "peakwidth": self.ui.doubleSpinBox_BG_width.value(),
        }
    
        result = self.data_manager.get_background(data_path, method, params)
        if result is None:
            return
    
        twoTheta, valueInt, valueBG, valueIntBGsub = result
    
        if self.ui.checkBox_subtractBG.isChecked():
            self.plot_raw.plot(twoTheta, valueIntBGsub, pen='w')
        else:
            self.plot_raw.plot(twoTheta, valueInt, pen='w')
            self.plot_raw.plot(twoTheta, valueBG,
                               pen=pg.mkPen('r', width=2, style=Qt.DashLine))
    
        self.update_processed_plot()
        
        # Allows to access the values from now on.
        self.current_twoTheta = twoTheta
        self.current_valueInt = valueInt
        self.current_valueBG = valueBG
        self.current_valueIntBGsub = valueIntBGsub
        
        # --- Update 2D plot: store this pattern and refresh display ---
        # PERFORMANCE: on a plain scroll the BG cache returns the SAME array
        # that's already sitting in all_patterns_BGsub[index] (initial load
        # via _build_full_2D_stack fills every row up front). In that case
        # the 2D heatmap and its fitted-peak overlays don't need to be
        # touched at all - and skipping the rebuild is what restores fast
        # scrolling once a batch has been fitted, because
        # _replot_2D_peak_overlays() otherwise rebuilds N_frames x N_phases
        # ScatterPlotItems on every arrow-key press.
        index = self.ui.spinBox_PatternNumber.value()
        need_2D_refresh = False
        if not hasattr(self, 'all_patterns_BGsub') or len(self.all_patterns_BGsub) == 0:
            # Initialise or reset when file count changes
            n = len(self.file_paths)
            n_pts = len(valueIntBGsub)
            self.all_patterns_BGsub = np.full((n, n_pts), np.nan)
            need_2D_refresh = True
        if index < len(self.all_patterns_BGsub):
            existing = self.all_patterns_BGsub[index]
            # Row needs writing if it's the wrong length, currently all-NaN
            # (never populated), or genuinely different. np.array_equal with
            # equal_nan=True treats NaN==NaN so an unchanged-but-NaN-filled
            # row would NOT trigger a refresh; we explicitly check for that.
            if (len(existing) != len(valueIntBGsub)
                    or np.isnan(existing).all()
                    or not np.array_equal(existing, valueIntBGsub,
                                          equal_nan=True)):
                self.all_patterns_BGsub[index] = valueIntBGsub
                need_2D_refresh = True
        if need_2D_refresh:
            self.update_2D_plot()
    
    def get_sigma_mode(self):
        """Read the sigma mode checkboxes and return the sigma_mode string."""
        if self.ui.checkBox_fixSigma_2.isChecked():
            return 'fixed'
        elif self.ui.checkBox.isChecked():
            return 'caglioti'
        elif self.ui.checkBox_2.isChecked():
            return 'separate'
        elif self.ui.checkBox_3.isChecked():
            return 'per_phase'
        else:
            return 'caglioti'  # safe fallback

    def get_sigma_bounds(self):
        """Read the Sigma Bounds spinbox (doubleSpinBox_3).
        Returns the multiplier for the upper sigma bound (sigGuess * bounds)."""
        val = self.ui.doubleSpinBox_3.value()
        return val if val > 0 else 2.0  # guard against 0

    def get_amp_prefactor(self):
        """Read the 'Amplitude' spinbox (self.doubleSpinBox, labelled
        'Amplitude' in the GUI). Returns a fractional pre-factor:
        ampGuess = max(data) * prefactor. Default 1.0 means start at the
        peak of the data."""
        val = self.ui.doubleSpinBox.value()
        return val if val > 0 else 1.0  # guard against 0

    def get_amp_bounds(self):
        """Read the 'Amp. Bounds' spinbox (self.doubleSpinBox_fitAmpGuess,
        labelled 'Amp. Bounds' in the GUI). Returns the multiplier for the
        upper amplitude bound (amp_init * bounds). Mirrors get_sigma_bounds."""
        val = self.ui.doubleSpinBox_fitAmpGuess.value()
        return val if val > 0 else 3.0  # guard against 0; default 3.0

    def get_max_shift(self):
        """Read the Max Shift % spinbox (doubleSpinBox_maxShiftFit).
        Returns fractional constraint on lattice parameter bounds (0.0 - 1.0).
        e.g. 0.05 = ±5%, 0.50 = ±50% (default).

        Special case: a value of 0.0 means "fix the lattice" - it is
        passed through verbatim and initialise_parameters_LB will set
        vary=False on every lattice parameter for every active phase.
        """
        val = self.ui.doubleSpinBox_maxShiftFit.value()
        # Negative values are nonsensical; clamp to 0. Positive values
        # are clamped to <= 1.0 (= ±100%) since the widget is a fraction.
        # Zero is preserved on purpose - it triggers the lattice-fix
        # path inside initialise_parameters_LB.
        if val <= 0:
            return 0.0
        return min(val, 1.0)

    def LB_fit_single(self):
        '''
        LB-Fit for ONE pattern only
        '''
        self.results = None          # clear stale fit before starting
        self.results_frame_index = None  # track which pattern this fit belongs to
        self.plot_processed.clear()
    
        # Set used phases (checkbox column 2)
        used_phases = []
    
        for row in range(self.ui.tableWidget_JCPDSTable.rowCount()):
            chk_use = self._get_table_checkbox(
                self.ui.tableWidget_JCPDSTable, row, 2)
            if chk_use is not None and chk_use.isChecked():
                used_phases.append(row)
    
        if len(used_phases) == 0:
            print("No phases selected for fitting.")
            return
    
        self.phases['phases_used'] = used_phases
    

        # Set pressure + update unit cells
        for phase_index in used_phases:
    
            spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 3)
            P = spin_pressure.value()
    
            UC0 = self.phases[f'phase_{phase_index}_unit_cell_0']
            comp = self.phases[f'phase_{phase_index}_compression_constants']
            V0, K0, K0P = comp[0], comp[1], comp[2]
    
            UC_at_P = EoS.find_UC_at_P(UC0, P, V0, K0, K0P)
            self.phases[f'phase_{phase_index}_unit_cell'] = UC_at_P
    
    
        # Get frame number
        frame = self.ui.spinBox_PatternNumber.value()
    
        # Get fit guesses.
        # Note: ampGuess passed to fit_LB is ignored inside fit_LB (it
        # rebuilds the guess as max(data) * ampPrefactor). The actual
        # amplitude guess control is the 'Amplitude' spinbox via
        # get_amp_prefactor(); the 'Amp. Bounds' spinbox controls the
        # upper bound via get_amp_bounds().
        sigGuess = self.ui.doubleSpinBox_fitSigGuess.value()
    
        twoThetaMin = self.ui.doubleSpinBox_twoThetaMin.value()
        twoThetaMax = self.ui.doubleSpinBox_twoThetaMax.value()
    
    
        # Run fit
        results = exc.fit_LB(
            self.current_twoTheta,
            self.current_valueIntBGsub,
            self.phases,
            self.WL,
            frame=frame,
            twoThetaMin=twoThetaMin,
            twoThetaMax=twoThetaMax,
            framePressureGuess=[],
            sigGuess=sigGuess,
            sigma_mode=self.get_sigma_mode(),
            sigmaBounds=self.get_sigma_bounds(),
            ampBounds=self.get_amp_bounds(),
            ampPrefactor=self.get_amp_prefactor(),
            maxShift=self.get_max_shift(),
            excludePeakList=[]
        )

        # ----------------------------------------------------------------
        # Did the fit actually produce a usable result? fit_LB swallows
        # exceptions internally and returns {} when the optimiser blew up
        # (e.g. KeyError inside the residual). Without this guard the GUI
        # used to print "Fitting for one pattern successful" and then
        # immediately crash on `self.results['data_fit']`.
        # ----------------------------------------------------------------
        if not results or 'data_fit' not in results:
            print("Fit failed - no usable result produced. "
                  "See the error printed by fit_LB above for details.")
            self.results = None
            self.results_frame_index = None
            try:
                QMessageBox.warning(self, "Fit failed",
                                    "The Le Bail fit did not produce a "
                                    "usable result for this pattern.\n\n"
                                    "Check the console for the underlying "
                                    "error. Common causes: a phase's HKL "
                                    "list contains reflections far outside "
                                    "the 2-theta window, lattice / sigma "
                                    "guesses are wildly off, or one of the "
                                    "spin-box bounds is too tight.")
            except Exception:
                pass
            # Re-draw the BG-subtracted data so the plot isn't blank.
            try:
                self.plot_processed.clear()
                self.plot_processed.plot(self.current_twoTheta,
                                         self.current_valueIntBGsub,
                                         pen=pg.mkPen('w', width=1),
                                         name='BG-subtracted data')
            except Exception:
                pass
            return

        # Store results, tagged to the current pattern index
        self.results = results
        self.results_frame_index = self.ui.spinBox_PatternNumber.value()
        print("Fitting for one pattern successful")
        
        # Overlay fit on plot
        self.plot_processed.clear()
        
        # 1) BG-subtracted data (white)
        self.plot_processed.plot(self.current_twoTheta,
                                 self.current_valueIntBGsub,
                                 pen=pg.mkPen('w', width=1),
                                 name='BG-subtracted data')
        
        # 2) Fitted envelope (red)
        twoTheta_fit, bestFit = self.results['data_fit']
        self.plot_processed.plot(twoTheta_fit,
                                 bestFit,
                                 pen=pg.mkPen('r', width=2),
                                 name='Fit')
        
        # 3) Optional: difference (blue dashed) - toggled by Difference checkbox
        if self.ui.checkBox_differencePlot.isChecked():
            diff = self.current_valueIntBGsub - bestFit
            offset = np.min(self.current_valueIntBGsub) * 0.5
            self.plot_processed.plot(twoTheta_fit,
                                     diff + offset,
                                     pen=pg.mkPen('b', width=1, style=Qt.DashLine),
                                     name='Difference')

        # Refresh the lambda/Rw label so the freshly-computed Rw is shown
        try:
            self._update_wavelength_label()
        except Exception:
            pass
    
        
    # def LB_fit_sequential(self):
    
    #     print("Starting sequential refinement...")
    
    #     sequentialFitResults = {}
    #     sequentialFitResultsPeaks = {}
    
    #     start_frame = 0
    #     end_frame = len(self.file_paths) - 1
    
    #     for frame in range(start_frame, end_frame + 1):
    
    #         print(f"Refining frame {frame}")
    
    #         # Set current frame in UI (important if your code depends on it)
    #         self.ui.spinBox_PatternNumber.setValue(frame)
    
    #         # ---- Determine used phases for this frame ----
    #         used_phases = []
    
    #         for phase_index in range(len(self.phases_list)):
    
    #             checkbox_use = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 1)
    #             checkbox_refine = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 2)
    
    #             frame_start = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 4).value()
    #             frame_end = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 5).value()
    
    #             if checkbox_use.isChecked() and checkbox_refine.isChecked():
    #                 if frame_start <= frame <= frame_end:
    #                     used_phases.append(phase_index)
    
    #         self.phases['phases_used'] = used_phases
    
    #         # ---- Update pressure + unit cells before fit ----
    #         for phase_index in used_phases:
    
    #             spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 3)
    #             P = spin_pressure.value()
    
    #             UC0 = self.phases[f'phase_{phase_index}_unit_cell']
    #             comp = self.phases[f'phase_{phase_index}_compression_constants']
    #             V0, K0, K0P = comp[0], comp[1], comp[2]
    
    #             UC_at_P = EoS.find_UC_at_P(UC0, P, V0, K0, K0P)
    #             self.phases[f'phase_{phase_index}_unit_cell'] = UC_at_P
    
    #         # ---- Run Fit ----
    #         results = exc.fit_LB(
    #             self.current_twoTheta,
    #             self.current_valueIntBGsub,
    #             self.phases,
    #             self.WL,
    #             frame=frame,
    #             twoThetaMin=self.ui.doubleSpinBox_twoThetaMin.value(),
    #             twoThetaMax=self.ui.doubleSpinBox_twoThetaMax.value(),
    #             framePressureGuess=[],
    #             ampGuess=self.ui.doubleSpinBox_fitAmpGuess.value(),
    #             sigGuess=self.ui.doubleSpinBox_fitSigGuess.value(),
    #             excludePeakList=[]
    #         )
    
    #         # ---- Store Results ----
    #         sequentialFitResults[f'frame_{frame}'] = {}
    #         sequentialFitResultsPeaks[f'frame_{frame}'] = {}
    
    #         for phase in used_phases:
    
    #             UC_fit = results[f'phase_{phase}_unit_cell_fit']
    #             peaks = results[f'phase_{phase}_peakPosition_fit']
    
    #             sequentialFitResults[f'frame_{frame}'][f'phase_{phase}'] = UC_fit
    #             sequentialFitResultsPeaks[f'frame_{frame}'][f'phase_{phase}'] = peaks
    
    #             # IMPORTANT: update phase UC for next frame
    #             self.phases[f'phase_{phase}_unit_cell'] = UC_fit
    
    #         # ---- Optional: update plots ----
    #         self.update_plot()
    #         self.plot_peaks_on_2D(sequentialFitResultsPeaks)
    
    #     self.sequentialFitResults = sequentialFitResults
    #     self.sequentialFitResultsPeaks = sequentialFitResultsPeaks
    
    #     print("Sequential refinement complete.")
    
    def LB_fit_sequential(self):
        print("Starting sequential refinement...")

        if not self._prepare_sequential_run(mode='sequential'):
            return
        self._seq_timer.start(10)  # 10 ms delay between frames

    # -------------------------------------------------------------------------
    # Pressure Walk Fitting
    # -------------------------------------------------------------------------
    def pressureWalk_fit(self):
        """
        Sequential fit where each frame's starting unit cell is predicted from
        the previous frame's fitted pressure + a user-supplied deltaP step.
        The FIRST frame is seeded from the per-phase JCPDS-table pressure.
        """
        print("Starting Pressure Walk refinement...")
        if not self._prepare_sequential_run(mode='pressureWalk'):
            return
        # delta P is read from doubleSpinBox_7 (label "Delta P" on the
        # Pressure Walk row)
        self._seq_deltaP = self.ui.doubleSpinBox_7.value()
        self._seq_timer.start(10)

    # -------------------------------------------------------------------------
    # Pressure Guess Fitting
    # -------------------------------------------------------------------------
    def pressureGuess_fit(self):
        """
        Each phase has its OWN [frame, P, fix] table (one tab per phase
        in tabWidget). For every frame we interpolate
        per-phase to get a starting pressure and (via the phase's EoS)
        a starting unit cell.  If a row in a phase's table has its
        'Fix' checkbox ticked, the lattice parameters of THAT phase
        are held fixed for the frame matched to that row.
        """
        print("Starting Pressure Guess refinement...")

        # Build self._framePressureGuessByPhase from every per-phase tab.
        # phase_index -> sorted list[(frame, P, fix)]
        self._framePressureGuessByPhase = {}
        for phase_index, tbl in self._phase_pg_tables.items():
            rows = self._snapshot_table_rows(tbl)
            if rows:
                rows.sort(key=lambda r: r[0])
                self._framePressureGuessByPhase[phase_index] = rows

        # Diagnostic: dump the captured per-phase tables so the user can
        # confirm the 'fix' checkboxes were read correctly.
        if self._framePressureGuessByPhase:
            print("[pressure-guess] captured tables:")
            for pi, rows in self._framePressureGuessByPhase.items():
                name = self.phases.get(f'phase_{pi}_name', f'phase {pi}')
                print(f"  phase {pi} ({name}):")
                for fr, P, fix in rows:
                    fix_str = 'FIX' if fix else '   '
                    print(f"    frame={fr:5d}  P={P:7.3f} GPa  {fix_str}")
        else:
            print("[pressure-guess] no per-phase rows captured.")

        # Need at least one phase with at least 2 rows in order to
        # interpolate across frames; otherwise fall back to the JCPDS
        # table pressure as a constant (handled in _process_next_frame).
        any_interpolatable = any(
            len(rows) >= 2 for rows in self._framePressureGuessByPhase.values())
        if not any_interpolatable:
            print("Pressure Guess needs at least one phase with >=2 rows in "
                  "its tab. Use the green '+' button to add rows.")
            return

        # Backwards-compat scaffold: keep _framePressureGuess populated
        # with the FIRST used phase's rows so any legacy code paths that
        # still read it don't crash. It is not used by the per-phase
        # branch in _process_next_frame.
        first_pi = next(iter(self._framePressureGuessByPhase))
        self._framePressureGuess = [
            [int(r[0]), float(r[1])]
            for r in self._framePressureGuessByPhase[first_pi]
        ]

        if not self._prepare_sequential_run(mode='pressureGuess'):
            return
        self._seq_timer.start(10)

    # -------------------------------------------------------------------------
    # Shared setup for any of the three batch-fit entry points
    # -------------------------------------------------------------------------
    def _prepare_sequential_run(self, mode):
        """Common guards / state reset used by sequential, pressureWalk and
        pressureGuess.  Returns False if the run cannot start (e.g. no data
        loaded, no JCPDS loaded, no wavelength)."""
        if not self.file_paths:
            QtWidgets.QMessageBox.warning(
                self, "Cannot start fit", "No data files loaded.")
            return False
        if not hasattr(self, 'phases') or self.phases is None or \
           self.ui.tableWidget_JCPDSTable.rowCount() == 0:
            QtWidgets.QMessageBox.warning(
                self, "Cannot start fit", "No JCPDS phases loaded.")
            return False
        if not hasattr(self, 'WL') or self.WL is None:
            QtWidgets.QMessageBox.warning(
                self, "Cannot start fit",
                "No PONI / wavelength loaded. Click 'Load PONI' first.")
            return False

        # Item 7 / fail-fast: verify at least one phase is marked "Use".
        # Fast check once here is vastly cheaper than discovering it frame by
        # frame inside _process_next_frame.
        any_used = False
        for r in range(self.ui.tableWidget_JCPDSTable.rowCount()):
            chk = self._get_table_checkbox(
                self.ui.tableWidget_JCPDSTable, r, 2)
            if chk is not None and chk.isChecked():
                any_used = True
                break
        if not any_used:
            QtWidgets.QMessageBox.warning(
                self, "Cannot start fit",
                "No phases are ticked 'Use' in the JCPDS table.")
            return False

        self.setEnabled(False)
        self.sequentialFitResults = {}
        self.sequentialFitResultsPeaks = {}
        self._clear_2D_overlays()

        self._seq_current_frame = 0
        self._seq_end_frame = len(self.file_paths) - 1
        self._seq_mode = mode

        # Item 7: snapshot every parameter the hot loop needs, ONCE. Reading
        # widget values is surprisingly costly when done 1000s of times.
        self._seq_tt_min        = self.ui.doubleSpinBox_twoThetaMin.value()
        self._seq_tt_max        = self.ui.doubleSpinBox_twoThetaMax.value()
        self._seq_sig_guess     = self.ui.doubleSpinBox_fitSigGuess.value()
        self._seq_sigma_mode    = self.get_sigma_mode()
        self._seq_sigma_bounds  = self.get_sigma_bounds()
        self._seq_amp_bounds    = self.get_amp_bounds()
        self._seq_amp_prefactor = self.get_amp_prefactor()
        self._seq_max_shift     = self.get_max_shift()
        self._seq_bg_method     = "ALS" if self.ui.checkBox_ALSfit.isChecked() else "POLY"
        self._seq_bg_params     = {
            "twoThetaMin": self._seq_tt_min,
            "twoThetaMax": self._seq_tt_max,
            "lam":         self.ui.doubleSpinBox_ALS_lam.value(),
            "p":           self.ui.doubleSpinBox_ALS_p.value(),
            "niter":       self.ui.doubleSpinBox_ALS_niter.value(),
            "prominence":  self.ui.doubleSpinBox_BG_Prominence.value(),
            "height":      self.ui.doubleSpinBox_BG_Height.value(),
            "order":       self.ui.spinBox_BG_Order.value(),
            "peakwidth":   self.ui.doubleSpinBox_BG_width.value(),
        }

        # Item 3: progress dialog
        self._show_progress_dialog(mode, self._seq_end_frame + 1)
        self._seq_start_time = time.monotonic()
        return True

    def _show_progress_dialog(self, mode, n_total):
        """Create and show a modal progress dialog with ETA + Cancel."""
        self._progress_cancelled = False

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Refinement running - {mode}")
        dlg.setModal(True)
        dlg.setMinimumWidth(380)
        # Prevent close via the X button - user must use Cancel (or wait)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)

        lay = QtWidgets.QVBoxLayout(dlg)

        lbl = QtWidgets.QLabel(f"Fitting {n_total} frames ({mode})")
        lay.addWidget(lbl)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, n_total)
        bar.setValue(0)
        lay.addWidget(bar)

        eta_lbl = QtWidgets.QLabel("Frame 0 of %d   |   elapsed 0 s   |   ETA -" % n_total)
        lay.addWidget(eta_lbl)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; "
            "font-weight: bold; padding: 6px 18px; border-radius: 4px; } "
            "QPushButton:hover { background-color: #c0392b; }")
        btn_row.addWidget(cancel_btn)
        lay.addLayout(btn_row)

        cancel_btn.clicked.connect(self._cancel_from_progress)

        self._progress_dialog    = dlg
        self._progress_bar       = bar
        self._progress_eta_label = eta_lbl
        self._progress_total     = n_total

        dlg.show()

    def _cancel_from_progress(self):
        """User clicked Cancel on the progress dialog."""
        self._progress_cancelled = True
        self.stop_fitting()
        # _close_progress_dialog is called from stop_fitting via _finish_run

    def _update_progress_dialog(self, done):
        """Update the progress bar + ETA. 'done' = number of frames completed."""
        if not hasattr(self, '_progress_bar') or self._progress_bar is None:
            return
        n_total = self._progress_total
        elapsed = time.monotonic() - self._seq_start_time
        self._progress_bar.setValue(done)
        if done > 0:
            per_frame = elapsed / done
            remaining = per_frame * (n_total - done)
            eta_str = self._format_seconds(remaining)
        else:
            eta_str = "-"
        self._progress_eta_label.setText(
            f"Frame {done} of {n_total}   |   elapsed "
            f"{self._format_seconds(elapsed)}   |   ETA {eta_str}")
        QtWidgets.QApplication.processEvents()   # keep UI responsive

    @staticmethod
    def _format_seconds(s):
        s = max(0.0, float(s))
        if s < 60:
            return f"{s:4.1f} s"
        m, sec = divmod(int(s), 60)
        if m < 60:
            return f"{m:d} m {sec:02d} s"
        h, m = divmod(m, 60)
        return f"{h:d} h {m:02d} m"

    def _close_progress_dialog(self):
        """Close and clear the progress dialog safely."""
        dlg = getattr(self, '_progress_dialog', None)
        if dlg is not None:
            try:
                dlg.close()
                dlg.deleteLater()
            except Exception:
                pass
        self._progress_dialog    = None
        self._progress_bar       = None
        self._progress_eta_label = None
    
    def plot_peaks_on_2D(self, frame, used_phases):
        for phase in used_phases:
    
            peaks = self.sequentialFitResultsPeaks[f'frame_{frame}'][f'phase_{phase}']
    
            for peak in peaks:
                line = pg.InfiniteLine(
                    pos=peak,
                    angle=90,
                    pen=pg.mkPen('r', width=1)
                )
                self.plotWidget_2D.addItem(line)
            
            
            
            
    def save_single_data(self):
        """
        Save the currently displayed fit as a single whitespace-delimited
        text file. Columns: TwoTheta, Intensity, Intensity-BG, BG, Fit,
        followed by one tickmark column per used phase.
        """
        if not hasattr(self, 'results'):
            print("No fit results available.")
            return
        if not hasattr(self, 'current_file_index'):
            print("No pattern loaded.")
            return

        # Default file name based on the loaded pattern
        orig_path = self.file_paths[self.current_file_index]
        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        default_name = f"{base_name}_fit.txt"

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save fit",
            default_name,
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if not save_path:
            return  # user cancelled

        # Pick delimiter from the chosen filter / extension
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.csv' or 'CSV' in selected_filter:
            delimiter = ','
            if ext == '':
                save_path += '.csv'
        else:
            delimiter = '\t'  # tab-delimited .txt - matches the .xy/.dat/.chi style
            if ext == '':
                save_path += '.txt'

        # Pull the displayed columns. self.current_valueBG is the
        # background that was subtracted to produce current_valueIntBGsub
        # (set during update_processed_plot); using it here means the
        # exported file actually round-trips: Intensity == BGsub + BG.
        twoTheta = np.asarray(self.current_twoTheta)
        intensity = np.asarray(getattr(self, 'current_valueInt', np.zeros_like(twoTheta)))
        BG_sub = np.asarray(self.current_valueIntBGsub)
        BG = np.asarray(getattr(self, 'current_valueBG', np.zeros_like(twoTheta)))
        fit = np.asarray(self.results['data_fit'][1])

        # Tickmark columns - one per used phase, NaN-padded so all columns
        # share the same length as twoTheta.
        tickmark_cols = []
        tick_headers = []
        if hasattr(self, 'phases') and self.phases is not None:
            for phase in self.phases.get('phases_used', []):
                key = f'phase_{phase}_peakPosition_fit'
                tick_padded = np.full_like(twoTheta, np.nan, dtype=float)
                if key in self.results:
                    tick = np.asarray(self.results[key], dtype=float)
                    n = min(len(tick), len(tick_padded))
                    tick_padded[:n] = tick[:n]
                tickmark_cols.append(tick_padded)
                tick_headers.append(f"Phase{phase}_Tick")

        # Stack and write
        data_array = np.column_stack([twoTheta, intensity, BG_sub, BG, fit] + tickmark_cols)
        header_cols = ["TwoTheta", "Intensity", "Intensity-BG", "BG", "Fit"] + tick_headers
        header = delimiter.join(header_cols)

        np.savetxt(save_path, data_array, delimiter=delimiter,
                   header=header, comments='')
        print(f"Saved fit to: {save_path}")

        # ------------------------------------------------------------------
        # ALSO save lattice parameters (*_lattParam.txt) and per-phase
        # reflection tables (*_fitParam.txt) in the same per-frame format
        # used by save_all_results, so single-fit and batch-fit outputs
        # can be concatenated / compared directly.
        # ------------------------------------------------------------------
        # Strip a trailing "_fit" from base_name so we don't get
        # "..._fit_lattParam.txt" / "..._fit_fitParam.txt".
        stem = base_name[:-4] if base_name.endswith('_fit') else base_name
        save_dir = os.path.dirname(save_path)
        ext = os.path.splitext(save_path)[1] or '.txt'
        frame_idx = int(getattr(self, 'current_file_index', -1))

        latt_path = os.path.join(save_dir, f'{stem}_lattParam{ext}')
        try:
            self._write_lattParam_file(latt_path, frame_idx,
                                       self.results, self.phases,
                                       delimiter)
        except Exception as e:
            print(f"  Could not save lattice parameters: {e}")

        fitp_path = os.path.join(save_dir, f'{stem}_fitParam{ext}')
        try:
            WL = float(self.results.get('_WL',
                                        getattr(self, 'WL', np.nan)))
            tt_min = float(self.results.get(
                '_twoThetaMin',
                self.ui.doubleSpinBox_twoThetaMin.value()))
            tt_max = float(self.results.get(
                '_twoThetaMax',
                self.ui.doubleSpinBox_twoThetaMax.value()))
            self._write_fitParam_file(fitp_path, frame_idx,
                                      self.results, self.phases,
                                      WL, tt_min, tt_max, delimiter)
        except Exception as e:
            print(f"  Could not save reflection table: {e}")

    def _write_lattParam_file(self, out_path, frame, results, phases, delimiter):
        """Write a one-row-per-phase lattice-parameters file.

        Columns: Frame, Pressure_GPa, a, b, c, alpha, beta, gamma,
                 da, db, dc, dalpha, dbeta, dgamma, Rw

        Format is identical to the aggregated *_sequential_results.csv that
        save_all_results produces, so per-frame and aggregated files are
        interoperable (one row per phase per frame either way).
        """
        if results is None:
            print("  Skipping lattice-parameter save: no fit results.")
            return
        if phases is None:
            print("  Skipping lattice-parameter save: no JCPDS loaded.")
            return

        rw = float(results.get('Rw', np.nan))

        rows = []
        for phase_index in phases.get('phases_used', []):
            UC = results.get(f'phase_{phase_index}_unit_cell_fit')
            if UC is None:
                continue
            UC = list(UC)
            err_arr = results.get(f'phase_{phase_index}_unit_cell_error',
                                  np.full(6, np.nan, dtype=float))
            err_arr = list(np.asarray(err_arr, dtype=float))
            comp = phases[f'phase_{phase_index}_compression_constants']
            V0, K0, K0P = comp[0], comp[1], comp[2]
            V = EoS.unitCellVolume(UC)
            P = float(EoS.BM3_EOS(V, V0, K0, K0P))

            rows.append([float(frame), P] + UC + err_arr + [rw])

        if not rows:
            print("  Skipping lattice-parameter save: no fitted phases.")
            return

        header_cols = ['Frame', 'Pressure_GPa',
                       'a', 'b', 'c', 'alpha', 'beta', 'gamma',
                       'da', 'db', 'dc', 'dalpha', 'dbeta', 'dgamma',
                       'Rw']
        header = delimiter.join(header_cols)
        np.savetxt(out_path, np.asarray(rows, dtype=float),
                   delimiter=delimiter, header=header, comments='',
                   fmt='%.6f')
        print(f"Saved lattice parameters to: {out_path}")

    def _write_fitParam_file(self, out_path, frame, results, phases,
                             WL, twoThetaMin, twoThetaMax, delimiter):
        """Write a reflection table grouped by phase.

        File layout - one block per phase in phases['phases_used']:
            # Phase <idx>: <name>
            # a=...  b=...  c=...  alpha=... beta=... gamma=... V=... P=... GPa  Rw=...
            #   h   k   l   2theta_fit   2theta_ref   d(A)   Q(A^-1)   FWHM   sig_FWHM   I_int   sig_I_int
            <one row per visible reflection>
            <blank line>

        Where:
          2theta_fit  : peak position from the fit (from peakPosition_fit)
          2theta_ref  : peak position predicted from phase['unit_cell_0']
                        (the JCPDS reference cell) - lets the user see
                        the shift due to pressure / strain at a glance.
          d, Q        : computed from 2theta_fit and the fit wavelength
          FWHM        : 2*sqrt(2*ln2) * sigma, in degrees 2theta
          I_int       : amp * sigma * sqrt(2*pi), the analytic Gaussian
                        integral (assumes sigma in 2theta degrees).
          sig_*       : standard errors propagated from lmfit covariance.

        Errors that lmfit could not estimate (e.g. fixed parameter or
        ill-conditioned fit) come out as NaN in the file.
        """
        if results is None:
            print("  Skipping fitParam save: no fit results.")
            return
        if phases is None:
            print("  Skipping fitParam save: no JCPDS loaded.")
            return

        lmfit_params = results.get('_lmfit_params')
        if lmfit_params is None:
            print("  Skipping fitParam save: per-peak fit info not cached "
                  "(re-run the fit to enable).")
            return
        sigma_mode = results.get('_sigma_mode', 'caglioti')

        # 2.3548 = 2*sqrt(2*ln(2))
        FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))
        SQRT_2PI    = np.sqrt(2.0 * np.pi)
        DEG2RAD     = np.pi / 180.0

        col_headers = ['h', 'k', 'l',
                       '2theta_fit', '2theta_ref',
                       'd(A)', 'Q(A^-1)',
                       'FWHM_deg', 'sig_FWHM',
                       'I_int', 'sig_I_int']

        # Write the file ourselves (not via np.savetxt) so we can mix
        # per-phase headers with the data tables.
        with open(out_path, 'w') as f:
            f.write(f'# Per-reflection fit parameters - frame {frame}\n')
            f.write(f'# Wavelength = {WL:.6f} A\n')
            f.write(f'# Sigma mode = {sigma_mode}\n')
            f.write(f'# Rw         = {results.get("Rw", float("nan")):.6f}\n')
            f.write('#\n')

            for phase_idx in phases.get('phases_used', []):
                # Header block for this phase: name + lattice + V + P + Rw
                phase_name = phases.get(f'phase_{phase_idx}_name',
                                        f'phase_{phase_idx}')
                UC = results.get(f'phase_{phase_idx}_unit_cell_fit')
                if UC is None:
                    continue
                UC = list(UC)
                err_arr = list(np.asarray(
                    results.get(f'phase_{phase_idx}_unit_cell_error',
                                np.full(6, np.nan, dtype=float)),
                    dtype=float))
                V = float(EoS.unitCellVolume(UC))
                comp = phases[f'phase_{phase_idx}_compression_constants']
                V0, K0, K0P = comp[0], comp[1], comp[2]
                P = float(EoS.BM3_EOS(V, V0, K0, K0P))

                f.write(f'# Phase {phase_idx}: {phase_name}\n')
                f.write(
                    f'# a={UC[0]:.5f}({err_arr[0]:.5f})  '
                    f'b={UC[1]:.5f}({err_arr[1]:.5f})  '
                    f'c={UC[2]:.5f}({err_arr[2]:.5f})  '
                    f'alpha={UC[3]:.4f}({err_arr[3]:.4f})  '
                    f'beta={UC[4]:.4f}({err_arr[4]:.4f})  '
                    f'gamma={UC[5]:.4f}({err_arr[5]:.4f})  '
                    f'V={V:.4f}  P={P:.4f} GPa\n')
                f.write('# ' + delimiter.join(col_headers) + '\n')

                # Build the in-window reflection list directly from the
                # fitted unit cell so we get the full [2theta, H, K, L]
                # rows. (results['phase_X_peakPosition_fit'] is only the
                # 2theta column - the HKL identity has to be recovered
                # from a fresh reflection_List call, which is cheap.)
                crystal_system = phases.get(
                    f'phase_{phase_idx}_crystal_system', 'TRICLINIC')
                try:
                    _, peak_rows = tb.reflection_List(
                        list(UC), phases[f'phase_{phase_idx}_HKL'],
                        WL,
                        twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                        crystal_system=crystal_system)
                except Exception as e:
                    print(f"  fitParam: could not compute reflection list "
                          f"for phase {phase_idx}: {e}")
                    f.write('\n')
                    continue
                if peak_rows is None or len(peak_rows) == 0:
                    f.write('\n')
                    continue
                peaks = np.asarray(peak_rows, dtype=float)

                # Reference 2theta from JCPDS card unit cell
                UC0 = phases.get(f'phase_{phase_idx}_unit_cell_0', None)
                tt_ref_lookup = {}
                if UC0 is not None:
                    try:
                        # Run with a permissive window so all visible
                        # HKLs round-trip even if the cell has shifted
                        # them out of the user's fit window.
                        _, ref_rows = tb.reflection_List(
                            list(UC0), phases[f'phase_{phase_idx}_HKL'],
                            WL,
                            twoThetaMin=0.0, twoThetaMax=180.0,
                            crystal_system=crystal_system)
                        for row in ref_rows:
                            key = (int(round(row[1])),
                                   int(round(row[2])),
                                   int(round(row[3])))
                            tt_ref_lookup[key] = float(row[0])
                    except Exception as e:
                        print(f"  fitParam: could not compute reference "
                              f"2theta for phase {phase_idx}: {e}")

                # HKL-index map (for amp/sig lookup by HKL triplet)
                HKL_full = np.asarray(phases[f'phase_{phase_idx}_HKL'])
                hkl_to_idx = {(int(round(h)), int(round(k)), int(round(l))): i
                              for i, (h, k, l) in enumerate(HKL_full[:, :3])}

                for row in peaks:
                    cen = float(row[0])
                    h, k, l = int(round(row[1])), int(round(row[2])), int(round(row[3]))
                    hkl_idx = hkl_to_idx.get((h, k, l))
                    if hkl_idx is None:
                        # Reflection not in master HKL list - shouldn't
                        # happen, but skip rather than crash.
                        continue

                    # d-spacing and Q from the fitted 2theta
                    sin_th = np.sin(cen / 2.0 * DEG2RAD)
                    if sin_th > 0:
                        d_val = WL / (2.0 * sin_th)
                        Q_val = 2.0 * np.pi / d_val
                    else:
                        d_val = np.nan
                        Q_val = np.nan

                    # Reference 2theta from JCPDS unit cell
                    tt_ref = tt_ref_lookup.get((h, k, l), np.nan)

                    # Amplitude and its standard error
                    amp_key = f'amp_{phase_idx}_{hkl_idx + 1}'
                    if amp_key in lmfit_params:
                        amp_p   = lmfit_params[amp_key]
                        amp_val = float(amp_p.value)
                        amp_err = (float(amp_p.stderr)
                                   if amp_p.stderr is not None else np.nan)
                    else:
                        amp_val = np.nan
                        amp_err = np.nan

                    # Sigma and its standard error. For modes where sigma
                    # is a derived/shared quantity (caglioti, fixed,
                    # per_phase) we use the underlying global/per-phase
                    # parameter's stderr scaled appropriately.
                    sig_val, sig_err = self._get_sigma_and_err(
                        lmfit_params, sigma_mode, phase_idx, hkl_idx,
                        cen, DEG2RAD)

                    # Integrated intensity (Gaussian area) and propagated
                    # error assuming uncorrelated amp/sigma errors.
                    if (np.isfinite(amp_val) and np.isfinite(sig_val)
                            and sig_val > 0):
                        I_int = amp_val * sig_val * SQRT_2PI
                        rel = 0.0
                        if (np.isfinite(amp_err) and amp_val != 0):
                            rel += (amp_err / amp_val) ** 2
                        if (np.isfinite(sig_err) and sig_val != 0):
                            rel += (sig_err / sig_val) ** 2
                        I_err = abs(I_int) * np.sqrt(rel) if rel > 0 else np.nan
                    else:
                        I_int = np.nan
                        I_err = np.nan

                    FWHM = FWHM_FACTOR * sig_val if np.isfinite(sig_val) else np.nan
                    FWHM_err = (FWHM_FACTOR * sig_err
                                if np.isfinite(sig_err) else np.nan)

                    fields = [
                        f'{h:d}', f'{k:d}', f'{l:d}',
                        f'{cen:.6f}',
                        f'{tt_ref:.6f}' if np.isfinite(tt_ref) else 'nan',
                        f'{d_val:.6f}' if np.isfinite(d_val) else 'nan',
                        f'{Q_val:.6f}' if np.isfinite(Q_val) else 'nan',
                        f'{FWHM:.6f}'   if np.isfinite(FWHM)   else 'nan',
                        f'{FWHM_err:.6f}' if np.isfinite(FWHM_err) else 'nan',
                        f'{I_int:.6f}'  if np.isfinite(I_int)  else 'nan',
                        f'{I_err:.6f}'  if np.isfinite(I_err)  else 'nan',
                    ]
                    f.write(delimiter.join(fields) + '\n')

                f.write('\n')  # blank line between phase blocks

        print(f"Saved reflection table to: {out_path}")

    @staticmethod
    def _get_sigma_and_err(params, sigma_mode, phase, hkl_idx, cen, deg_to_rad):
        """Return (sigma_value, sigma_stderr) for one peak.

        Mirrors get_sigma() in BatchFit_toolbox but also returns the
        propagated standard error for the value used at this peak. For
        caglioti mode the error scales as 1/cos(theta) the same way the
        value does.
        """
        def _stderr(p):
            return float(p.stderr) if p.stderr is not None else np.nan

        if sigma_mode == 'fixed':
            p = params['sig_global']
            return float(p.value), _stderr(p)

        if sigma_mode == 'per_phase':
            p = params[f'sig_{phase}']
            return float(p.value), _stderr(p)

        if sigma_mode == 'caglioti':
            theta = cen / 2.0 * deg_to_rad
            cos_th = np.cos(theta) if np.cos(theta) != 0 else np.nan
            p = params['sig_global']
            val = float(p.value) / cos_th
            err = _stderr(p) / cos_th if np.isfinite(_stderr(p)) else np.nan
            return val, err

        if sigma_mode == 'separate':
            key = f'sig_{phase}_{hkl_idx + 1}'
            if key not in params:
                # Fall back to highest available index for this phase
                existing = [k for k in params
                            if k.startswith(f'sig_{phase}_')]
                if not existing:
                    return np.nan, np.nan
                key = sorted(existing,
                             key=lambda k: int(k.rsplit('_', 1)[-1]))[-1]
            p = params[key]
            return float(p.value), _stderr(p)

        return np.nan, np.nan
    
    def update_processed_plot(self):
        self.plot_processed.clear()

        data_path = self.get_current_file()
        if data_path is None:
            return

        method = "ALS" if self.ui.checkBox_ALSfit.isChecked() else "POLY"

        params = {
            "twoThetaMin": self.ui.doubleSpinBox_twoThetaMin.value(),
            "twoThetaMax": self.ui.doubleSpinBox_twoThetaMax.value(),
            "lam": self.ui.doubleSpinBox_ALS_lam.value(),
            "p": self.ui.doubleSpinBox_ALS_p.value(),
            "niter": self.ui.doubleSpinBox_ALS_niter.value(),
            "prominence": self.ui.doubleSpinBox_BG_Prominence.value(),
            "height": self.ui.doubleSpinBox_BG_Height.value(),
            "order": self.ui.spinBox_BG_Order.value(),
            "peakwidth": self.ui.doubleSpinBox_BG_width.value(),
        }

        result = self.data_manager.get_background(data_path, method, params)
        if result is None:
            return

        twoTheta, valueInt, valueBG, valueIntBGsub = result
        self.plot_processed.plot(twoTheta, valueIntBGsub, pen='w')

        # --- ITEM 5: wavelength label (top-left) ---
        self._update_wavelength_label()

        # --- Redraw fit overlay ---
        # Priority: single-fit result for this frame first, then sequential
        # cache for this frame. Only one is shown at a time.
        current_frame = self.ui.spinBox_PatternNumber.value()
        fit_drawn     = False

        fit_frame = getattr(self, 'results_frame_index', None)
        if hasattr(self, 'results') and self.results is not None and current_frame == fit_frame:
            try:
                twoTheta_fit, bestFit = self.results['data_fit']
                self.plot_processed.plot(twoTheta_fit, bestFit,
                                         pen=pg.mkPen('r', width=2))
                if self.ui.checkBox_differencePlot.isChecked():
                    diff = self.current_valueIntBGsub - bestFit
                    offset = np.min(self.current_valueIntBGsub) * 0.5
                    self.plot_processed.plot(twoTheta_fit, diff + offset,
                                             pen=pg.mkPen('b', width=1,
                                                          style=Qt.DashLine))
                fit_drawn = True
            except (KeyError, TypeError, ValueError):
                pass

        # Item 2: Cached sequential-fit curve for this frame
        if not fit_drawn:
            frame_key = f'frame_{current_frame}'
            frame_res = self.sequentialFitResults.get(frame_key)
            if frame_res and '_data_fit' in frame_res:
                try:
                    tt_fit, bestFit = frame_res['_data_fit']
                    self.plot_processed.plot(tt_fit, bestFit,
                                             pen=pg.mkPen('r', width=2))
                    if self.ui.checkBox_differencePlot.isChecked() and \
                       '_data_BGsub' in frame_res:
                        _, bgsub_cached = frame_res['_data_BGsub']
                        # align lengths defensively
                        n = min(len(bestFit), len(bgsub_cached))
                        diff = bgsub_cached[:n] - bestFit[:n]
                        offset = np.min(bgsub_cached[:n]) * 0.5 \
                                 if n > 0 else 0.0
                        self.plot_processed.plot(
                            tt_fit[:n], diff + offset,
                            pen=pg.mkPen('b', width=1, style=Qt.DashLine))
                except (KeyError, TypeError, ValueError):
                    pass  # results exist but are stale/incomplete - skip silently

        # --- Tick marks (loop over JCPDS table rows) ---
        if not hasattr(self, 'phases') or self.phases is None:
            return
        for row in range(self.ui.tableWidget_JCPDSTable.rowCount()):
            chk_show = self._get_table_checkbox(
                self.ui.tableWidget_JCPDSTable, row, 1)
            if chk_show is None or not chk_show.isChecked():
                continue
            color = self._get_phase_color(row)
            try:
                peak_positions = exc.calculate_tickmarks(
                    self.phases,
                    self.WL,
                    self.ui.doubleSpinBox_twoThetaMin.value(),
                    self.ui.doubleSpinBox_twoThetaMax.value(),
                    row
                )
                for pos in peak_positions[:, 0]:
                    line = pg.InfiniteLine(
                        pos=float(pos),
                        angle=90,
                        pen=pg.mkPen(color, width=1)
                    )
                    self.plot_processed.addItem(line)
            except Exception:
                pass  # phase not fully loaded yet
    
    def save_2D_plot(self):
        """Save the current 2D heatmap (including the colour-bar / LUT and
        any peak overlays) as a TIFF or PNG file.

        We use pyqtgraph's ImageExporter on the PlotItem rather than just
        grabbing the widget's pixels, because that way the export respects
        the active LUT levels and renders at higher resolution. If the
        ImageExporter import fails (very old pyqtgraph), we fall back to
        QWidget.grab() which simply screenshots the widget.
        """
        if not hasattr(self, 'plotWidget_2D'):
            QMessageBox.warning(self, "Save 2D Plot", "2D plot is not initialised yet.")
            return

        # Build a sensible default filename based on the current dataset
        default_name = "2D_plot"
        if getattr(self, 'file_paths', None):
            base = os.path.splitext(os.path.basename(self.file_paths[0]))[0]
            default_name = f"{base}_2D"

        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save 2D plot",
            default_name,
            "TIFF Image (*.tiff *.tif);;PNG Image (*.png);;All Files (*)"
        )
        if not save_path:
            return  # user cancelled

        # Ensure the file has a sensible extension
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in ('.tif', '.tiff', '.png'):
            if 'PNG' in selected_filter:
                save_path += '.png'
            else:
                save_path += '.tiff'

        # Preferred path: pyqtgraph ImageExporter on the PlotItem.
        try:
            from pyqtgraph.exporters import ImageExporter
            exporter = ImageExporter(self.plotWidget_2D.getPlotItem())
            # Bump width to ~2x the on-screen size for a crisper output;
            # pyqtgraph keeps the aspect ratio automatically.
            on_screen_w = max(1, int(self.plotWidget_2D.width()))
            exporter.parameters()['width'] = on_screen_w * 2
            exporter.export(save_path)
        except Exception as e:
            # Fallback - grab the widget pixmap. This loses the LUT widget
            # but always works regardless of pyqtgraph version.
            print(f"ImageExporter failed ({e}); falling back to QWidget.grab().")
            try:
                self.plotWidget_2D.grab().save(save_path)
            except Exception as e2:
                QMessageBox.critical(
                    self, "Save 2D Plot",
                    f"Could not save the 2D plot:\n{e2}")
                return

        print(f"2D plot saved to: {save_path}")

    def update_2D_plot(self):
        """
        Re-render the 2D heatmap from the accumulated all_patterns_BGsub array.
        Each row is a pattern; x-axis is 2-theta; y-axis is pattern index.

        IMPORTANT: we REUSE the persistent self.ui.img_item that the LUT widget
        is bound to. Creating a new ImageItem every refresh (the old behaviour)
        would detach the LUT and was the reason the contrast slider did nothing.

        Intensity scaling (self._intensity_scale: linear/log/sqrt) is applied
        BEFORE the data is handed to the LUT so the histogram spans the
        transformed range - this is what fixes the "everything bunches to the
        left" complaint with a linear LUT on diffraction data.
        """
        if not hasattr(self, 'all_patterns_BGsub') or not hasattr(self, 'current_twoTheta'):
            return
        arr = np.asarray(self.all_patterns_BGsub)
        if arr.size == 0 or arr.ndim != 2:
            return

        # Apply user-selected intensity scaling. The transformed array is
        # what gets shown AND what the LUT histogram is computed from.
        transformed = self._apply_2D_intensity_scaling(arr)
        numPatterns, numPoints = transformed.shape

        # ImageItem expects (x, y) = (n_points, n_patterns)
        img_data = transformed.T

        # Clear only the overlay items (fitted-peak scatter dots etc.);
        # keep the persistent ImageItem alive so the LUT stays connected.
        self._clear_2D_overlays()

        # Make sure the persistent ImageItem is in the plot
        if hasattr(self.ui, 'img_item'):
            img_item = self.ui.img_item
            if img_item.scene() is None:
                # Has been removed at some point - re-add it.
                self.plotWidget_2D.addItem(img_item)
        else:
            # Fallback (shouldn't normally happen)
            img_item = pg.ImageItem()
            self.ui.img_item = img_item
            self.plotWidget_2D.addItem(img_item)

        img_item.setImage(img_data, autoLevels=False)

        # Geometry: x = 2-theta, y = pattern index
        x0 = float(self.current_twoTheta[0])
        xRange = float(self.current_twoTheta[-1] - self.current_twoTheta[0])
        img_item.setRect(QRectF(x0, 0, xRange, numPatterns))

        self.plotWidget_2D.setLabel('bottom', 'Two Theta (°)')
        self.plotWidget_2D.setLabel('left', 'Pattern No.')
        self.plotWidget_2D.getViewBox().setAspectLocked(False)

        # --- Wire the LUT widget to this (persistent) ImageItem ---
        # setImageItem is idempotent: calling it again is safe and re-establishes
        # the connection if anything detached it.
        if hasattr(self.ui, 'lut_widget'):
            try:
                self.ui.lut_widget.setImageItem(img_item)
                data_min = float(np.nanmin(img_data))
                data_max = float(np.nanmax(img_data))
                if data_max > data_min:
                    # Seed the LUT range with the (transformed) data range
                    # so the histogram fills the bar instead of bunching
                    # everything into a thin strip on the left.
                    self.ui.lut_widget.setLevels(data_min, data_max)
                # Apply the currently-selected colormap. Only set the
                # initial preset on the very first render of the session;
                # otherwise respect whatever the user picked via the
                # colormap-cycle button.
                if not getattr(self, '_cmap_initialised', False):
                    cmaps = COLORMAP_CYCLE if COLORMAP_CYCLE else ['thermal']
                    self.ui.lut_widget.gradient.loadPreset(
                        cmaps[self._cmap_idx % len(cmaps)])
                    self._cmap_initialised = True
            except Exception:
                pass

        # --- Overlay fitted peak positions (colour-coded per phase) ---
        self._replot_2D_peak_overlays()

    def _clear_2D_overlays(self):
        """Remove every overlay item we added to the 2D plot (ticks/dots),
        but leave the persistent ImageItem in place."""
        if not hasattr(self, '_peak_overlay_items'):
            self._peak_overlay_items = []
        for it in self._peak_overlay_items:
            try:
                self.plotWidget_2D.removeItem(it)
            except Exception:
                pass
        self._peak_overlay_items = []

    def _replot_2D_peak_overlays(self):
        """Draw coloured dots on the 2D heatmap for each fitted peak."""
        if not hasattr(self, '_peak_overlay_items'):
            self._peak_overlay_items = []
        if not getattr(self, 'sequentialFitResultsPeaks', None):
            return
        for frame_key, frame_data in self.sequentialFitResultsPeaks.items():
            try:
                frame_num = int(frame_key.split('_')[1])
            except (IndexError, ValueError):
                continue
            for phase_key, peaks in frame_data.items():
                try:
                    phase_idx = int(phase_key.split('_')[1])
                except (IndexError, ValueError):
                    continue
                color = self._get_phase_color(phase_idx)
                if peaks is None or len(peaks) == 0:
                    continue
                xs = [float(p) for p in peaks]
                ys = [float(frame_num) + 0.5] * len(xs)
                dot = pg.ScatterPlotItem(
                    x=xs, y=ys,
                    size=4, pen=None,
                    brush=pg.mkBrush(color)
                )
                self.plotWidget_2D.addItem(dot)
                self._peak_overlay_items.append(dot)

    def plot2D_raw_data(self, resample_points=None):
        """
        Prepare 2D array for plotting:
        - shape = (num_patterns, num_points)
        - intensity only (background-subtracted)
        - twoTheta stored separately
        Optional resampling for faster plotting.
        """
        if not hasattr(self, 'file_paths') or len(self.file_paths) == 0:
            print("No files loaded.")
            return
    
        method = "ALS" if self.ui.checkBox_ALSfit.isChecked() else "POLY"
    
        params = {
            "twoThetaMin": self.ui.doubleSpinBox_twoThetaMin.value(),
            "twoThetaMax": self.ui.doubleSpinBox_twoThetaMax.value(),
            "lam": self.ui.doubleSpinBox_ALS_lam.value(),
            "p": self.ui.doubleSpinBox_ALS_p.value(),
            "niter": self.ui.doubleSpinBox_ALS_niter.value(),
            "prominence": self.ui.doubleSpinBox_BG_Prominence.value(),
            "height": self.ui.doubleSpinBox_BG_Height.value(),
            "order": self.ui.spinBox_BG_Order.value(),
            "peakwidth": self.ui.doubleSpinBox_BG_width.value(),
        }
    
        allIntensities = []
        twoTheta_ref = None
    
        for idx, filePath in enumerate(self.file_paths):
            result = self.data_manager.get_background(filePath, method, params)
            if result is None:
                continue
    
            twoTheta, _, _, valueIntBGsub = result
    
            if resample_points is not None and len(twoTheta) != resample_points:
                from scipy.signal import resample
                valueIntBGsub = resample(valueIntBGsub, resample_points)
                twoTheta = resample(twoTheta, resample_points)
    
            allIntensities.append(valueIntBGsub)
    
            if twoTheta_ref is None:
                twoTheta_ref = twoTheta
    
        if not allIntensities:
            print("No patterns could be processed.")
            return
    
        # Flip the list so that pattern 0 is last (bottom)
        allIntensities = allIntensities[::-1]
    
        # Convert to 2D array: (num_patterns, num_points)
        self.plot2DrawData = np.array(allIntensities)
        self.twoTheta = twoTheta_ref
    
        print("plot2DrawData shape:", self.plot2DrawData.shape)
    
    def plot2D_from_raw(self):
        """
        Plot the 2D heatmap with axes, labels, and stretched view.
        Uses PlotWidget + ImageItem for proper axes display.
        """
        if not hasattr(self, 'plot2DrawData') or not hasattr(self, 'twoTheta'):
            print("No raw 2D data available.")
            return
    
        intensity = self.plot2DrawData.copy()  # (num_patterns, num_points)
        numPatterns, numPoints = intensity.shape
    
        # Flip vertically so pattern 0 is at bottom, then transpose
        intensity = intensity[::-1, :].T  # (num_points, num_patterns)
        #intensity = self.plot2DrawData.T
        
        # If the PlotWidget doesn't exist yet, create it
        if not hasattr(self, 'plotWidget_2D'):
            self.plotWidget_2D = pg.PlotWidget()
            self.layout().addWidget(self.plotWidget_2D)  # add to your UI layout
    
        self.plotWidget_2D.clear()  # remove old plots
    
        # Create ImageItem and add to PlotWidget
        img_item = pg.ImageItem(intensity)
        self.plotWidget_2D.addItem(img_item)
    
        # Set image coordinates
        x0 = self.twoTheta[0]
        xRange = self.twoTheta[-1] - self.twoTheta[0]
        yRange = numPatterns
        img_item.setRect(QRectF(x0, 0, xRange, yRange))
    
        # Make axes visible and labeled
        self.plotWidget_2D.setLabel('bottom', 'Two Theta (°)')
        self.plotWidget_2D.setLabel('left', 'Pattern Index')
    
        # --- LUT setup: only now, after data is known ---
        self.lut_widget.setLevels([np.nanmin(self.plot2DrawData),
                                   np.nanmax(self.plot2DrawData)])
        self.lut_widget.gradient.loadPreset('thermal')  # optional color map
    
        # Stretch to fill the widget
        self.plotWidget_2D.getViewBox().setAspectLocked(False)
        self.plotWidget_2D.getViewBox().invertY(False)
    
        print("Shape (transposed):", intensity.shape)
        print("Min:", np.nanmin(intensity))
        print("Max:", np.nanmax(intensity))
        print("NaN count:", np.isnan(intensity).sum())            
    
    
    
    
    
    
    # LEGACY: update_unit_cells_for_pressure commented out - not used by GUI
    # def update_unit_cells_for_pressure(self):
    #     """
    #     Loop over all phases in the table and update
    #     phases[phase_i+'_unit_cell'] according to the spinbox pressure.
    #     """
    #     for row in range(self.ui.tableWidget_JCPDSTable.rowCount()):
    #         phase_index = row  # assumes table row order matches phases
    #         # Read pressure from column 3 spinbox
    #         spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(row, 3)
    #         P = spin_pressure.value()

    #         # Get the original unit cell and compression constants
    #         UC0 = self.phases[f'phase_{phase_index}_unit_cell_0']
    #         comp = self.phases[f'phase_{phase_index}_compression_constants']
    #         V0, K0, K0P = comp[0], comp[1], comp[2]

    #         # Compute unit cell at this pressure
    #         UC_at_P = tb.find_UC_at_P(UC0, P, V0, K0, K0P)

    #         # Update the variable unit cell used for plotting
    #         self.phases[f'phase_{phase_index}_unit_cell'] = np.array(UC_at_P)            

    def on_pressure_changed(self, row):
        """
        Update the unit cell for the phase corresponding to the table row.
        """
        phase_index = row
        spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(row, 3)
        P = spin_pressure.value()
    
        UC0 = self.phases[f'phase_{phase_index}_unit_cell_0']
        comp = self.phases[f'phase_{phase_index}_compression_constants']
        V0, K0, K0P = comp[0], comp[1], comp[2]
    
        # Compute unit cell at this pressure
        UC_at_P = EoS.find_UC_at_P(UC0, P, V0, K0, K0P)
        self.phases[f'phase_{phase_index}_unit_cell'] = np.array(UC_at_P)
    
        # Re-draw plot
        self.update_processed_plot()

    def _update_table_frame_ranges(self):
        """Update all Start No./End No. spinboxes when new files are loaded.
        Start No. stays at its current value; End No. is bumped up to the new max
        if it was previously at the old max (i.e. not manually overridden)."""
        max_f = getattr(self, '_max_frame', 0)
        for row in range(self.ui.tableWidget_JCPDSTable.rowCount()):
            spin_end = self.ui.tableWidget_JCPDSTable.cellWidget(row, 5)
            spin_start = self.ui.tableWidget_JCPDSTable.cellWidget(row, 4)
            if spin_end is not None:
                spin_end.setRange(0, max_f)
                # Only auto-update if it was at the previous max (not manually changed)
                if spin_end.value() == spin_end.maximum() or spin_end.value() == 0:
                    spin_end.setValue(max_f)
            if spin_start is not None:
                spin_start.setRange(0, max_f)

    # ------------------------------------------------------------------
    # JCPDS per-phase colour helpers
    # ------------------------------------------------------------------
    # Default palette - same letters that used to be hard-coded in the
    # tickmark/dot plot routines, so existing visuals stay identical until
    # the user picks something new.
    _DEFAULT_PHASE_PALETTE = ['y', 'c', 'm', 'g', 'r', 'b']

    def _default_phase_color(self, phase_index):
        """Return the default QColor for a given phase index, cycling
        through _DEFAULT_PHASE_PALETTE."""
        name = self._DEFAULT_PHASE_PALETTE[phase_index % len(self._DEFAULT_PHASE_PALETTE)]
        # pyqtgraph's mkColor accepts the same single-letter codes as
        # matplotlib ('y', 'c', 'm' ...); convert to QColor for the button.
        qc = pg.mkColor(name)
        return QColor(qc)

    def _apply_colour_button_style(self, btn, qcolor):
        """Paint a colour-picker cell button with the given QColor and
        store the colour both as a Qt property (for round-trip reads) and
        in the button's stylesheet (for the visible swatch)."""
        if not isinstance(qcolor, QColor):
            qcolor = QColor(qcolor)
        hex_str = qcolor.name()
        btn.setProperty('phase_color', hex_str)
        # Force a visible swatch even when 'flat'; keep a thin border so the
        # cell still reads as a clickable element on light themes.
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {hex_str}; "
            f"border: 1px solid #888; min-height: 16px; }}"
            f"QPushButton:hover {{ border: 1px solid #000; }}"
        )

    def _on_colour_button_clicked(self, row):
        """Open a QColorDialog and update the row's colour. After the
        change we refresh BOTH dependent plots so every appearance of
        this phase (tickmarks AND fitted-peak dots) gets the new colour."""
        btn = self.ui.tableWidget_JCPDSTable.cellWidget(row, 6)
        if btn is None:
            return
        current_hex = btn.property('phase_color') or '#ffff00'
        new_color = QColorDialog.getColor(
            QColor(current_hex), self, f"Pick colour for row {row}")
        if not new_color.isValid():
            return  # user cancelled
        self._apply_colour_button_style(btn, new_color)
        # Push the change to both plots
        self.update_processed_plot()
        if hasattr(self, 'plotWidget_2D'):
            self._clear_2D_overlays()
            self._replot_2D_peak_overlays()

    def _get_phase_color(self, row):
        """Return a pyqtgraph-compatible colour for the JCPDS table row.
        Falls back to the default palette if the colour widget is missing
        (e.g. for rows created by old code paths)."""
        if 0 <= row < self.ui.tableWidget_JCPDSTable.rowCount():
            btn = self.ui.tableWidget_JCPDSTable.cellWidget(row, 6)
            if btn is not None:
                hex_str = btn.property('phase_color')
                if hex_str:
                    return hex_str
        return self._DEFAULT_PHASE_PALETTE[row % len(self._DEFAULT_PHASE_PALETTE)]

    def load_jcpds_files(self):
        """
        Ask user to select multiple JCPDS or CIF files and load all phases
        into a single dictionary. CIF files are auto-converted to JCPDS
        with default EoS parameters (K0 = 20 GPa, K0P = 4) before loading.
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select JCPDS or CIF files",
            "",
            "JCPDS / CIF (*.jcpds *.cif);;JCPDS Files (*.jcpds);;CIF Files (*.cif);;All Files (*)"
        )
    
        if not file_paths:
            return

        # --- Convert any CIF inputs to JCPDS first --------------------------
        # The output JCPDS is written next to the CIF and replaces the CIF
        # in the list of paths handed to load_JCPDS.
        resolved_paths = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".cif":
                try:
                    jcpds_path = cif2jcpds.cif_to_jcpds(
                        path,
                        jcpds_path=None,    # default: same dir, .jcpds extension
                        K0=20.0,
                        K0P=4.0,
                    )
                except ImportError as e:
                    QMessageBox.critical(
                        self, "CIF conversion requires pymatgen", str(e))
                    continue
                except Exception as e:
                    QMessageBox.critical(
                        self, "CIF conversion failed",
                        f"Could not convert {os.path.basename(path)}:\n{e}")
                    continue
                resolved_paths.append(jcpds_path)
            else:
                resolved_paths.append(path)

        if not resolved_paths:
            return

        # --- LOAD AND MERGE PHASES ---
        new_phases = exc.load_JCPDS(resolved_paths)  # new batch
    
        if not hasattr(self, 'phases') or self.phases is None:
            self.phases = new_phases
            offset = 0
            # Track per-phase file paths so the JCPDS editor can save back.
            for i in range(new_phases['phases_Number']):
                self.phases[f'phase_{i}_file_path'] = resolved_paths[i]
        else:
            # Merge new phases into existing dictionary
            offset = self.phases['phases_Number']
            for i in range(new_phases['phases_Number']):
                phase_i_new = f'phase_{i + offset}'
                phase_i_old = f'phase_{i}'
                self.phases[phase_i_new + '_name'] = new_phases[phase_i_old + '_name']
                self.phases[phase_i_new + '_crystal_system'] = new_phases[phase_i_old + '_crystal_system']
                self.phases[phase_i_new + '_unit_cell'] = new_phases[phase_i_old + '_unit_cell']
                self.phases[phase_i_new + '_unit_cell_0'] = new_phases[phase_i_old + '_unit_cell_0']
                self.phases[phase_i_new + '_HKL'] = new_phases[phase_i_old + '_HKL']
                self.phases[phase_i_new + '_compression_constants'] = new_phases[phase_i_old + '_compression_constants']
                self.phases[phase_i_new + '_file_path'] = resolved_paths[i]
    
            # Update global counters
            total_phases = self.phases['phases_Number'] + new_phases['phases_Number']
            self.phases['phases_Number'] = total_phases
            self.phases['phases_used'] = np.arange(total_phases)
            self.phases['phases_Number_array'] = np.arange(total_phases)
    
        # --- POPULATE TABLE ONLY WITH NEW PHASES ---
        for i in range(new_phases['phases_Number']):
            phase_index = i + offset  # actual index in self.phases
            row = self.ui.tableWidget_JCPDSTable.rowCount()
            self.ui.tableWidget_JCPDSTable.insertRow(row)
    
            # Column 0: File name
            item_file = QtWidgets.QTableWidgetItem(self.phases[f'phase_{phase_index}_name'])
            item_file.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.ui.tableWidget_JCPDSTable.setItem(row, 0, item_file)
    
            # Column 1: Show (centred checkbox)
            wrap_show = self._centered_checkbox(
                checked=True,
                on_state_changed=self.update_processed_plot)
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 1, wrap_show)
    
            # Column 2: Use (centred checkbox)
            # Whenever the user ticks/unticks 'Use' for a phase, rebuild
            # the pressure-guess tab widget so a tab appears/disappears
            # for that phase. We use a lambda so the int that
            # stateChanged passes through doesn't collide with any
            # positional args on _rebuild_pressure_guess_tabs.
            wrap_use = self._centered_checkbox(
                checked=False,
                on_state_changed=lambda _state: self._rebuild_pressure_guess_tabs())
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 2, wrap_use)
    
            # Column 3: Pressure (now allows a moderate negative excursion
            # so the user can tweak unit cells past zero pressure).
            spin_pressure = QtWidgets.QDoubleSpinBox()
            spin_pressure.setDecimals(2)
            spin_pressure.setRange(-100.0, 1000.0)
            spin_pressure.setValue(0.0)
            # Use partial to pass the phase index
            spin_pressure.valueChanged.connect(partial(self.on_pressure_changed, row))
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 3, spin_pressure)
    
            # Column 4: Start No.
            spin_start = QtWidgets.QSpinBox()
            spin_start.setRange(0, 100000)
            spin_start.setValue(0)
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 4, spin_start)
    
            # Column 5: End No. (auto-set to max loaded frame)
            spin_end = QtWidgets.QSpinBox()
            spin_end.setRange(0, 100000)
            max_f = getattr(self, '_max_frame', 0)
            spin_end.setValue(max_f)
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 5, spin_end)

            # Column 6: Colour (clickable button that opens a colour picker).
            # The default palette cycles through the same colours we used to
            # hard-code in the plotting routines, so existing visuals stay
            # unchanged until the user picks something new.
            default_color = self._default_phase_color(phase_index)
            colour_btn = QtWidgets.QPushButton()
            colour_btn.setFlat(True)
            colour_btn.setAutoFillBackground(True)
            colour_btn.setProperty('phase_color', default_color.name())
            self._apply_colour_button_style(colour_btn, default_color)
            colour_btn.clicked.connect(partial(self._on_colour_button_clicked, row))
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 6, colour_btn)

            # Column 7: Edit (opens the JCPDS editor for this row).
            edit_btn = QtWidgets.QPushButton("Edit")
            edit_btn.setToolTip("Open the JCPDS editor for this phase")
            edit_btn.clicked.connect(partial(self._on_edit_button_clicked, row))
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 7, edit_btn)

        # Refresh the processed plot immediately so ticks appear on load
        if hasattr(self, 'current_twoTheta'):
            self.update_processed_plot()

    # -------------------------------------------------------------------------
    # JCPDS editor support
    # -------------------------------------------------------------------------
    def _on_edit_button_clicked(self, row):
        """Open the JCPDS editor dialog for the phase in the given row.
        On accept, the in-memory phases dict is updated and dependent
        plots are refreshed."""
        if not hasattr(self, 'phases') or self.phases is None:
            return
        if row < 0 or row >= self.phases.get('phases_Number', 0):
            return

        phase_dict = {
            'name':                  self.phases[f'phase_{row}_name'],
            'file_path':             self.phases.get(f'phase_{row}_file_path', ''),
            'crystal_system':        self.phases[f'phase_{row}_crystal_system'],
            'unit_cell_0':           self.phases[f'phase_{row}_unit_cell_0'],
            'compression_constants': self.phases[f'phase_{row}_compression_constants'],
            'HKL':                   self.phases[f'phase_{row}_HKL'],
        }

        dlg = JCPDSEditDialog(self, phase_dict)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        if dlg.result_phase is None:
            return

        # --- Update in-memory phases ---
        rp = dlg.result_phase
        self.phases[f'phase_{row}_crystal_system']        = rp['crystal_system']
        self.phases[f'phase_{row}_unit_cell_0']           = np.array(rp['unit_cell_0'], dtype=float)
        self.phases[f'phase_{row}_unit_cell']             = np.array(rp['unit_cell_0'], dtype=float)
        self.phases[f'phase_{row}_HKL']                   = np.array(rp['HKL'], dtype=float)
        self.phases[f'phase_{row}_compression_constants'] = np.array(rp['compression_constants'], dtype=float)
        if rp.get('file_path'):
            self.phases[f'phase_{row}_file_path'] = rp['file_path']
            # Update the visible name in the table if Save As gave us a new file.
            new_name = os.path.splitext(os.path.basename(rp['file_path']))[0]
            self.phases[f'phase_{row}_name'] = new_name
            it = self.ui.tableWidget_JCPDSTable.item(row, 0)
            if it is not None:
                it.setText(new_name)

        # Re-apply the current pressure so the working unit cell is consistent
        # with the new ambient one + new EoS.
        spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(row, 3)
        if spin_pressure is not None and spin_pressure.value() != 0.0:
            self.on_pressure_changed(row)

        # Refresh both dependent plots
        if hasattr(self, 'current_twoTheta'):
            self.update_processed_plot()
        if hasattr(self, 'plotWidget_2D'):
            self._clear_2D_overlays()
            self._replot_2D_peak_overlays()

    # -------------------------------------------------------------------------
    # Unified _process_next_frame (handles all three seq modes)
    # -------------------------------------------------------------------------
    def _process_next_frame(self):
        """Handles sequential, pressureWalk, and pressureGuess modes via
        self._seq_mode. Uses the JCPDS table as the source of truth for which
        phases are active and (for mode == sequential / frame 0 of the other
        modes) their starting pressure."""

        # Cancel from progress dialog?
        if getattr(self, '_progress_cancelled', False):
            self._finish_batch_run(cancelled=True)
            return

        frame = self._seq_current_frame

        if frame > self._seq_end_frame:
            self._finish_batch_run(cancelled=False)
            return

        print(f"Refining frame {frame}")

        # Item 7: DO NOT move the spinbox / redraw the 1D plot on every frame.
        # The spinbox valueChanged signal triggers BG + processed plot redraw,
        # which is dominant per-frame cost in the GUI path.  Instead we load
        # the data for this frame ourselves, cheap and direct.
        data_path = self.file_paths[frame]
        cached_params = self._seq_bg_params
        cached_method = self._seq_bg_method
        bg_result = self.data_manager.get_background(
            data_path, cached_method, cached_params)
        if bg_result is None:
            print(f"  Could not load frame {frame}; skipping.")
            self._seq_current_frame += 1
            self._update_progress_dialog(self._seq_current_frame)
            return
        twoTheta, _, _, valueIntBGsub = bg_result
        # Keep self.current_* in sync in case other code reads them
        self.current_twoTheta       = twoTheta
        self.current_valueIntBGsub  = valueIntBGsub

        # ---- Determine used phases for this frame ----
        # Columns in tableWidget_JCPDSTable:
        #   0 = file name, 1 = Show, 2 = Use, 3 = Pressure,
        #   4 = Start No., 5 = End No.
        used_phases = []
        n_rows = self.ui.tableWidget_JCPDSTable.rowCount()
        for phase_index in range(n_rows):
            chk_use    = self._get_table_checkbox(
                self.ui.tableWidget_JCPDSTable, phase_index, 2)
            spin_start = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 4)
            spin_end   = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 5)
            if chk_use is None or not chk_use.isChecked():
                continue
            fs = spin_start.value() if spin_start else 0
            fe = spin_end.value()   if spin_end   else self._seq_end_frame
            if fs <= frame <= fe:
                used_phases.append(phase_index)

        if not used_phases:
            print(f"  No phases active in frame {frame}; skipping.")
            self._seq_current_frame += 1
            self._update_progress_dialog(self._seq_current_frame)
            return

        self.phases['phases_used'] = used_phases

        mode = getattr(self, '_seq_mode', 'sequential')

        # Reset every active phase's per-frame "fix lattice" flag at the
        # start of each frame; the pressureGuess branch below sets it on
        # again for phases whose tab row at the matched frame has
        # 'Fix' ticked. Other modes leave it cleared, so initialise_*
        # falls back to its normal "vary=True with maxShift bounds"
        # behaviour.
        for phase_index in used_phases:
            self.phases[f'phase_{phase_index}_fix_lattice'] = False

        # ---- Determine a starting unit cell for every active phase ----
        for phase_index in used_phases:
            spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 3)
            P_jcpds = spin_pressure.value() if spin_pressure else 0.0  # JCPDS table P

            UC0  = self.phases[f'phase_{phase_index}_unit_cell_0']
            comp = self.phases[f'phase_{phase_index}_compression_constants']
            V0, K0, K0P = comp[0], comp[1], comp[2]

            UC_start = None

            if mode == 'pressureWalk':
                # NOTE: P_prev is re-derived from the previous fitted UC
                # via BM3 every frame. The walk is internally self-
                # consistent but locked to the BM3 EOS: if the actual
                # experimental P-V trajectory deviates from BM3 (e.g.
                # because of thermal effects, or because the loaded K0/K0P
                # are approximate), small errors compound across frames.
                # For tighter control, use 'pressureGuess' with measured
                # ruby pressures as anchor points.
                if frame > 0 and \
                   f'frame_{frame - 1}' in self.sequentialFitResults and \
                   f'phase_{phase_index}' in self.sequentialFitResults[f'frame_{frame - 1}']:
                    prev_UC = self.sequentialFitResults[f'frame_{frame - 1}'][f'phase_{phase_index}']
                    deltaP  = getattr(self, '_seq_deltaP', 0.5)
                    P_prev  = EoS.BM3_EOS(EoS.unitCellVolume(prev_UC), V0, K0, K0P)
                    UC_start = EoS.scale_UC_at_P(prev_UC, P_prev + deltaP, UC0, K0, K0P)
                else:
                    UC_start = EoS.find_UC_at_P(UC0, P_jcpds, V0, K0, K0P)

            elif mode == 'pressureGuess':
                # Per-phase table source of truth. Falls back to the JCPDS
                # table pressure if this phase's tab has fewer than 2 rows.
                fpg_by_phase = getattr(self, '_framePressureGuessByPhase', {})
                rows = fpg_by_phase.get(phase_index, [])

                # Default: P from JCPDS spinbox; default no fix.
                P_target = P_jcpds
                fix_this = False

                if len(rows) >= 2:
                    # Linear interpolation across the [(frame, P), ...]
                    # pairs. We use np.interp directly: pressure_interpolation
                    # in BatchFit_toolbox.py builds an `frame_arr` via
                    # np.linspace which does NOT step at integer frames
                    # (it's len = int(last_frame) samples from first to
                    # last), so the index lookup `P_arr[frame - first]`
                    # is off by a fractional position. np.interp is
                    # exact here.
                    fpts = np.array([int(r[0]) for r in rows], dtype=float)
                    Ppts = np.array([float(r[1]) for r in rows], dtype=float)
                    P_target = float(np.interp(float(frame), fpts, Ppts))
                elif len(rows) == 1:
                    P_target = float(rows[0][1])

                # 'Fix' applies only to frames that EXACTLY match the
                # 'Frame' value of a row whose 'Fix' checkbox is ticked.
                # This is intentionally strict: nearest-neighbour
                # matching would propagate the flag to dozens of
                # neighbouring frames, which is rarely what the user
                # wants. To fix several frames, add a row for each.
                fix_this = any(int(r[0]) == int(frame) and bool(r[2])
                               for r in rows)

                # Stamp per-phase fix flag onto the phases dict; the
                # initialise_parameters_LB function reads it for THIS
                # frame and clears it again on the next frame below.
                self.phases[f'phase_{phase_index}_fix_lattice'] = fix_this

                UC_start = EoS.find_UC_at_P(UC0, P_target, V0, K0, K0P)

            else:
                if frame == 0 or \
                   f'frame_{frame - 1}' not in self.sequentialFitResults or \
                   f'phase_{phase_index}' not in self.sequentialFitResults.get(
                                                    f'frame_{frame - 1}', {}):
                    UC_start = EoS.find_UC_at_P(UC0, P_jcpds, V0, K0, K0P)
                else:
                    UC_start = self.sequentialFitResults[f'frame_{frame - 1}'][f'phase_{phase_index}']

            if UC_start is not None:
                self.phases[f'phase_{phase_index}_unit_cell'] = np.array(UC_start)

        # ---- Run the Le Bail fit for this frame ----
        try:
            results = exc.fit_LB(
                twoTheta,
                valueIntBGsub,
                self.phases,
                self.WL,
                frame=frame,
                twoThetaMin=self._seq_tt_min,
                twoThetaMax=self._seq_tt_max,
                framePressureGuess=[],
                sigGuess=self._seq_sig_guess,
                sigma_mode=self._seq_sigma_mode,
                sigmaBounds=self._seq_sigma_bounds,
                ampBounds=self._seq_amp_bounds,
                ampPrefactor=self._seq_amp_prefactor,
                maxShift=self._seq_max_shift,
                excludePeakList=[]
            )
        except Exception as e:
            print(f"  Fit failed for frame {frame}: {e}")
            self._seq_current_frame += 1
            self._update_progress_dialog(self._seq_current_frame)
            return

        # ---- Store results ----
        self.sequentialFitResults[f'frame_{frame}']      = {}
        self.sequentialFitResultsPeaks[f'frame_{frame}'] = {}

        # Item 2: cache fit curve for replay when the user scrolls back
        try:
            tt_fit, bestFit = results['data_fit']
            _,      bgsub   = results['data_BGsub']
            self.sequentialFitResults[f'frame_{frame}']['_data_fit']   = (
                np.asarray(tt_fit), np.asarray(bestFit))
            self.sequentialFitResults[f'frame_{frame}']['_data_BGsub'] = (
                np.asarray(tt_fit), np.asarray(bgsub))
        except (KeyError, TypeError, ValueError):
            pass

        # Cache fit-quality stats so the wavelength/Rw label can show them
        # when the user scrolls back to a fitted frame.
        if 'Rw' in results:
            self.sequentialFitResults[f'frame_{frame}']['Rw'] = results['Rw']
        for stat in ('chisqr', 'redchi', 'nfev'):
            if stat in results:
                self.sequentialFitResults[f'frame_{frame}'][stat] = results[stat]

        # Cache the per-peak fit context so save_all_results can later
        # build *_fitParam.txt files. We copy the same five keys that
        # fit_LB stamps on its return value; all start with '_' so they
        # don't collide with phase_* entries and are skipped by the
        # existing per-phase loop in save_all_results.
        for k in ('_lmfit_params', '_sigma_mode',
                  '_WL', '_twoThetaMin', '_twoThetaMax'):
            if k in results:
                self.sequentialFitResults[f'frame_{frame}'][k] = results[k]

        for phase in used_phases:
            UC_fit = results.get(f'phase_{phase}_unit_cell_fit')
            peaks  = results.get(f'phase_{phase}_peakPosition_fit')
            if UC_fit is None:
                continue
            self.sequentialFitResults[f'frame_{frame}'][f'phase_{phase}']      = UC_fit
            self.sequentialFitResultsPeaks[f'frame_{frame}'][f'phase_{phase}'] = peaks
            # Also stash peakPosition under the standard key so the
            # *_fitParam.txt writer can find it via results.get(...).
            if peaks is not None:
                self.sequentialFitResults[f'frame_{frame}'][
                    f'phase_{phase}_peakPosition_fit'] = peaks
            # Per-parameter unit-cell errors (6-element ndarray, see
            # BatchFit_toolbox.unitCell_errors). Stored under two keys:
            # the legacy 'phase_X_error' (used by save_all_results' loop)
            # and the standard 'phase_X_unit_cell_error' (used by the new
            # lattParam/fitParam writers, matching the single-fit dict).
            UC_err = results.get(f'phase_{phase}_unit_cell_error')
            if UC_err is not None:
                self.sequentialFitResults[f'frame_{frame}'][f'phase_{phase}_error'] = UC_err
                self.sequentialFitResults[f'frame_{frame}'][
                    f'phase_{phase}_unit_cell_fit'] = UC_fit
                self.sequentialFitResults[f'frame_{frame}'][
                    f'phase_{phase}_unit_cell_error'] = UC_err
            self.phases[f'phase_{phase}_unit_cell'] = UC_fit

        # ---- Item 7: cheap live feedback ----
        # Don't redraw the whole processed plot. Just add this frame's peak
        # dots to the 2D plot incrementally (no full 2D rebuild).
        self._overlay_peaks_for_frame(frame)

        # NEW: also redraw the 1D fit plot with the just-fitted curve so
        # the user sees the fit evolving frame by frame. This is cheap
        # (5-20 ms in pyqtgraph) but for huge batches you can throttle by
        # changing FIT_PLOT_LIVE_EVERY_N below.
        if FIT_PLOT_LIVE_EVERY_N > 0 and (frame % FIT_PLOT_LIVE_EVERY_N == 0):
            try:
                self._draw_fit_overlay_for_frame(frame)
                # Sync the spinbox so the user can also see which frame is
                # being shown (without re-triggering BG/processed redraws).
                sb = self.ui.spinBox_PatternNumber
                sb.blockSignals(True)
                sb.setValue(frame)
                sb.blockSignals(False)
                QtWidgets.QApplication.processEvents()
            except Exception as e:
                print(f"  live fit-plot update failed for frame {frame}: {e}")

        self._seq_current_frame += 1
        self._update_progress_dialog(self._seq_current_frame)

    def _draw_fit_overlay_for_frame(self, frame):
        """Quick redraw of plot_processed using cached batch-fit results.

        Reads from self.sequentialFitResults[f'frame_{frame}']['_data_fit']
        and ['_data_BGsub'], which _process_next_frame populates as soon
        as a fit completes. Cheap because it skips the BG recomputation
        that update_plot/run_background would do."""
        res = self.sequentialFitResults.get(f'frame_{frame}')
        if not res or '_data_fit' not in res or '_data_BGsub' not in res:
            return

        tt_fit, bestFit = res['_data_fit']
        _,      bgsub   = res['_data_BGsub']
        diff = np.asarray(bgsub) - np.asarray(bestFit)

        # Keep the difference curve below zero so it never sits on top of
        # the data. Offset is a fraction of the data range.
        try:
            data_range = float(np.nanmax(bgsub) - np.nanmin(bgsub))
        except Exception:
            data_range = 1.0
        diff_offset = -0.10 * data_range if data_range > 0 else -0.1

        self.plot_processed.clear()
        # BG-subtracted data (gray dots)
        self.plot_processed.plot(
            tt_fit, bgsub, pen=None,
            symbol='o', symbolSize=2,
            symbolBrush=pg.mkBrush(180, 180, 180),
            symbolPen=None)
        # Fit (red line)
        self.plot_processed.plot(
            tt_fit, bestFit, pen=pg.mkPen('r', width=1.2))
        # Difference (green, offset below zero)
        self.plot_processed.plot(
            tt_fit, diff + diff_offset, pen=pg.mkPen((0, 200, 100), width=1))

        # Re-apply the wavelength/Rw label so it shows the current frame's Rw
        try:
            self._update_wavelength_label()
        except Exception:
            pass

    def _overlay_peaks_for_frame(self, frame):
        """Add peak dots on the 2D plot for just one frame. Cheap per-frame
        update used during batch fitting (vs full _replot_2D_peak_overlays)."""
        frame_data = self.sequentialFitResultsPeaks.get(f'frame_{frame}')
        if not frame_data:
            return
        if not hasattr(self, '_peak_overlay_items'):
            self._peak_overlay_items = []
        for phase_key, peaks in frame_data.items():
            try:
                phase_idx = int(phase_key.split('_')[1])
            except (IndexError, ValueError):
                continue
            if peaks is None or len(peaks) == 0:
                continue
            color = self._get_phase_color(phase_idx)
            xs = [float(p) for p in peaks]
            ys = [float(frame) + 0.5] * len(xs)
            dot = pg.ScatterPlotItem(
                x=xs, y=ys,
                size=4, pen=None,
                brush=pg.mkBrush(color))
            self.plotWidget_2D.addItem(dot)
            self._peak_overlay_items.append(dot)

    def _finish_batch_run(self, cancelled):
        """Called when a batch run finishes naturally or is cancelled."""
        self._seq_timer.stop()
        self.setEnabled(True)
        self._close_progress_dialog()

        if cancelled:
            print("Refinement cancelled.")
        else:
            print("Refinement complete.")

        # Refresh: full 2D overlay (shows every fitted frame, not just
        # those accumulated during the run) and processed plot for the
        # currently selected frame.
        try:
            self._clear_2D_overlays()
            self._replot_2D_peak_overlays()
        except Exception:
            pass
        try:
            self.update_plot()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Stop fitting
    # -------------------------------------------------------------------------
    def stop_fitting(self):
        """Stop any running sequential fit."""
        if self._seq_timer.isActive():
            self._seq_timer.stop()
            self.setEnabled(True)
            self._close_progress_dialog()
            print("Fitting stopped by user.")

    # -------------------------------------------------------------------------
    # Save all sequential results
    # -------------------------------------------------------------------------
    def save_all_results(self):
        """Save unit cell parameters and pressures for all fitted frames,
        plus (Item 6) one txt per frame with columns
        2theta | Int | IntBGsub | Fit | Diff."""
        if not self.sequentialFitResults:
            print("No sequential fit results to save.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save results")
        if not save_dir:
            return

        # Collect per-phase data
        phase_data = {}
        for frame_key in sorted(self.sequentialFitResults.keys(),
                                key=lambda k: int(k.split('_')[1])):
            frame_num = int(frame_key.split('_')[1])
            frame_res = self.sequentialFitResults[frame_key]
            # Rw is stored at the frame level (one Rw per fit, all phases)
            rw_frame = frame_res.get('Rw', np.nan)
            for phase_key, UC in frame_res.items():
                # skip metadata entries: _data_fit, _data_BGsub, Rw, chisqr,
                # redchi, nfev, phase_*_error, and the auxiliary per-phase
                # caches added for the *_fitParam.txt writer
                # (phase_X_peakPosition_fit, phase_X_unit_cell_fit,
                # phase_X_unit_cell_error).
                if phase_key.startswith('_'):
                    continue
                if phase_key in ('Rw', 'chisqr', 'redchi', 'nfev'):
                    continue
                if phase_key.endswith('_error'):
                    continue
                if phase_key.endswith('_peakPosition_fit'):
                    continue
                if phase_key.endswith('_unit_cell_fit'):
                    continue
                if phase_key.endswith('_unit_cell_error'):
                    continue
                phase_idx = int(phase_key.split('_')[1])
                if phase_idx not in phase_data:
                    phase_data[phase_idx] = []
                comp = self.phases[f'phase_{phase_idx}_compression_constants']
                V0, K0, K0P = comp[0], comp[1], comp[2]
                V   = EoS.unitCellVolume(UC)
                P   = EoS.BM3_EOS(V, V0, K0, K0P)

                # Per-parameter unit-cell errors (defaults to NaN if the
                # frame's fit didn't produce them, e.g. ill-conditioned).
                err_key = f'phase_{phase_idx}_error'
                UC_err = frame_res.get(err_key,
                                       np.full(6, np.nan, dtype=float))

                # row layout:  Frame, Pressure, a, b, c, alpha, beta, gamma,
                #              da, db, dc, dalpha, dbeta, dgamma, Rw
                phase_data[phase_idx].append(
                    [frame_num, P]
                    + list(UC)
                    + list(UC_err)
                    + [rw_frame])

        for phase_idx, rows in phase_data.items():
            phase_name = self.phases.get(f'phase_{phase_idx}_name', f'phase_{phase_idx}')
            fname = os.path.join(save_dir, f'{phase_name}_sequential_results.csv')
            header = ('Frame,Pressure_GPa,'
                      'a,b,c,alpha,beta,gamma,'
                      'da,db,dc,dalpha,dbeta,dgamma,'
                      'Rw')
            np.savetxt(fname, np.array(rows), delimiter=',',
                       header=header, comments='', fmt='%.6f')
            print(f"Saved: {fname}")

        # -------- Item 6: per-frame fitted-curve txt files --------
        curves_dir = os.path.join(save_dir, 'fit_curves')
        os.makedirs(curves_dir, exist_ok=True)
        n_saved = 0
        for frame_key, frame_res in self.sequentialFitResults.items():
            frame_num = int(frame_key.split('_')[1])
            if '_data_fit'   not in frame_res: continue
            if '_data_BGsub' not in frame_res: continue

            tt_fit,  bestFit = frame_res['_data_fit']
            _,       bgsub   = frame_res['_data_BGsub']
            diff = bgsub - bestFit

            # Try to also pull the raw intensity (Int) from the BG cache so
            # Int is from the original data, not BG-subtracted.
            try:
                raw_path = self.file_paths[frame_num]
                bg_res = self.data_manager.get_background(
                    raw_path, self._seq_bg_method, self._seq_bg_params) \
                    if hasattr(self, '_seq_bg_method') else None
                if bg_res is not None:
                    _, valueInt, _, _ = bg_res
                else:
                    valueInt = bgsub  # fallback
            except Exception:
                valueInt = bgsub

            # Guard against tiny length mismatches
            n = min(len(tt_fit), len(bestFit), len(bgsub), len(valueInt))
            out = np.column_stack([
                tt_fit[:n], valueInt[:n], bgsub[:n], bestFit[:n], diff[:n]])

            # Name the file after the source data file if possible
            try:
                base = os.path.splitext(os.path.basename(self.file_paths[frame_num]))[0]
            except (IndexError, AttributeError):
                base = f'frame_{frame_num:04d}'
            fname = os.path.join(curves_dir, f'{base}_fit.txt')
            np.savetxt(fname, out,
                       header='twoTheta\tInt\tIntBGsub\tFit\tDiff',
                       comments='# ', fmt='%.6f', delimiter='\t')
            n_saved += 1

            # ---- Per-frame _lattParam.txt (same format as Save Fit) ----
            try:
                latt_path = os.path.join(curves_dir, f'{base}_lattParam.txt')
                self._write_lattParam_file(latt_path, frame_num,
                                          frame_res, self.phases, '\t')
            except Exception as e:
                print(f"  Could not save lattParam for frame {frame_num}: {e}")

            # ---- Per-frame _fitParam.txt (per-reflection table) --------
            try:
                fitp_path = os.path.join(curves_dir, f'{base}_fitParam.txt')
                WL = float(frame_res.get('_WL',
                                         getattr(self, 'WL', np.nan)))
                tt_min = float(frame_res.get(
                    '_twoThetaMin',
                    self.ui.doubleSpinBox_twoThetaMin.value()))
                tt_max = float(frame_res.get(
                    '_twoThetaMax',
                    self.ui.doubleSpinBox_twoThetaMax.value()))
                self._write_fitParam_file(fitp_path, frame_num,
                                          frame_res, self.phases,
                                          WL, tt_min, tt_max, '\t')
            except Exception as e:
                print(f"  Could not save fitParam for frame {frame_num}: {e}")

        print(f"Saved {n_saved} fit-curve file(s) to: {curves_dir}")
        print("All results saved.")





#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#


# ----------------------------------------------------------------------------
# Parallel-pool worker for background subtraction.
#
# Lives at module level (not as a method) so that ProcessPoolExecutor can
# pickle it and ship it to a child process. Threads can pickle methods too,
# but defining this once and reusing for both modes keeps the code simple.
#
# Inputs are arrays (already cropped), NOT a file path - the parent process
# already has the raw cache warm, and re-loading from disk inside the
# workers would defeat the whole point of the raw cache and double the
# I/O cost. The parent feeds slices in and gets BG arrays back.
# ----------------------------------------------------------------------------
def _bg_worker(task):
    """Compute background for one pattern.

    task is (idx, twoTheta, valueInt, errorInt, method, params)
    where method is 'ALS' or 'POLY'. params is the same dict the
    DataManager builds (with twoThetaMin/Max etc. already used to crop).

    Returns (idx, processed_tuple) on success, (idx, exception) on failure.
    The idx round-trips so callers using as_completed() can re-establish
    input order.
    """
    idx, tt, valI, valE, method, params = task
    try:
        if method == 'ALS':
            twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = \
                tb.backgroundFit_ALS_arrays(
                    tt, valI, valE,
                    lam=params['lam'] * 1e6,
                    p=params['p'],
                    niter=params['niter'])
        elif method == 'POLY':
            (twoTheta, valueInt, errorInt, valueBG, peaks) = \
                tb.backgroundFit_arrays(
                    tt, valI, valE,
                    prominence=params['prominence'],
                    height=params['height'],
                    order=params['order'],
                    excludePeakList=[],
                    peakwidth=params['peakwidth'])
            twoTheta, valueInt, valueBG, valueIntBGsub = tb.subtract_BG(
                twoTheta, valueInt, valueBG)
        else:
            raise ValueError(f"Unknown background method: {method}")
        return idx, (twoTheta, valueInt, valueBG, valueIntBGsub)
    except Exception as e:
        return idx, e


class DataManager:
    """
    Two-layer cache for diffraction patterns:

      raw_cache : path -> (twoTheta_full, valueInt_full, errorInt_full)
          Untruncated arrays straight from disk, loaded once per file.
          Cropping (changing twoThetaMin/Max) becomes a NumPy slice on
          arrays already in memory - no disk I/O, no re-parsing.

      bg_cache  : (path, method, sorted(params)) -> (tt, valInt, valBG, valBGsub)
          Background-subtracted patterns. Identical to before, but the
          underlying load is now cheap because raw_cache supplies the
          input data without hitting disk.

    Pre-loading the raw cache in parallel (prefetch_raw) is what gives
    the big speedup on initial load - file I/O releases the GIL so a
    ThreadPoolExecutor scales nicely.
    """
    def __init__(self):
        self.bg_cache = {}
        self.raw_cache = {}    # path -> (twoTheta_full, valueInt_full, errorInt_full)

    # ------------------------------------------------------------------
    # Raw-pattern cache
    # ------------------------------------------------------------------
    def get_raw(self, path):
        """Return (twoTheta_full, valueInt_full, errorInt_full) for path,
        loading from disk on first call."""
        cached = self.raw_cache.get(path)
        if cached is not None:
            return cached
        # Load FULL pattern (no cropping). Pass an extreme range so the
        # loader's _truncate_range is effectively a no-op.
        try:
            tt, I, e = tb.loadData(path, twoThetaMin=-1e9, twoThetaMax=1e9)
        except Exception as exc_:
            print(f"  raw-load failed for {path}: {exc_}")
            return None
        cached = (np.asarray(tt), np.asarray(I), np.asarray(e))
        self.raw_cache[path] = cached
        return cached

    def get_cropped(self, path, twoThetaMin, twoThetaMax):
        """Return raw data clipped to [twoThetaMin, twoThetaMax]. Pure
        NumPy slicing - no disk I/O if the path is already in raw_cache."""
        raw = self.get_raw(path)
        if raw is None:
            return None
        tt, I, e = raw
        # twoTheta may be unsorted in pathological inputs; assume sorted
        # (which all our supported formats are) for speed.
        lo = np.searchsorted(tt, twoThetaMin, side='left')
        hi = np.searchsorted(tt, twoThetaMax, side='right')
        return tt[lo:hi].copy(), I[lo:hi].copy(), e[lo:hi].copy()

    def prefetch_raw(self, file_paths, max_workers=None,
                     progress_callback=None, cancel_flag=None):
        """Load every path in file_paths into raw_cache in parallel.

        Threads (not processes) are correct here because np.loadtxt /
        pandas releases the GIL on disk I/O, and we avoid the pickling
        cost a process pool would incur. Order-preserving via map().

        progress_callback(i, n) is called from the main thread after
        each completion, so it's safe to update Qt widgets from there.
        cancel_flag, if provided, is a list whose [0] element is checked
        between completions; set [0]=True to stop early."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Skip files we've already cached.
        todo = [p for p in file_paths if p not in self.raw_cache]
        n = len(todo)
        if n == 0:
            if progress_callback is not None:
                progress_callback(len(file_paths), len(file_paths))
            return

        if max_workers is None:
            # I/O scales past CPU count - threads are cheap, disk is the
            # bottleneck. Cap at 32 to be polite on networked drives.
            import os as _os
            max_workers = min(32, (_os.cpu_count() or 4) * 2)

        def _worker(p):
            # Worker runs in a thread; touches no Qt objects. Loads the
            # full pattern (no crop) so the raw cache is range-agnostic.
            try:
                tt, I, e = tb.loadData(p, twoThetaMin=-1e9, twoThetaMax=1e9)
                return p, (np.asarray(tt), np.asarray(I), np.asarray(e))
            except Exception as exc_:
                return p, exc_

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_worker, p): p for p in todo}
            for fut in as_completed(futures):
                if cancel_flag is not None and cancel_flag[0]:
                    # Best-effort cancel: stop consuming results, let
                    # remaining futures finish in the background (they
                    # cannot be killed mid-loadtxt safely).
                    break
                p, payload = fut.result()
                if isinstance(payload, Exception):
                    print(f"  raw-load failed for {p}: {payload}")
                else:
                    self.raw_cache[p] = payload
                done += 1
                if progress_callback is not None:
                    # progress is over the WHOLE file_paths list so the
                    # ETA looks right even when most were already cached.
                    already = len(file_paths) - n
                    progress_callback(already + done, len(file_paths))

    # ------------------------------------------------------------------
    # Parallel background subtraction
    # ------------------------------------------------------------------
    def compute_backgrounds_parallel(self, file_paths, method, params,
                                     mode='threads', max_workers=None,
                                     progress_callback=None,
                                     cancel_flag=None):
        """Compute BG-subtracted patterns for many files in parallel.

        mode in {'serial', 'threads', 'processes'}:
          - serial    : just calls get_background(path) in a loop. Use as
                        a fallback when pool startup would be wasteful.
          - threads   : ThreadPoolExecutor. Modest speedup; safe everywhere.
          - processes : ProcessPoolExecutor. Best for large batches on
                        multi-core machines; ~1 s pool-startup overhead
                        and pickling cost per pattern.

        Workers operate on arrays from the raw cache, NOT file paths -
        we feed cropped arrays in and get BG arrays back. This means
        callers should call prefetch_raw first; if a path isn't in the
        raw cache we transparently fall back to disk-based loading for
        that one file (via get_background).

        Results are written into self.bg_cache so subsequent
        get_background() calls hit the cache.

        Returns a list of results in the SAME ORDER as file_paths.
        Entries are either (twoTheta, valueInt, valueBG, valueIntBGsub)
        or None for paths that failed.
        """
        n = len(file_paths)
        results = [None] * n

        # Build the cache keys up front. Anything already cached returns
        # immediately and isn't shipped to a worker.
        keys = [(p, method, tuple(sorted(params.items()))) for p in file_paths]
        todo = []   # list of (idx, path)
        for i, key in enumerate(keys):
            cached = self.bg_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                todo.append((i, file_paths[i]))

        # Progress accounts for both already-cached and newly-computed items
        already_done = n - len(todo)

        def _bump(extra):
            if progress_callback is not None:
                progress_callback(already_done + extra, n)

        _bump(0)
        if not todo:
            return results

        # Decide on actual mode. Below the threshold, force serial.
        if len(todo) < BG_PARALLEL_MIN_FILES and mode != 'serial':
            mode = 'serial'

        if mode == 'serial':
            for done, (i, path) in enumerate(todo, start=1):
                if cancel_flag is not None and cancel_flag[0]:
                    break
                results[i] = self.get_background(path, method, params)
                _bump(done)
            return results

        # Pool-based path. Build tasks from the (warm) raw cache; if a
        # raw entry is missing we fall back to disk for that one file.
        tasks = []
        twoThetaMin = params['twoThetaMin']
        twoThetaMax = params['twoThetaMax']
        for i, path in todo:
            cropped = self.get_cropped(path, twoThetaMin, twoThetaMax)
            if cropped is None:
                # Couldn't even load the raw file; mark as failed and
                # don't enqueue. Caller will see results[i] is None.
                continue
            tt, I, e = cropped
            tasks.append((i, tt, I, e, method, dict(params)))

        if not tasks:
            return results

        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

        if max_workers is None:
            import os as _os
            cpu = _os.cpu_count() or 4
            # Threads can over-subscribe; processes shouldn't.
            max_workers = cpu if mode == 'processes' else max(2, cpu)

        Pool = ProcessPoolExecutor if mode == 'processes' else ThreadPoolExecutor
        completed = 0

        try:
            with Pool(max_workers=max_workers) as pool:
                futures = {pool.submit(_bg_worker, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    if cancel_flag is not None and cancel_flag[0]:
                        # Cancel any not-yet-started futures; let in-flight
                        # ones complete (we ignore their results).
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                    try:
                        idx, payload = fut.result()
                    except Exception as e:
                        print(f"  BG worker crashed: {e}")
                        completed += 1
                        _bump(completed)
                        continue
                    if isinstance(payload, Exception):
                        path = file_paths[idx]
                        print(f"  BG fit failed for {path}: {payload}")
                    else:
                        results[idx] = payload
                        # Populate the cache for follow-up get_background
                        # calls (e.g. when save_all_results re-asks for
                        # the same (path, method, params) tuple).
                        self.bg_cache[keys[idx]] = payload
                    completed += 1
                    _bump(completed)
        except Exception as e:
            # ProcessPoolExecutor can fail to start on some frozen builds;
            # gracefully fall back to serial so the user still gets data.
            print(f"  Parallel BG pool failed ({e}); falling back to serial.")
            for done, (i, path) in enumerate(todo, start=1):
                if cancel_flag is not None and cancel_flag[0]:
                    break
                if results[i] is None:
                    results[i] = self.get_background(path, method, params)
                _bump(already_done if False else done)

        return results

    # ------------------------------------------------------------------
    # Background-subtracted cache (unchanged contract)
    # ------------------------------------------------------------------
    def get_background(self, path, method, params):
        # Build cache key
        key = (path, method, tuple(sorted(params.items())))

        if key in self.bg_cache:
            return self.bg_cache[key]

        twoThetaMin = params["twoThetaMin"]
        twoThetaMax = params["twoThetaMax"]

        # Fast path: if the raw pattern is in memory, do BG subtraction
        # directly on the cropped arrays - no disk I/O.
        cropped = None
        if path in self.raw_cache:
            cropped = self.get_cropped(path, twoThetaMin, twoThetaMax)

        try:
            if method == "ALS":
                if cropped is not None:
                    tt_c, I_c, e_c = cropped
                    twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = \
                        tb.backgroundFit_ALS_arrays(
                            tt_c, I_c, e_c,
                            lam=params["lam"] * 1e6,
                            p=params["p"],
                            niter=params["niter"])
                else:
                    result = exc.run_backgroundFit_ALS(
                        path,
                        twoThetaMin,
                        twoThetaMax,
                        params["lam"] * 1e6,
                        params["p"],
                        params["niter"],
                    )
                    if result is None:
                        return None
                    twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = result

            elif method == "POLY":
                excludePeakList = []
                if cropped is not None:
                    tt_c, I_c, e_c = cropped
                    (twoTheta, valueInt, errorInt, valueBG, peaks) = \
                        tb.backgroundFit_arrays(
                            tt_c, I_c, e_c,
                            prominence=params["prominence"],
                            height=params["height"],
                            order=params["order"],
                            excludePeakList=excludePeakList,
                            peakwidth=params["peakwidth"])
                    twoTheta, valueInt, valueBG, valueIntBGsub = tb.subtract_BG(
                        twoTheta, valueInt, valueBG)
                else:
                    result = exc.run_background_fit(
                        path,
                        twoThetaMin,
                        twoThetaMax,
                        params["prominence"],
                        params["height"],
                        params["order"],
                        excludePeakList,
                        params["peakwidth"],
                    )
                    if result is None:
                        return None
                    twoTheta, valueInt, errorInt, valueBG, valueIntBGsub, peaks = result

            else:
                raise ValueError("Unknown background method")

        except Exception as e:
            print(f"  BG fit failed for {path}: {e}")
            return None

        processed = (twoTheta, valueInt, valueBG, valueIntBGsub)
        self.bg_cache[key] = processed
        return processed

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    # Windows: tell the OS this app has its own identity (not a Python
    # child process). Without this, Windows groups the taskbar entry
    # under python.exe and shows the Python icon instead of icon.png.
    # Must run BEFORE QApplication is created.
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'exodus.massani.exodus.1')
    elif sys.platform == 'darwin':
        # macOS: set the name shown in the menu bar / Dock when run as a
        # script. (Inside a bundled .app this comes from Info.plist instead,
        # but this makes `python EXODUS_main.py` look right too.)
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info is not None:
                info['CFBundleName'] = 'Exodus'
        except ImportError:
            pass  # pyobjc not installed; harmless, just shows "Python"
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # ---------------------------------------------------------------
    # Window / taskbar icon
    # ---------------------------------------------------------------
    # QIcon(path) silently returns an empty icon if the path doesn't
    # exist (it does NOT raise and isNull() returns False!), so any
    # typo or wrong-CWD problem just shows the default Qt logo with
    # no warning. We therefore (a) verify the file exists, (b) fall
    # back to a couple of sensible search locations, and (c) print
    # the resolved path / a clear warning so it's debuggable.
    #
    # On Windows we prefer .ico when available - PNG works for the
    # title bar but multi-resolution .ico looks crisper in the
    # taskbar at every DPI.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if sys.platform == 'win32':
        candidates += [os.path.join(script_dir, 'icon.ico'),
                       os.path.join(os.getcwd(),  'icon.ico')]
    candidates += [os.path.join(script_dir, 'icon.png'),
                   os.path.join(os.getcwd(),  'icon.png')]

    icon_path = next((p for p in candidates if os.path.isfile(p)), None)
    if icon_path is None:
        print("[icon] No icon.png/icon.ico found. Searched:")
        for p in candidates:
            print(f"       {p}")
        print(f"[icon] Place icon.png next to {os.path.basename(__file__)} "
              f"(or set the working directory to its folder).")
        app_icon = QIcon()
    else:
        app_icon = QIcon(icon_path)
        if app_icon.isNull() or not app_icon.availableSizes():
            print(f"[icon] Found {icon_path} but Qt could not decode it "
                  "as an image. Check the file is a valid PNG/ICO.")
        else:
            print(f"[icon] Using {icon_path} "
                  f"(sizes: {[(s.width(), s.height()) for s in app_icon.availableSizes()]})")

    # Setting on QApplication is what Windows actually reads for the
    # taskbar; setting on the window covers the title bar / Alt-Tab.
    app.setWindowIcon(app_icon)

    window = MainWindow()
    window.setWindowTitle('Exodus v0.1.3')
    window.setWindowIcon(app_icon)   # belt and braces

    window.show()
    sys.exit(app.exec_())