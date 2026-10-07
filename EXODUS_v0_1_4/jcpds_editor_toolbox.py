# jcpds_editor_toolbox.py for EXODUS v0.1.4
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
# The JCPDS editor pop-up opened by the 'Edit' button of the JCPDS table:
# lattice parameters, EoS constants and the (h k l, intensity, d) reflection
# list of one phase, with a live preview of the reflections at the table
# pressure. Edits can be saved back to the JCPDS file (jcpds_toolbox).
# The main window only copies the result back into its phases dict
# (MainWindow._on_edit_button_clicked).
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import os

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import crystallography_toolbox as cryst
import eos_toolbox as eos
import jcpds_toolbox as jcpds


def _volume(unitCell):
    """Cell volume for display (NaN instead of None for an invalid cell)."""
    v = cryst.unit_cell_volume(unitCell)
    return float('nan') if v is None else v


# ============================================================================
#                             JCPDS EDIT DIALOG
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

    Optional ``context`` dict:
        WL               : wavelength in A (for the 2theta columns)
        pressure         : current pressure in GPa (for the (P) columns)
        pressure_source  : short human-readable origin of that pressure
        preview_callback : callable(preview_dict) - called on EVERY edit so
                           the main window can redraw the tick marks live.
                           preview_dict keys: crystal_system, unit_cell_0,
                           unit_cell_P, HKL, compression_constants.
    """

    # Reflection-table columns
    COL_H, COL_K, COL_L, COL_I, COL_D0, COL_DP, COL_TT0, COL_TTP = range(8)
    _N_EDITABLE = 4   # H, K, L, Int.

    SYMMETRIES = [
        "CUBIC", "TETRAGONAL", "HEXAGONAL", "RHOMBOHEDRAL",
        "ORTHORHOMBIC", "MONOCLINIC", "TRICLINIC",
    ]

    # Which lattice fields the user may edit for each symmetry. The others
    # are kept in sync automatically (or hard-coded - e.g. alpha=gamma=90 for
    # monoclinic). Order matches [a, b, c, alpha, beta, gamma].
    _ENABLED_LATTICE = {
        "CUBIC":         (True,  False, False, False, False, False),
        "TETRAGONAL":    (True,  False, True,  False, False, False),
        "HEXAGONAL":     (True,  False, True,  False, False, False),
        "RHOMBOHEDRAL":  (True,  False, False, True,  False, False),
        "ORTHORHOMBIC":  (True,  True,  True,  False, False, False),
        "MONOCLINIC":    (True,  True,  True,  False, True,  False),
        "TRICLINIC":     (True,  True,  True,  True,  True,  True),
    }

    def __init__(self, parent, phase, context=None):
        super().__init__(parent)
        self.setWindowTitle(f"JCPDS Editor - {phase.get('name', '')}")
        self.setModal(True)
        self.resize(1280, 760)

        ctx = context or {}
        self._WL = ctx.get('WL', None)
        try:
            self._P = float(ctx.get('pressure', 0.0) or 0.0)
        except (TypeError, ValueError):
            self._P = 0.0
        self._P_source = ctx.get('pressure_source', '')
        self._preview_cb = ctx.get('preview_callback', None)

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

    # ---- pressure-dependent cells --------------------------------
    def _cells(self):
        """Return (uc0, ucP): the ambient cell exactly as typed, and the
        same cell compressed to the current pressure with the CURRENT
        (possibly edited) EoS constants."""
        uc0 = self._read_unit_cell()
        ucP = eos.safe_uc_at_p(uc0, self._P, self.spn_K0.value(),
                               self.spn_K0P.value())
        return uc0, ucP

    def _emit_preview(self):
        """Push the current editor state to the main window so the tick
        marks in the pattern panel follow the edits live."""
        if self._preview_cb is None:
            return
        try:
            uc0, ucP = self._cells()
            self._preview_cb({
                "crystal_system": self.cmb_symmetry.currentText(),
                "unit_cell_0": uc0,
                "unit_cell_P": np.asarray(ucP, dtype=float),
                "HKL": self._read_hkl(),
                "compression_constants": np.array([
                    _volume(uc0),
                    self.spn_K0.value(), self.spn_K0P.value(),
                    self.spn_alphaT.value(), self.spn_dK0dT.value(),
                    self.spn_dK0PdT.value()], dtype=float),
            })
        except Exception as e:
            print(f"  JCPDS editor preview failed: {e}")

    # ---- internal helpers -------------------------------------------------
    @staticmethod
    def _copy_phase(phase):
        """Deep-ish copy of the phase dict so edits don't leak before Save."""
        return {
            "name":                  phase.get("name", ""),
            "file_path":             phase.get("file_path", ""),
            # Normalise so legacy 'ORTHOROMBIC' maps onto the combo entry
            "crystal_system":        cryst.normalise_crystal_system(
                                         phase.get("crystal_system", "CUBIC")),
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
        gb_ref = QtWidgets.QGroupBox(
            "Reflections (H, K, L, Int. editable; d-spacing and 2\u03b8 calculated)")
        v = QtWidgets.QVBoxLayout(gb_ref)
        # Pressure used for the (P) columns
        wl_txt = "-" if not self._WL else f"{float(self._WL):.5f} \u00c5"
        src = f"  ({self._P_source})" if self._P_source else ""
        self.lbl_pressure = QtWidgets.QLabel(
            f"(0) = ambient cell above.   (P) = cell at P = {self._P:.3f} GPa"
            f"{src}, using the EoS above.   \u03bb = {wl_txt}")
        self.lbl_pressure.setWordWrap(True)
        self.lbl_pressure.setStyleSheet("color: #aaaaaa;")
        v.addWidget(self.lbl_pressure)
        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["H", "K", "L", "Int.", "d-spacing(0)", "d-spacing(P)",
             "twoTheta(0)", "twoTheta(P)"])
        self.table.verticalHeader().setVisible(False)
        # Column widths: H/K/L only ever hold small integers; the four
        # calculated columns get equal room and share any spare width.
        hh = self.table.horizontalHeader()
        for col, w in ((self.COL_H, 36), (self.COL_K, 36), (self.COL_L, 36),
                       (self.COL_I, 62)):
            hh.setSectionResizeMode(col, QtWidgets.QHeaderView.Fixed)
            self.table.setColumnWidth(col, w)
        for col in (self.COL_D0, self.COL_DP, self.COL_TT0, self.COL_TTP):
            hh.setSectionResizeMode(col, QtWidgets.QHeaderView.Stretch)
        # The calculated columns are read-only (flags set per item).
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
                                            else f"{float(val):.1f}")
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, col, it)
        # Calculated cells (d0, dP, 2theta0, 2thetaP): read-only
        for col in (self.COL_D0, self.COL_DP, self.COL_TT0, self.COL_TTP):
            c_item = QtWidgets.QTableWidgetItem("")
            c_item.setFlags(c_item.flags() & ~Qt.ItemIsEditable)
            c_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, col, c_item)

    def _read_unit_cell(self):
        return np.array([
            self.spn_a.value(), self.spn_b.value(), self.spn_c.value(),
            self.spn_alpha.value(), self.spn_beta.value(), self.spn_gamma.value(),
        ], dtype=float)

    def _read_hkl(self):
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
        # EoS edits change the (P) columns. (Before v0.1.4 these spins were
        # not connected at all, so editing only K0 did not even mark the
        # dialog as dirty and the change could be lost on OK.)
        for spin in (self.spn_K0, self.spn_K0P, self.spn_alphaT,
                     self.spn_dK0dT, self.spn_dK0PdT):
            spin.valueChanged.connect(self._on_eos_changed)
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
        self._emit_preview()

    def _on_lattice_changed(self, *_):
        if self._suppress_updates:
            return
        self.dirty = True
        self._enforce_symmetry_constraints()
        self._refresh_volume()
        self._refresh_reflection_table_d()
        self._emit_preview()

    def _on_eos_changed(self, *_):
        if self._suppress_updates:
            return
        self.dirty = True
        self._refresh_reflection_table_d()
        self._emit_preview()

    def _on_table_item_changed(self, item):
        if self._suppress_updates:
            return
        if item.column() >= self._N_EDITABLE:
            return   # calculated column (we set these ourselves)
        self.dirty = True
        # Recompute d / 2theta for the changed row if h/k/l changed.
        if item.column() in (self.COL_H, self.COL_K, self.COL_L):
            self._refresh_reflection_row_d(item.row())
        self._emit_preview()

    def _on_add_row(self):
        self._suppress_updates = True
        try:
            self._append_reflection_row(0, 0, 1, 100.0)
        finally:
            self._suppress_updates = False
        self._refresh_reflection_row_d(self.table.rowCount() - 1)
        self.dirty = True
        self._emit_preview()

    def _on_remove_row(self):
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        if not rows:
            return
        for r in rows:
            self.table.removeRow(r)
        self.dirty = True
        self._emit_preview()

    def _on_reload(self):
        if QMessageBox.question(self, "Reload",
                                "Discard all unsaved changes and reload from disk?",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        path = self._original.get("file_path")
        if path and os.path.isfile(path):
            try:
                fresh = jcpds.load_jcpds([path])
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
        self._emit_preview()

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
            jcpds.write_jcpds(
                jcpdsPath=path,
                unitCell=self._read_unit_cell(),
                HKL=self._read_hkl(),
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
                _volume(self._read_unit_cell()),
                self.spn_K0.value(), self.spn_K0P.value(),
                self.spn_alphaT.value(), self.spn_dK0dT.value(),
                self.spn_dK0PdT.value(),
            ]),
            "HKL": self._read_hkl(),
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
            _volume(uc),
            self.spn_K0.value(), self.spn_K0P.value(),
            self.spn_alphaT.value(), self.spn_dK0dT.value(),
            self.spn_dK0PdT.value(),
        ], dtype=float)
        _, ucP = self._cells()
        self.result_phase = {
            "name": self._original.get("name", ""),
            "file_path": self.result_path or self._original.get("file_path", ""),
            "crystal_system": self.cmb_symmetry.currentText(),
            "unit_cell_0": uc,
            "unit_cell_P": np.asarray(ucP, dtype=float),
            "compression_constants": cc,
            "HKL": self._read_hkl(),
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
            elif sym == "ORTHORHOMBIC":
                self.spn_alpha.setValue(90.0); self.spn_beta.setValue(90.0); self.spn_gamma.setValue(90.0)
            elif sym == "MONOCLINIC":
                self.spn_alpha.setValue(90.0); self.spn_gamma.setValue(90.0)
            # TRICLINIC: no constraints
        finally:
            self._suppress_updates = False

    def _refresh_volume(self):
        try:
            uc = self._read_unit_cell()
            vol = _volume(uc)
            self.lbl_volume.setText(f"V0 = {vol:.4f} A^3")
        except Exception:
            self.lbl_volume.setText("V0 = -")

    def _refresh_reflection_table_d(self):
        # Compute both cells ONCE (the EoS solve is the expensive bit) and
        # reuse them for every row.
        cells = self._cells()
        for r in range(self.table.rowCount()):
            self._refresh_reflection_row_d(r, cells)

    def _refresh_reflection_row_d(self, row, cells=None):
        """Fill d-spacing(0), d-spacing(P), twoTheta(0), twoTheta(P)."""
        if row < 0 or row >= self.table.rowCount():
            return
        try:
            h = float(self.table.item(row, self.COL_H).text())
            k = float(self.table.item(row, self.COL_K).text())
            l = float(self.table.item(row, self.COL_L).text())
        except (ValueError, AttributeError):
            return
        uc0, ucP = cells if cells is not None else self._cells()
        sym = self.cmb_symmetry.currentText()
        d0, tt0 = cryst.d_and_twotheta(uc0, sym, h, k, l, self._WL)
        dP, ttP = cryst.d_and_twotheta(ucP, sym, h, k, l, self._WL)

        def _fmt(x, n):
            return "-" if (x is None or not np.isfinite(x)) else f"{x:.{n}f}"

        self._suppress_updates = True
        try:
            for col, txt in ((self.COL_D0, _fmt(d0, 5)), (self.COL_DP, _fmt(dP, 5)),
                             (self.COL_TT0, _fmt(tt0, 4)), (self.COL_TTP, _fmt(ttP, 4))):
                it = self.table.item(row, col)
                if it is None:
                    it = QtWidgets.QTableWidgetItem()
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self.table.setItem(row, col, it)
                it.setText(txt)
        finally:
            self._suppress_updates = False

    @staticmethod
    def _phase_dict_from_loaded(phases, idx, filePath):
        """Convert the dict structure returned by load_jcpds into the flat
        single-phase layout the dialog uses internally."""
        return {
            "name": phases[f"phase_{idx}_name"],
            "file_path": filePath,
            "crystal_system": phases[f"phase_{idx}_crystal_system"],
            "unit_cell_0": np.array(phases[f"phase_{idx}_unit_cell_0"], dtype=float),
            "compression_constants": np.array(
                phases[f"phase_{idx}_compression_constants"], dtype=float),
            "HKL": np.array(phases[f"phase_{idx}_HKL"], dtype=float).reshape(-1, 4),
        }
