# .-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-. #
# | <>-------------------------------------------------------------------<> | #
# |                                                                         | #
# |                      _______  ______  ____  __  _______                 | #
# |                     / ____/ |/ / __ \/ __ \/ / / / ___/                 | #
# |                    / __/  |   / / / / / / / / / /\__ \                  | #
# |                   / /___ /   / /_/ / /_/ / /_/ /___/ /                  | #
# |                  /_____//_/|_\____/_____/\____//____/                   | #
# |                                                                         | #
# |   Equation-of-state X-ray Observation and Diffraction Unit-cell Solver  | #
# |                                                                         | #
# |                      -=*=-=*=-  [ Main ]  -=*=-=*=-                     | #
# | <>-------------------------------------------------------------------<> | #
# '-~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-' #

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
# Main window of EXODUS (start the program with `python EXODUS_main.py`).
#
# Program structure
# -----------------
#   EXODUS_main.py               GUI: main window, signal wiring, batch loop
#   EXODUS_GUI_ui.py             layout of the main window
#   EXODUS_core.py               workflow: one Pawley fit -> results dict
#
#   crystallography_toolbox.py   unit-cell geometry, d-spacings, symmetry
#   eos_toolbox.py               equations of state (BM2/3/4, Vinet, ...)
#   pawley_toolbox.py            Pawley model: parameters, profiles, residual
#   statistics_toolbox.py        Rp/Rwp/chi^2, lattice errors, error propagation
#   background_toolbox.py        polynomial and ALS backgrounds
#   data_toolbox.py              PONI and pattern files, pattern cache
#   jcpds_toolbox.py             JCPDS read / write, CIF -> JCPDS
#   results_toolbox.py           output files (fit, lattParam, fitParam)
#
#   jcpds_editor_toolbox.py      pop-up: JCPDS editor
#   fix_parameters_toolbox.py    pop-up: per-frame overrides (pressure guess)
#   plot_data_toolbox.py         pop-up: plot fitted quantities
#   eos_fit_toolbox.py           pop-up: EoS fit
#   lut_toolbox.py               colour look-up-table widget of the 2D plot
#   legacy_toolbox.py            retired code (commented out, not imported)
#
# Naming: functions and methods in snake_case, arguments in camelCase. Qt /
# pyqtgraph overrides (showEvent, setLevels, ...) and the generated setupUi /
# retranslateUi keep the framework names.
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import sys
import os
import time
import numpy as np

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog, QColorDialog, QMessageBox, QPushButton
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

import crystallography_toolbox as cryst
import data_toolbox as dt
import eos_fit_toolbox as eosfit
import eos_toolbox as eos
import fix_parameters_toolbox as fpt
import jcpds_editor_toolbox as jcpdsedit
import jcpds_toolbox as jcpds
import pawley_toolbox as pawley
import plot_data_toolbox as pdt
import results_toolbox as rt
import statistics_toolbox as stats

# Force PyInstaller to bundle pymatgen even though the real import is lazy
# inside jcpds_toolbox._import_pymatgen(). Do not remove.
import pymatgen
import pymatgen.core
import pymatgen.analysis.diffraction.xrd  # noqa: F401


# ============================================================================
#                       USER-TWEAKABLE SETTINGS
# ============================================================================
# Live fit-plot updates during batch fitting. Set to N>0 to redraw the 1D
# fit plot every N frames (1 = every frame, the smoothest visual feedback).
# Set to 0 to disable live updates entirely (no live preview;
# fastest batch throughput at the cost of no live preview).
FIT_PLOT_LIVE_EVERY_N = 0

# Easter egg: comment out the next line (or set False) to disable the
# RSW.png popup when the user clicks the log-scale button on the 2D plot.
ENABLE_LOG_EASTER_EGG = True

# Default intensity scaling on the 2D plot. 'log' is usually the right
# default for diffraction data because the dynamic range is huge and the
# LUT histogram bunches everything into a tiny strip otherwise. Acceptable
# values: 'linear', 'log', 'sqrt'.
DEFAULT_2D_SCALE = 'linear'

# Color-map cycle for the 2D plot (see pyqtgraph gradient presets).
COLORMAP_CYCLE = ['grey', 'thermal', 'flame', 'inferno', 'viridis',
                  'plasma', 'magma',  'cyclic']

# Parallelism for the background-subtraction pass on the 2D-stack build.
# 'serial'    - single-threaded (safest, no overhead)
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

# Start-up window size and panel proportions. Taken 1:1 from the
# reference screenshot (logical pixels). The window is clamped to the
# available screen; the panel sizes are applied as PROPORTIONS of whatever
# space the window actually gets, so they also look right on smaller
# screens.
START_WINDOW_SIZE = (2440, 1920)          # width, height
START_COLUMN_WIDTHS = (1340, 1155)        # controls | plots
START_CONTROL_HEIGHTS = (300, 250, 630, 400)   # Data, BG, JCPDS, 2D Plot+Fits
START_PLOT_HEIGHTS = (300, 600, 715)      # Raw, Pattern, 2D plot


# ============================================================================
#                          JCPDS EDIT DIALOG
# ============================================================================
# The JCPDS editor pop-up lives in jcpds_editor_toolbox.JCPDSEditDialog.
# MainWindow._on_edit_button_clicked holds the glue that feeds it the
# current phase and copies the result back.
# ============================================================================


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialise Data Manager
        self.data_manager = dt.DataManager()

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
        # Enable draw-a-rectangle-to-zoom on all plots.
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
        self.ui.pushButton_fitJCPDS_single.clicked.connect(self.pawley_fit_single)
        self.ui.pushButton_save_single.clicked.connect(self.save_single_data)

        # Save the 2D plot as TIFF/PNG
        self.ui.pushButton_save2D.clicked.connect(self.save_2d_plot)

        # Sequential fitting
        self.ui.pushButton_fitAll.clicked.connect(self.run_fit_all)
        # Cancel-fitting is handled by the progress-dialog's own button.
        self.ui.pushButton_saveAll.clicked.connect(self.save_all_results)  # CONNECTED
        self.fitModeGroup = QtWidgets.QButtonGroup(self)
        self.fitModeGroup.setExclusive(True)
        self.fitModeGroup.addButton(self.ui.checkBox_sequential)
        self.fitModeGroup.addButton(self.ui.checkBox_pressureWalk)
        self.fitModeGroup.addButton(self.ui.checkBox_poressureGuess)

        # Peak-broadening checkboxes - mutually exclusive group, used for
        # both peak profiles. Maps to the sigma_mode strings of
        # pawley_toolbox:
        self.sigmaModeGroup = QtWidgets.QButtonGroup(self)
        self.sigmaModeGroup.setExclusive(True)
        self.sigmaModeGroup.addButton(self.ui.checkBox_fixSigma_2)  # "Fix Sigma"          -> 'fixed'
        self.sigmaModeGroup.addButton(self.ui.checkBox)              # "Fix sigma 1/cos"    -> 'scherrer'
        self.sigmaModeGroup.addButton(self.ui.checkBox_2)            # "Vary Sigma"         -> 'separate'
        self.sigmaModeGroup.addButton(self.ui.checkBox_3)            # "Vary by phase"      -> 'per_phase'
        self.sigmaModeGroup.addButton(self.ui.checkBox_4_Caglioti)   # "Caglioti (U, V, W)" -> 'caglioti'
        self.ui.checkBox.setChecked(True)   # Scherrer 1/cos(theta) by default
        self._setup_profile_controls()

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

        # --- "Remove Pattern" button ---
        btn_rm = getattr(self.ui, 'pushButton_8_deltePattern', None)
        if btn_rm is not None:
            btn_rm.setToolTip("Remove the currently selected pattern from this "
                              "session (the file on disk is NOT deleted).")
            btn_rm.clicked.connect(self.remove_current_pattern)

        # --- Pre-check ALS ---
        self.ui.checkBox_ALSfit.setChecked(True)      # ALS ticked by default
        self.ui.checkBox_polyFit.setChecked(False)    # Poly unticked
        # --- Connect signals AFTER setting initial state ---
        self.ui.checkBox_ALSfit.stateChanged.connect(self.bg_fit_checkbox_changed)
        self.ui.checkBox_polyFit.stateChanged.connect(self.bg_fit_checkbox_changed)

        # 2D Plot
        # Assign UI widget to self
        self.plotWidget_2D = self.ui.plotWidget_2D

        # Initialise empty containers - 2D stack will be built when data is loaded
        self._peak_overlay_items = []

        # ---------------------------------------------------------------
        # 2D plot toolbar: intensity scaling (Lin / Log / Sqrt) and a
        # cycle-colormap button. Inserted ABOVE the LUT inside the
        # existing verticalLayout_8 so the user gets a small row of
        # buttons followed by the LUT, followed by the heatmap.
        # ---------------------------------------------------------------
        self._intensity_scale = DEFAULT_2D_SCALE      # 'linear' / 'log' / 'sqrt'
        self._cmap_idx        = 0
        self._setup_2d_toolbar()

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
        # Green "+" button next to the Pressure Guess table so
        # the user can dynamically add [frame, P] rows.
        # ---------------------------------------------------------------
        self._setup_pressure_guess_table()

        # Right-click context menu on the JCPDS table: allows the
        # user to delete a phase entirely (with confirmation).
        self._setup_jcpds_context_menu()

        # point-and-click selection + green crosshair
        self._setup_selection_crosshair()

        # start-up size (clamped to the screen); the panel
        # proportions are applied on the first show, once the real
        # geometry is known (see showEvent / _init_splitter_sizes).
        self._apply_start_size()
        self._splitters_initialised = False

        # placeholder buttons for planned features
        self._setup_placeholder_buttons()

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
        self._phase_pg_data   = {}        # phase_index -> [(frame, P, fix, overrides), ...]

        def _style_btn(btn, bgHex, tooltip):
            if btn is None:
                return
            btn.setToolTip(tooltip)
            btn.setFixedSize(28, 24)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {bgHex}; color: white; "
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


    # ---------------------------------------------------------------------
    # Tab (re)build
    # ---------------------------------------------------------------------
    def _rebuild_pressure_guess_tabs(self, *args, skipSnapshot=False, **kwargs):
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
        # (skip_snapshot=True: the caller has already edited
        # _phase_pg_data directly, e.g. remove_current_pattern)
        if not skipSnapshot:
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
            tbl.setColumnCount(4)
            tbl.setHorizontalHeaderLabels(['Frame', 'Pressure', 'Fix', 'Edit'])
            tbl.setRowCount(0)
            tbl.setProperty('phase_index', int(phase_index))
            font = QtGui.QFont()
            font.setPointSize(8)
            tbl.setFont(font)
            try:
                hh = tbl.horizontalHeader()
                # Same small font in the header as in the cells, so all
                # four columns fit the (narrow) tab. Frame / Pressure /
                # Fix are sized to their header text; the Edit button
                # column takes whatever is left.
                hh.setFont(font)
                # Stretched Edit column must be allowed to be narrow (Qt
                # never stretches it below defaultSectionSize = 100). Set
                # BEFORE the explicit widths - it resets every column.
                hh.setMinimumSectionSize(30)
                hh.setDefaultSectionSize(44)
                fm = QtGui.QFontMetrics(font)
                for col, txt, extra in ((0, 'Frame', 14), (1, 'Pressure', 14),
                                        (2, 'Fix', 14)):
                    tbl.setColumnWidth(col, max(fm.horizontalAdvance(txt) + extra, 30))
                hh.setStretchLastSection(True)
                tbl.verticalHeader().setDefaultSectionSize(22)
            except Exception:
                pass

            lay.addWidget(tbl)
            tabw.addTab(page, tab_label)
            self._phase_pg_tables[phase_index] = tbl

            # Restore any prior rows for this phase
            for row in self._phase_pg_data.get(phase_index, []):
                self._append_pressure_guess_row(tbl, *row)

        tabw.blockSignals(False)

    def _phase_tab_label(self, phaseIndex):
        """Pretty label for a tab: the phase name if known, else its index."""
        try:
            name = self.phases.get(f'phase_{phaseIndex}_name', '')
            if name:
                return f"{name}"
        except Exception:
            pass
        return f"Phase {phaseIndex}"

    @staticmethod
    def _snapshot_table_rows(tbl):
        """Return a list of (frame, pressure, fix, overrides) tuples
        extracted from a per-phase pressure-guess QTableWidget. Robust
        against missing items / empty cells / unparseable text.
        `overrides` is a deep copy of the per-frame override dict stored
        on the row's Edit button (empty dict if none)."""
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
            edit_btn = tbl.cellWidget(r, 3)
            ov = fpt.copy_overrides(getattr(edit_btn, '_exodus_overrides', None))
            out.append((frame, pres, fix, ov))
        return out

    # -------------------------------------------------------------------
    # Centred-checkbox cell widgets
    # -------------------------------------------------------------------
    # Each QCheckBox is wrapped in a tiny QWidget with a zero-margin,
    # centred QHBoxLayout, so the checkbox always sits in the geometric
    # middle of the cell regardless of column width.
    #
    # NB. setCellWidget() stores the WRAPPER, not the checkbox itself,
    # so instead of ``tableWidget.cellWidget(row, col).isChecked()``
    # every caller must go via
    # ``self._get_table_checkbox(tbl, row, col)`` instead. That helper
    # is also tolerant of cells that hold a bare QCheckBox.
    @staticmethod
    def _centered_checkbox(checked=False, tooltip=None, onStateChanged=None):
        """Create a (wrapper, checkbox) pair where the checkbox is
        horizontally centred inside the wrapper widget. Returns the
        WRAPPER -- pass it to ``setCellWidget``. The checkbox itself is
        attached as ``wrapper._checkbox`` for retrieval.

        ``onStateChanged`` is connected to ``stateChanged`` if given.
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
        if onStateChanged is not None:
            chk.stateChanged.connect(onStateChanged)
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

    def _append_pressure_guess_row(self, tbl, frame=0, pressure=0.0, fix=False,
                                   overrides=None):
        """Append a [frame, pressure, fix, edit] row to a given per-phase
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
                     "are held fixed for the frame matched to this row.\n"
                     "Individual parameters fixed via 'Edit' take "
                     "precedence over this for that frame."))
        tbl.setCellWidget(row, 2, wrapper)

        # per-frame parameter popup
        btn = QtWidgets.QPushButton("Edit")
        btn._exodus_overrides = fpt.copy_overrides(overrides)
        btn.clicked.connect(lambda _c=False, t=tbl, b=btn:
                            self._on_pressure_guess_edit(t, b))
        self._style_override_button(btn)
        tbl.setCellWidget(row, 3, btn)
        return row

    @staticmethod
    def _style_override_button(btn):
        """Orange 'Edit*' when the row carries overrides, plain otherwise."""
        ov = getattr(btn, '_exodus_overrides', None)
        if fpt.has_overrides(ov):
            btn.setText("Edit*")
            btn.setStyleSheet("QPushButton { background-color: #d68910; "
                              "color: white; font-weight: bold; "
                              "font-size: 8pt; }")
            btn.setToolTip("Overrides for this frame:\n" + fpt.summarise(ov))
        else:
            btn.setText("Edit")
            btn.setStyleSheet("QPushButton { font-size: 8pt; }")
            btn.setToolTip("Open the per-frame parameter editor "
                           "(fix / set single parameters, fit inputs).")

    def _on_pressure_guess_edit(self, tbl, btn):
        """Open the per-frame parameter popup for the row that owns `btn`."""
        row = -1
        for r in range(tbl.rowCount()):
            if tbl.cellWidget(r, 3) is btn:
                row = r
                break
        if row < 0:
            return
        try:
            phase_index = int(tbl.property('phase_index'))
        except (TypeError, ValueError):
            return
        if not hasattr(self, 'phases') or self.phases is None:
            return
        try:
            frame = int(float(tbl.item(row, 0).text()))
            pressure = float(tbl.item(row, 1).text())
        except (AttributeError, ValueError):
            QtWidgets.QMessageBox.warning(
                self, "Edit frame parameters",
                "Please enter a valid frame number and pressure first.")
            return
        fix_chk = self._get_table_checkbox(tbl, row, 2)
        fix = bool(fix_chk.isChecked()) if fix_chk is not None else False

        phase = {
            'name': self.phases.get(f'phase_{phase_index}_name', f'phase {phase_index}'),
            'crystal_system': self.phases[f'phase_{phase_index}_crystal_system'],
            'unit_cell_0': self.phases[f'phase_{phase_index}_unit_cell_0'],
            'compression_constants': self.phases[f'phase_{phase_index}_compression_constants'],
            'HKL': self.phases[f'phase_{phase_index}_HKL'],
        }
        global_inputs = {
            'amp_prefactor': self.get_amp_prefactor(),
            'amp_bounds': self.get_amp_bounds(),
            'max_shift': self.get_max_shift(),
            'sig_guess': self.ui.doubleSpinBox_fitSigGuess.value(),
            'sig_bounds': self.get_sigma_bounds(),
        }
        dlg = fpt.FrameParameterDialog(
            self, phase, frame, pressure, fix,
            getattr(btn, '_exodus_overrides', None),
            WL=getattr(self, 'WL', None),
            twoThetaMin=self.ui.doubleSpinBox_twoThetaMin.value(),
            twoThetaMax=self.ui.doubleSpinBox_twoThetaMax.value(),
            fittedUc=self._fitted_uc_for(phase_index, frame),
            globalInputs=global_inputs)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        # Write the results back onto the row
        tbl.item(row, 1).setText(f"{dlg.result_pressure:g}")
        if fix_chk is not None:
            fix_chk.setChecked(dlg.result_fix)
        btn._exodus_overrides = fpt.copy_overrides(dlg.result_overrides)
        self._style_override_button(btn)

    # -------------------------------------------------------------------
    # 2D-plot toolbar: intensity scale (Lin / Log / Sqrt) and color cycle
    # -------------------------------------------------------------------
    # =====================================================================
    # JCPDS CONTEXT MENU: right-click -> delete phase
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
            '_fix_lattice', '_fix_params', '_max_shift', '_amp_scale',
            '_amp_bounds')

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

    # =====================================================================
    # point-and-click selection with a green crosshair
    # =====================================================================
    # State:
    #   self._sel_twoTheta : currently selected 2-theta (None = nothing
    #                        picked yet -> vertical lines hidden)
    #   the selected PATTERN is always spinBox_PatternNumber.value()
    # Items:
    #   _cross2d_v / _cross2d_h : vertical + horizontal line on the 2D plot
    #   _cross1d_v              : vertical line on the pattern panel
    # Clicks are read from the scene's sigMouseClicked, which only fires on
    # a click (press + release without dragging) - so left-drag rectangle
    # zoom keeps working, and double-click (auto-range) is ignored here.
    # =====================================================================
    CROSSHAIR_COLOR = (0, 230, 0)

    def _setup_selection_crosshair(self):
        self._sel_twoTheta = None
        pen = pg.mkPen(self.CROSSHAIR_COLOR, width=1.5, style=Qt.DashLine)
        self._cross2d_v = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        self._cross2d_h = pg.InfiniteLine(angle=0, movable=False, pen=pen)
        self._cross1d_v = pg.InfiniteLine(angle=90, movable=False, pen=pen)
        for ln in (self._cross2d_v, self._cross2d_h, self._cross1d_v):
            ln.setZValue(1000)
            ln.setVisible(False)
        # ignoreBounds: the lines must never influence auto-ranging
        self.plotWidget_2D.addItem(self._cross2d_v, ignoreBounds=True)
        self.plotWidget_2D.addItem(self._cross2d_h, ignoreBounds=True)
        self._ensure_1d_crosshair()

        self.plotWidget_2D.scene().sigMouseClicked.connect(self._on_2d_plot_clicked)
        self.plot_processed.scene().sigMouseClicked.connect(self._on_pattern_plot_clicked)
        # Any change of the pattern number (spinbox, file list, arrow keys,
        # Remove Pattern, ...) moves the horizontal line.
        self.ui.spinBox_PatternNumber.valueChanged.connect(
            lambda _v: self._update_crosshair())

    def _ensure_1d_crosshair(self):
        """(Re-)attach the pattern-panel crosshair. plot_processed.clear()
        removes every item, so this is called after each clear()."""
        ln = getattr(self, '_cross1d_v', None)
        if ln is None:
            return
        if ln.scene() is None:
            self.plot_processed.addItem(ln, ignoreBounds=True)
        tt = getattr(self, '_sel_twoTheta', None)
        if tt is not None:
            ln.setValue(tt)
        ln.setVisible(tt is not None)
        self._update_crosshair_readout()

    def _update_crosshair_readout(self):
        """green '2θ = ... / I = ...' readout of the crosshair in
        the top-right corner of the pattern panel. The intensity is the
        plotted (background-subtracted) pattern interpolated at 2θ."""
        lbl = getattr(self, '_cross_label', None)
        if lbl is None:
            vb = self.plot_processed.getViewBox()
            lbl = pg.TextItem(anchor=(1, 0), color=self.CROSSHAIR_COLOR)
            lbl.setParentItem(vb)     # pinned in pixels, ignores zoom / pan
            lbl.setZValue(1001)
            self._cross_label = lbl

            def _place(*_a, _vb=vb, _l=lbl):
                _l.setPos(_vb.width() - 6, 4)
            vb.sigResized.connect(_place)
            _place()
        tt = getattr(self, '_sel_twoTheta', None)
        xy = getattr(self, '_pp_xy', None)
        if tt is None or xy is None:
            lbl.setVisible(False)
            return
        x, y = np.asarray(xy[0], float), np.asarray(xy[1], float)
        n = min(len(x), len(y))
        txt = f'2\u03b8 = {tt:.4f}\u00b0'
        if n > 1:
            order = np.argsort(x[:n])
            inten = float(np.interp(tt, x[:n][order], y[:n][order]))
            txt += f'<br>I = {inten:.6g}'
        g = '#%02x%02x%02x' % tuple(self.CROSSHAIR_COLOR)
        lbl.setHtml(f'<div style="font-size:9pt; color:{g}; '
                    f'text-align:right;">{txt}</div>')
        lbl.setVisible(True)

    def _update_crosshair(self):
        """Move all crosshair lines to the current selection."""
        if not hasattr(self, '_cross2d_v'):
            return
        tt = self._sel_twoTheta
        if tt is not None:
            self._cross2d_v.setValue(tt)
        self._cross2d_v.setVisible(tt is not None)
        has_data = bool(getattr(self, 'file_paths', None))
        if has_data:
            # Pattern i occupies rows [i, i+1) of the image -> centre line
            self._cross2d_h.setValue(self.ui.spinBox_PatternNumber.value() + 0.5)
        self._cross2d_h.setVisible(has_data)
        self._ensure_1d_crosshair()

    @staticmethod
    def _click_to_view(plotWidget, ev):
        """Map a scene click to data coordinates; None if the click was not
        a plain left-click inside the plot's ViewBox."""
        if ev.button() != Qt.LeftButton or ev.double():
            return None
        vb = plotWidget.getPlotItem().getViewBox()
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return None
        pt = vb.mapSceneToView(pos)
        return float(pt.x()), float(pt.y())

    def _clamp_two_theta(self, x):
        tt = getattr(self, 'current_twoTheta', None)
        if tt is not None and len(tt) > 0:
            return float(min(max(x, float(tt[0])), float(tt[-1])))
        return float(x)

    def _on_2d_plot_clicked(self, ev):
        """Left-click on the 2D plot: select that 2-theta AND load the
        pattern under the cursor."""
        if not getattr(self, 'file_paths', None):
            return
        xy = self._click_to_view(self.plotWidget_2D, ev)
        if xy is None:
            return
        x, y = xy
        self._sel_twoTheta = self._clamp_two_theta(x)
        n = len(self.file_paths)
        idx = int(np.floor(y))
        idx = min(max(idx, 0), n - 1)
        if idx != self.ui.spinBox_PatternNumber.value():
            # Drives list selection, BG, pattern panel and crosshair via
            # the existing valueChanged connections.
            self.ui.spinBox_PatternNumber.setValue(idx)
        self._update_crosshair()

    def _on_pattern_plot_clicked(self, ev):
        """Left-click on the pattern panel: select that 2-theta only
        (the pattern stays the same)."""
        if not getattr(self, 'file_paths', None):
            return
        xy = self._click_to_view(self.plot_processed, ev)
        if xy is None:
            return
        self._sel_twoTheta = self._clamp_two_theta(xy[0])
        self._update_crosshair()

    # =====================================================================
    # splitter initial sizes
    # =====================================================================
    def _apply_start_size(self):
        """Resize to START_WINDOW_SIZE, clamped to the available screen."""
        w, h = START_WINDOW_SIZE
        try:
            avail = QtWidgets.QApplication.primaryScreen().availableGeometry()
            # leave room for the title bar / window frame
            w = min(w, avail.width() - 16)
            h = min(h, avail.height() - 60)
        except Exception:
            pass
        self.resize(max(w, 800), max(h, 600))

    def showEvent(self, event):
        super().showEvent(event)
        # Final widget sizes are known only now (style sheet applied).
        self.ui.equalise_broadening_rows()
        if not getattr(self, '_splitters_initialised', True):
            self._splitters_initialised = True
            # 0 ms: run after this show has been laid out
            QTimer.singleShot(0, self._init_splitter_sizes)

    @staticmethod
    def _proportional_sizes(total, ratios):
        """Split `total` pixels in the given ratios."""
        s = float(sum(ratios))
        sizes = [int(round(total * r / s)) for r in ratios]
        sizes[-1] = max(0, total - sum(sizes[:-1]))
        return sizes

    def _init_splitter_sizes(self):
        """Apply the start-up panel proportions (START_* settings).
        QSplitter.setSizes() respects every pane's minimum, so on a small
        screen the panels shrink as far as allowed and then scroll."""
        sl = getattr(self.ui, 'splitter_controls', None)
        panels = getattr(self.ui, '_control_panels', ())
        if sl is not None and len(panels) == sl.count():
            # Pin each pane's minimum WIDTH to its panel's real minimum.
            # Must happen after show(): before that the stylesheet/fonts
            # are not applied yet and the measured widths are too small,
            # which clips the panels on the right.
            min_w = 0
            for i, p in enumerate(panels):
                sa = sl.widget(i)
                sb_w = sa.verticalScrollBar().sizeHint().width()
                min_w = max(min_w, p.minimumSizeHint().width() + sb_w)
            for i in range(sl.count()):
                sl.widget(i).setMinimumWidth(min_w)
            sl.setSizes(self._proportional_sizes(
                max(sl.height(), 1), START_CONTROL_HEIGHTS))

        sp = getattr(self.ui, 'splitter_plots', None)
        if sp is not None:
            sp.setSizes(self._proportional_sizes(
                max(sp.height(), 1), START_PLOT_HEIGHTS))

        sm = getattr(self.ui, 'splitter_main', None)
        if sm is not None:
            sm.setSizes(self._proportional_sizes(
                max(sm.width(), 1), START_COLUMN_WIDTHS))

    # =====================================================================
    # placeholder buttons for planned features
    # =====================================================================
    def _setup_placeholder_buttons(self):
        """Analysis buttons below the fit controls. The pop-ups
        themselves live in their own toolboxes; the main window only hands
        over a snapshot of its fit results (see _fit_data_snapshot)."""
        wiring = (
            ('pushButton_9_plotData', self.open_plot_data,
             "Plot fitted / guessed quantities (V, P, lattice, Rp, Rwp, chi2 ...)."),
            ('pushButton_10_EoSFit', self.open_eos_fit,
             "Fit an equation of state to the fitted P-V data."),
            ('pushButton_8_stressStrain', self.open_stress_strain,
             "Stress/Strain analysis (not implemented yet)."),
        )
        for name, slot, tip in wiring:
            btn = getattr(self.ui, name, None)
            if btn is None:
                continue
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
        self._analysis_dialogs = []   # keeps open pop-ups alive

    # ---- analysis pop-ups ---------------------------------------
    def _fit_data_snapshot(self):
        """Everything plot_data_toolbox / eos_fit_toolbox need, as one dict.
        This is the ONLY interface between the main window and those
        pop-ups (they never touch the main window directly)."""
        phases = getattr(self, 'phases', None) or {}
        n = int(phases.get('phases_Number', 0) or 0)
        return {
            'phases':         phases,
            'sequential':     getattr(self, 'sequentialFitResults', {}) or {},
            'single':         getattr(self, 'results', None),
            'single_frame':   getattr(self, 'results_frame_index', None),
            'pressure_guess': {i: self._pressure_guess_rows(i) for i in range(n)},
            'colors':         {i: self._get_phase_color(i) for i in range(n)},
        }

    def _keep_dialog(self, dlg):
        if dlg is None:
            return
        self._analysis_dialogs.append(dlg)
        dlg.destroyed.connect(
            lambda *_a, d=dlg: self._analysis_dialogs.remove(d)
            if d in self._analysis_dialogs else None)

    def open_plot_data(self):
        self._keep_dialog(pdt.open_plot_data_dialog(self, self._fit_data_snapshot()))

    def open_eos_fit(self):
        self._keep_dialog(eosfit.open_fit_eos_dialog(self, self._fit_data_snapshot()))

    def open_stress_strain(self):
        QMessageBox.information(self, "Stress/Strain",
                                "This functionality has not been implemented yet.")

    def _setup_2d_toolbar(self):
        """Wire up the intensity-scale and colormap buttons that live in
        the new ``2D Plot and Fits`` panel of the UI:
            pushButton_4 -> Lin
            pushButton_5 -> Log
            pushButton_6 -> Sqrt
            pushButton_7 -> Colour (cycle colormaps)

        The buttons are static UI elements (defined in EXODUS_GUI_ui.py);
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
            lambda: self._set_2d_scale('linear'))
        self._btn_2D_log = _config_scale_btn(
            getattr(self.ui, 'pushButton_5', None),
            "Logarithmic intensity scale (usually best for diffraction data)",
            lambda: self._set_2d_scale('log'))
        self._btn_2D_sqrt = _config_scale_btn(
            getattr(self.ui, 'pushButton_6', None),
            "Square-root intensity scale",
            lambda: self._set_2d_scale('sqrt'))

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
        self._refresh_2d_scale_buttons()

    def _refresh_2d_scale_buttons(self):
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

    def _set_2d_scale(self, mode):
        """Switch the 2D heatmap intensity scaling. mode in
        {'linear', 'log', 'sqrt'}.  Triggers an Easter-egg popup the
        first time the user picks 'log' in a session (if enabled)."""
        prev_mode = getattr(self, '_intensity_scale', 'linear')
        self._intensity_scale = mode
        self._refresh_2d_scale_buttons()
        # Re-render with the new transform; the LUT histogram will then
        # span the transformed data range, which is much more uniform.
        try:
            self.update_2d_plot()
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

    def _apply_2d_intensity_scaling(self, arr):
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

        row = self._append_pressure_guess_row(tbl, default_frame, 0.0, False, None)
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
            "Delete all [frame, P, fix] entries (and their per-frame "
            "overrides) in this tab?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            tbl.setRowCount(0)

    def run_fit_all(self):
        # Logic for sequential buttons
        if self.ui.checkBox_sequential.isChecked():
            print("Running Sequential Pawley fit")
            self.pawley_fit_sequential()

        elif self.ui.checkBox_pressureWalk.isChecked():
            print("Running Pressure Walk")
            self.pressure_walk_fit()

        elif self.ui.checkBox_poressureGuess.isChecked():
            print("Running Pressure Guess")
            self.pressure_guess_fit()

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
            self.WL = dt.load_poni(self.poni_path)
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
        # Reset all plots and stored results when new data loaded
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
        # Build 2D stack for ALL files immediately so the 2D plot
        # is populated as soon as data is loaded (not lazily per-pattern).
        # ---------------------------------------------------------------
        self._build_full_2d_stack()

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
        self.update_2d_plot()

    # -------------------------------------------------------------------
    # Helper: hard reset of plots / cached fit results
    # -------------------------------------------------------------------
    def _reset_all_plots_and_results(self):
        """Clear every plot, stored fit result, and cached background
        so loading a fresh set of data files starts from a clean state."""
        # Clear plot widgets
        try:
            self.plot_raw.clear()
        except Exception:
            pass
        self._pp_xy = None   # crosshair readout has no pattern any more
        try:
            self.plot_processed.clear()
            self._ensure_1d_crosshair()   # clear() drops the crosshair
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
        self._clear_2d_overlays()

        # Forget any stored per-pattern state
        self.all_patterns_BGsub = []
        self.sequentialFitResults = {}
        self.sequentialFitResultsPeaks = {}
        self.results = None
        self.results_frame_index = None

        # Drop raw-2D cache (will be rebuilt by _build_full_2d_stack)
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
    # Helper: compute BG-subtracted stack for every loaded file
    # -------------------------------------------------------------------
    def _build_full_2d_stack(self, showProgressDialog=True):
        """Populate self.all_patterns_BGsub and self.current_twoTheta
        so update_2d_plot() can render everything without needing the
        user to click each pattern first.

        If showProgressDialog is True (default), shows a cancellable
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
        show_dialog = showProgressDialog and n_files > 4
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
                progressCallback=_on_phase1_progress if show_dialog else None,
                cancelFlag=load_cancelled,
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
                minParallelFiles=BG_PARALLEL_MIN_FILES,
                progressCallback=_on_phase2_progress if show_dialog else None,
                cancelFlag=load_cancelled,
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
        # match. Failed entries become a row of NaN (skipping them would
        # compress the stack and, if the very first file failed, leave
        # twoTheta_ref unset, which in turn makes _build_full_2d_stack bail at the
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
            print("  _build_full_2d_stack: every BG fit failed; "
                  "2D state unchanged.")
            return

        # Build a rectangular array with one row per file. Missing rows
        # stay as NaN, so update_2d_plot's nan_to_num turns them to zero
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
        # Selects file by clicking
        row = self.ui.listWidget_fileList.row(item)  # index number
        full_path = item.data(Qt.UserRole)

        # store current pattern
        self.current_file_index = row
        self.current_file_path = full_path

        # trigger plotting/loading

        # Update spinbox (this will NOT cause recursion unless connected incorrectly)
        self.ui.spinBox_PatternNumber.setValue(row)


    def selection_changed(self, row):
        if row < 0:
            return

        # Prevent double-calling
        self.ui.spinBox_PatternNumber.blockSignals(True)
        self.ui.spinBox_PatternNumber.setValue(row)
        self.ui.spinBox_PatternNumber.blockSignals(False)

        self.update_plot()
        self._update_crosshair()

    def bg_fit_checkbox_changed(self, state):
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
            self._ensure_1d_crosshair()   # clear() drops the crosshair
        except Exception:
            pass
        try:
            if hasattr(self.ui, 'img_item'):
                self.ui.img_item.clear()
        except Exception:
            pass
        # The 2D plot's per-frame peak overlays are tied to fit results
        # that survived the reset, but their pixel positions depend on
        # the new image geometry; clearing now lets _replot_2d_peak_overlays
        # redraw them at the right spot when update_2d_plot runs.
        self._clear_2d_overlays()

        # Force the in-memory 2D stack to be re-allocated under the new
        # crop. Leaving the old (n_files, old_n_pts) array in place would
        # mean run_background's "len(...) > 0" check skips re-init and
        # we get a shape mismatch on row assignment.
        self.all_patterns_BGsub = []

        # 4. Rebuild the full 2D stack with the progress popup.
        try:
            self._build_full_2d_stack(showProgressDialog=True)
        except Exception as e:
            print(f"  Recalculate BG: 2D rebuild failed: {e}")
            return

        # 5. Re-render the 1D plot for whichever pattern is selected
        #    (preserve the user's choice; don't snap back to pattern 0).
        self.run_background()

        # 6. Re-render the 2D heatmap.
        self.update_2d_plot()

    # =====================================================================
    # remove the selected pattern from the session
    # =====================================================================
    def remove_current_pattern(self):
        """Remove the currently selected pattern from this session.

        The file on disk is NOT touched. Everything that is indexed by
        pattern number is kept consistent:
          * file list, pattern spinbox range, 2D stack (row removed)
          * batch-fit results / fitted peaks (later frames shift down)
          * the single-pattern fit (dropped if it belonged to this pattern)
          * pressure-guess rows: rows ON the removed pattern are deleted,
            rows on later patterns shift down by one
          * JCPDS Start No. / End No. ranges shift down accordingly
        """
        if not getattr(self, 'file_paths', None):
            return
        n_old = len(self.file_paths)
        idx = int(self.ui.spinBox_PatternNumber.value())
        if not (0 <= idx < n_old):
            return
        name = os.path.basename(self.file_paths[idx])

        # Short Yes / No / Cancel confirmation (No and Cancel both abort)
        reply = QMessageBox.question(
            self, "Remove pattern",
            f"Do you want to remove Frame {idx}?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # ---- last pattern: just reset everything --------------------------
        if n_old == 1:
            self._reset_all_plots_and_results()
            self.file_paths = []
            self.ui.listWidget_fileList.clear()
            sb = self.ui.spinBox_PatternNumber
            sb.blockSignals(True)
            sb.setMaximum(0)
            sb.setValue(0)
            sb.blockSignals(False)
            self._max_frame = 0
            self._update_crosshair()
            self._update_wavelength_label()
            return

        # ---- 1. file list ----------------------------------------------
        del self.file_paths[idx]
        lw = self.ui.listWidget_fileList
        lw.blockSignals(True)
        it = lw.takeItem(idx)
        del it
        lw.blockSignals(False)

        # ---- 2. 2D stack -----------------------------------------------
        arr = getattr(self, 'all_patterns_BGsub', None)
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > idx:
            self.all_patterns_BGsub = np.delete(arr, idx, axis=0)
        else:
            self.all_patterns_BGsub = []

        # ---- 3. fit results --------------------------------------------
        def _renumber(cache):
            out = {}
            for k, v in (cache or {}).items():
                try:
                    fr = int(k.split('_')[1])
                except (IndexError, ValueError):
                    out[k] = v
                    continue
                if fr == idx:
                    continue
                out[f'frame_{fr - 1 if fr > idx else fr}'] = v
            return out
        self.sequentialFitResults = _renumber(self.sequentialFitResults)
        self.sequentialFitResultsPeaks = _renumber(self.sequentialFitResultsPeaks)

        rfi = getattr(self, 'results_frame_index', None)
        if rfi is not None:
            if rfi == idx:
                self.results = None
                self.results_frame_index = None
            elif rfi > idx:
                self.results_frame_index = rfi - 1

        # ---- 4. pressure-guess rows ------------------------------------
        if hasattr(self, '_phase_pg_tables'):
            for pi, tbl in list(self._phase_pg_tables.items()):
                self._phase_pg_data[pi] = self._snapshot_table_rows(tbl)
            for pi in list(self._phase_pg_data.keys()):
                self._phase_pg_data[pi] = fpt.shift_frame_rows(
                    self._phase_pg_data[pi], idx)
            self._rebuild_pressure_guess_tabs(skipSnapshot=True)

        # ---- 5. JCPDS Start No. / End No. -------------------------------
        tblj = self.ui.tableWidget_JCPDSTable
        for r in range(tblj.rowCount()):
            sp_s, sp_e = tblj.cellWidget(r, 4), tblj.cellWidget(r, 5)
            if not (isinstance(sp_s, QtWidgets.QSpinBox)
                    and isinstance(sp_e, QtWidgets.QSpinBox)):
                continue
            fs, fe = sp_s.value(), sp_e.value()
            if fs > idx:
                fs -= 1
            if fe > idx or (fe == idx and fe > fs):
                fe -= 1
            sp_s.setValue(max(fs, 0))
            sp_e.setValue(max(fe, 0))
        self._max_frame = len(self.file_paths) - 1
        self._update_table_frame_ranges()

        # ---- 6. select the pattern that moved into this slot -----------
        new_idx = min(idx, len(self.file_paths) - 1)
        sb = self.ui.spinBox_PatternNumber
        sb.blockSignals(True)
        sb.setMaximum(len(self.file_paths) - 1)
        sb.setValue(new_idx)
        sb.blockSignals(False)
        lw.blockSignals(True)
        lw.setCurrentRow(new_idx)
        lw.blockSignals(False)
        self.current_file_index = new_idx
        self.current_file_path = self.file_paths[new_idx]

        self.selection_changed(new_idx)   # 1D plots + crosshair
        self.update_2d_plot()             # 2D image + fitted-peak dots
        self._update_crosshair()
        print(f"Removed pattern {idx}: {name}  ({len(self.file_paths)} left)")


    def _update_wavelength_label(self):
        """draw/update a small 'λ = ... Å' label in the top-left of
        the processed plot. Uses a TextItem pinned to the viewbox corner so
        it tracks zoom/pan.

        If the currently-displayed frame has been fitted (either a single
        fit or a cached sequential-fit result), append Rp, Rwp and chi^2
        so the user can see the fit quality at a glance.
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

        # Fit-statistics portion (Rp, Rwp, chi^2) for the fit that
        # belongs to the displayed frame.
        stat_lines = []
        current_frame = None
        try:
            current_frame = self.ui.spinBox_PatternNumber.value()
            st = self._get_fit_stats_for_frame(current_frame)
            if st:
                def _f(key, fmt):
                    v = st.get(key)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        return 'n/a'
                    return format(v, fmt) if np.isfinite(v) else 'n/a'
                stat_lines.append(f"R<sub>p</sub> = {_f('Rp', '.4f')}")
                stat_lines.append(f"R<sub>wp</sub> = {_f('Rwp', '.4f')}")
                stat_lines.append(
                    f"\u03c7\u00b2<sub>red</sub> = {_f('chi2_red', '.4g')}")
        except Exception:
            pass

        wl_col = '#c8c8c8' if WL is None else '#ffffff'
        lines = [f'<span style="color:{wl_col};">{wl_str}</span>']
        for ln in stat_lines:
            lines.append(f'<span style="color:#ffffff;">{ln}</span>')
        self._wl_label.setHtml(
            '<div style="font-size:9pt;">' + '<br>'.join(lines) + '</div>')

    # Keys shown on the 1D plot / cached per frame
    _FIT_STAT_KEYS = ('Rp', 'Rwp', 'Rp_bgcorr', 'Rwp_bgcorr', 'Rexp',
                      'chi2', 'chi2_red', 'N_points', 'N_params')

    def _get_fit_stats_for_frame(self, frame):
        """Return a dict of fit statistics (Rp, Rwp, chi2, chi2_red, ...)
        for the given frame index, or None if the frame hasn't been fitted.
        Single-fit results take priority over sequential-fit results since
        the user has just run them."""
        fit_frame = getattr(self, 'results_frame_index', None)
        single_results = getattr(self, 'results', None)
        if (single_results is not None and fit_frame is not None
                and fit_frame == frame and 'Rp' in single_results):
            return {k: single_results.get(k) for k in self._FIT_STAT_KEYS}

        seq = getattr(self, 'sequentialFitResults', None)
        if seq is not None:
            frame_res = seq.get(f'frame_{frame}')
            if frame_res is not None and 'Rp' in frame_res:
                return {k: frame_res.get(k) for k in self._FIT_STAT_KEYS}
        return None

    # -------------------------------------------------------------------
    # Helper: make a pyqtgraph plot widget use rectangle-zoom
    # (drag a box with the left mouse button to magnify that area).
    # Right-click context menu and double-click-to-auto-range still work.
    # -------------------------------------------------------------------
    def _enable_rect_zoom(self, plotWidget):
        try:
            vb = plotWidget.getViewBox()
            vb.setMouseMode(pg.ViewBox.RectMode)
        except Exception:
            pass

    def spinbox_changed(self, index):
        if not self.file_paths:
            return
        # Keep the file-list bookkeeping in step with the spin box, however
        # the pattern was changed (arrows, typing, 2D plot, keys, list).
        if 0 <= index < len(self.file_paths):
            self.current_file_index = index
            self.current_file_path = self.file_paths[index]

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
        # via _build_full_2d_stack fills every row up front). In that case
        # the 2D heatmap and its fitted-peak overlays don't need to be
        # touched at all - and skipping the rebuild is what restores fast
        # scrolling once a batch has been fitted, because
        # _replot_2d_peak_overlays() otherwise rebuilds N_frames x N_phases
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
            self.update_2d_plot()

    def _setup_profile_controls(self):
        """Peak profile (Gaussian / pseudo-Voigt with eta) and the Caglioti
        U, V, W spin boxes. Start values: pawley_toolbox.DEFAULT_ETA and
        DEFAULT_CAGLIOTI_UVW."""
        ui = self.ui
        self.profileGroup = QtWidgets.QButtonGroup(self)
        self.profileGroup.setExclusive(True)
        self.profileGroup.addButton(ui.checkBox_4)     # Gaussian
        self.profileGroup.addButton(ui.checkBox_5)     # Pseudo Voigt
        ui.checkBox_4.setChecked(True)
        ui.checkBox_4.setToolTip("Gaussian peak profile.")
        ui.checkBox_5.setToolTip(
            "Pseudo-Voigt: eta * Lorentzian + (1 - eta) * Gaussian with the "
            "same FWHM. One global eta, fixed at the value on the right.")

        ui.doubleSpinBox_2.setRange(0.0, 1.0)
        ui.doubleSpinBox_2.setDecimals(3)
        ui.doubleSpinBox_2.setSingleStep(0.05)
        ui.doubleSpinBox_2.setValue(pawley.DEFAULT_ETA)
        ui.doubleSpinBox_2.setToolTip(
            "Pseudo-Voigt mixing parameter eta (0 = Gaussian, "
            "1 = Lorentzian); held fixed in the fit.")

        U0, V0, W0 = pawley.DEFAULT_CAGLIOTI_UVW
        for spin, lo, hi, val, nm in ((ui.doubleSpinBox_5_CagliotiU, 0.0, 1.0, U0, 'U'),
                                      (ui.doubleSpinBox_2_CagliotiV, -1.0, 1.0, V0, 'V'),
                                      (ui.doubleSpinBox_4__CagliotiW, 0.0, 1.0, W0, 'W')):
            spin.setDecimals(5)
            spin.setRange(lo, hi)
            spin.setSingleStep(0.0001)
            spin.setValue(val)
            spin.setToolTip(f"Start value of the Caglioti parameter {nm} "
                            "(deg^2); FWHM^2 = U tan^2(theta) + "
                            "V tan(theta) + W. Refined in the fit.")
        ui.checkBox_4_Caglioti.setToolTip(
            "Caglioti peak broadening: FWHM^2 = U tan^2(theta) + V tan(theta) "
            "+ W with global U, V, W (start values on the right).")

        ui.checkBox_5.toggled.connect(self._refresh_profile_controls)
        ui.checkBox_4_Caglioti.toggled.connect(self._refresh_profile_controls)
        self._refresh_profile_controls()

    def _refresh_profile_controls(self, *_args):
        """Enable eta only for pseudo-Voigt and U, V, W only for Caglioti."""
        ui = self.ui
        ui.doubleSpinBox_2.setEnabled(ui.checkBox_5.isChecked())
        on = ui.checkBox_4_Caglioti.isChecked()
        for spin in (ui.doubleSpinBox_5_CagliotiU, ui.doubleSpinBox_2_CagliotiV,
                     ui.doubleSpinBox_4__CagliotiW):
            spin.setEnabled(on)

    def get_profile(self):
        """'pseudo_voigt' or 'gaussian' from the profile checkboxes."""
        return 'pseudo_voigt' if self.ui.checkBox_5.isChecked() else 'gaussian'

    def get_eta(self):
        """Pseudo-Voigt mixing parameter eta (held fixed in the fit)."""
        return float(self.ui.doubleSpinBox_2.value())

    def get_caglioti_uvw(self):
        """Start values (U, V, W) of the Caglioti broadening (deg^2)."""
        return (float(self.ui.doubleSpinBox_5_CagliotiU.value()),
                float(self.ui.doubleSpinBox_2_CagliotiV.value()),
                float(self.ui.doubleSpinBox_4__CagliotiW.value()))

    def get_sigma_mode(self):
        """Read the peak-broadening checkboxes and return the sigma_mode
        string (see pawley_toolbox.SIGMA_MODES)."""
        if self.ui.checkBox_fixSigma_2.isChecked():
            return 'fixed'
        elif self.ui.checkBox.isChecked():
            return 'scherrer'
        elif self.ui.checkBox_2.isChecked():
            return 'separate'
        elif self.ui.checkBox_3.isChecked():
            return 'per_phase'
        elif self.ui.checkBox_4_Caglioti.isChecked():
            return 'caglioti'
        else:
            return 'scherrer'  # safe fallback

    def _show_errors_fallback_warning(self, frames):
        """Pop-up when the lattice-parameter errors of a fit had to be
        estimated with the sqrt(chi^2) fallback (no covariance matrix)."""
        text = ("Warning: A parameter is sitting on a bound; construction of "
                "covariance matrix failed. Fallback routine: Errors estimated "
                "as &radic;&chi;<sup>2</sup>.")
        if frames:
            text += ("<br><br>Affected frame(s): "
                     + ", ".join(str(f) for f in frames))
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Error estimate")
        msg.setTextFormat(Qt.RichText)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.button(QMessageBox.Ok).setText("Okay")
        msg.exec_()

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
        passed through verbatim and initialise_parameters will set
        vary=False on every lattice parameter for every active phase.
        """
        val = self.ui.doubleSpinBox_maxShiftFit.value()
        # Negative values are nonsensical; clamp to 0. Positive values
        # are clamped to <= 1.0 (= ±100%) since the widget is a fraction.
        # Zero is preserved on purpose - it triggers the lattice-fix
        # path inside initialise_parameters.
        if val <= 0:
            return 0.0
        return min(val, 1.0)

    def pawley_fit_single(self):
        '''
        Pawley fit of ONE pattern only
        '''
        self.results = None          # clear stale fit before starting
        self.results_frame_index = None  # track which pattern this fit belongs to
        self.plot_processed.clear()
        self._ensure_1d_crosshair()   # clear() drops the crosshair
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

        # Single fits never use the per-frame pressure-guess overrides;
        # make sure nothing from a previous batch run is still stamped on.
        fpt.clear_phase_override_keys(
            self.phases, range(int(self.phases.get('phases_Number', 0))))

        # Set pressure + update unit cells
        for phase_index in used_phases:

            spin_pressure = self.ui.tableWidget_JCPDSTable.cellWidget(phase_index, 3)
            P = spin_pressure.value()

            UC0 = self.phases[f'phase_{phase_index}_unit_cell_0']
            comp = self.phases[f'phase_{phase_index}_compression_constants']
            V0, K0, K0P = comp[0], comp[1], comp[2]

            UC_at_P = eos.find_uc_at_p(UC0, P, V0, K0, K0P)
            self.phases[f'phase_{phase_index}_unit_cell'] = UC_at_P


        # Get frame number
        frame = self.ui.spinBox_PatternNumber.value()

        # Get fit guesses.
        # Note: ampGuess passed to fit_pawley is ignored inside fit_pawley (it
        # rebuilds the guess as max(data) * ampPrefactor). The actual
        # amplitude guess control is the 'Amplitude' spinbox via
        # get_amp_prefactor(); the 'Amp. Bounds' spinbox controls the
        # upper bound via get_amp_bounds().
        sigGuess = self.ui.doubleSpinBox_fitSigGuess.value()

        twoThetaMin = self.ui.doubleSpinBox_twoThetaMin.value()
        twoThetaMax = self.ui.doubleSpinBox_twoThetaMax.value()


        # Run fit
        results = exc.fit_pawley(
            self.current_twoTheta,
            self.current_valueIntBGsub,
            self.phases,
            self.WL,
            frame=frame,
            twoThetaMin=twoThetaMin,
            twoThetaMax=twoThetaMax,
            sigGuess=sigGuess,
            sigmaMode=self.get_sigma_mode(),
            sigmaBounds=self.get_sigma_bounds(),
            ampBounds=self.get_amp_bounds(),
            ampPrefactor=self.get_amp_prefactor(),
            maxShift=self.get_max_shift(),
            valueIntRaw=getattr(self, 'current_valueInt', None),
            profile=self.get_profile(),
            etaGuess=self.get_eta(),
            cagliotiUVW=self.get_caglioti_uvw(),
        )

        # ----------------------------------------------------------------
        # Did the fit actually produce a usable result? fit_pawley swallows
        # exceptions internally and returns {} when the optimiser blew up
        # (e.g. KeyError inside the residual); without this guard the GUI
        # would crash on `self.results['data_fit']`.
        # ----------------------------------------------------------------
        if not results or 'data_fit' not in results:
            print("Fit failed - no usable result produced. "
                  "See the error printed by fit_pawley above for details.")
            self.results = None
            self.results_frame_index = None
            try:
                QMessageBox.warning(self, "Fit failed",
                                    "The Pawley fit did not produce a "
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
                self._ensure_1d_crosshair()   # clear() drops the crosshair
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
        if results.get('_errors_fallback'):
            self._show_errors_fallback_warning([self.results_frame_index])

        # Overlay fit on plot
        self.plot_processed.clear()
        self._ensure_1d_crosshair()   # clear() drops the crosshair
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

        # Refresh the lambda/fit-statistics label (Rp, Rwp, chi2)
        try:
            self._update_wavelength_label()
        except Exception:
            pass


    def pawley_fit_sequential(self):
        print("Starting sequential refinement...")

        if not self._prepare_sequential_run(mode='sequential'):
            return
        self._seq_timer.start(10)  # 10 ms delay between frames

    # -------------------------------------------------------------------------
    # Pressure Walk Fitting
    # -------------------------------------------------------------------------
    def pressure_walk_fit(self):
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
    def pressure_guess_fit(self):
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
                for fr, P, fix, ov in rows:
                    fix_str = 'FIX' if fix else '   '
                    ov_str = ('  overrides: ' + fpt.summarise(ov)
                              if fpt.has_overrides(ov) else '')
                    print(f"    frame={fr:5d}  P={P:7.3f} GPa  {fix_str}{ov_str}")
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

        # Fail-fast: verify at least one phase is marked "Use".
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
        self._clear_2d_overlays()

        self._seq_current_frame = 0
        self._seq_end_frame = len(self.file_paths) - 1
        self._seq_mode = mode

        # snapshot every parameter the hot loop needs, ONCE. Reading
        # widget values is surprisingly costly when done 1000s of times.
        self._seq_tt_min        = self.ui.doubleSpinBox_twoThetaMin.value()
        self._seq_tt_max        = self.ui.doubleSpinBox_twoThetaMax.value()
        self._seq_sig_guess     = self.ui.doubleSpinBox_fitSigGuess.value()
        self._seq_sigma_mode    = self.get_sigma_mode()
        self._seq_sigma_bounds  = self.get_sigma_bounds()
        self._seq_amp_bounds    = self.get_amp_bounds()
        self._seq_amp_prefactor = self.get_amp_prefactor()
        self._seq_max_shift     = self.get_max_shift()
        self._seq_profile       = self.get_profile()
        self._seq_eta           = self.get_eta()
        self._seq_caglioti_uvw  = self.get_caglioti_uvw()
        self._seq_fallback_frames = []
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

        # progress dialog
        self._show_progress_dialog(mode, self._seq_end_frame + 1)
        self._seq_start_time = time.monotonic()
        return True

    def _show_progress_dialog(self, mode, nTotal):
        """Create and show a modal progress dialog with ETA + Cancel."""
        self._progress_cancelled = False

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Refinement running - {mode}")
        dlg.setModal(True)
        dlg.setMinimumWidth(380)
        # Prevent close via the X button - user must use Cancel (or wait)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowCloseButtonHint)

        lay = QtWidgets.QVBoxLayout(dlg)

        lbl = QtWidgets.QLabel(f"Fitting {nTotal} frames ({mode})")
        lay.addWidget(lbl)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, nTotal)
        bar.setValue(0)
        lay.addWidget(bar)

        eta_lbl = QtWidgets.QLabel("Frame 0 of %d   |   elapsed 0 s   |   ETA -" % nTotal)
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
        self._progress_total     = nTotal

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





    # =====================================================================
    # Output files (writers in results_toolbox):
    #   <stem>_fit<ext>       : profile incl. BG, residual, tick marks
    #   <stem>_lattParam<ext> : one row per phase (P, dP, V, dV, cell, errors,
    #                           fit statistics)
    #   <stem>_fitParam<ext>  : per-reflection table with EoS block
    # The same writers are used by "Save" and "Save all".
    # =====================================================================



    def _source_name(self, frame):
        try:
            return os.path.basename(self.file_paths[int(frame)])
        except (IndexError, TypeError, ValueError, AttributeError):
            return 'n/a'





    def save_single_data(self):
        """Save the fit of the displayed pattern: <stem>_fit, <stem>_lattParam
        and <stem>_fitParam (see the block comment above)."""
        if getattr(self, 'results', None) is None:
            print("No fit results available.")
            return
        if not getattr(self, 'file_paths', None):
            print("No pattern loaded.")
            return
        # The displayed pattern is always the spin-box value (the same
        # source the fit uses for results_frame_index).
        frame_idx = int(self.ui.spinBox_PatternNumber.value())
        fit_frame = getattr(self, 'results_frame_index', None)
        if fit_frame is not None and int(fit_frame) != frame_idx:
            QMessageBox.warning(
                self, "Save fit",
                f"The single fit belongs to pattern {fit_frame}, but pattern "
                f"{frame_idx} is displayed. Go back to pattern {fit_frame} "
                "(or re-fit this one) before saving.")
            return

        orig_path = self.file_paths[frame_idx]
        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save fit", f"{base_name}.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)")
        if not save_path:
            return

        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.csv' or ('CSV' in selected_filter and ext == ''):
            delimiter, ext = ',', '.csv'
        else:
            delimiter = '\t'
            if ext == '':
                ext = '.txt'
        save_dir = os.path.dirname(save_path)
        stem = os.path.splitext(os.path.basename(save_path))[0]
        if stem.lower().endswith('_fit'):
            stem = stem[:-4]

        # Data of the fit itself (same grid as the fit); raw intensity from
        # the fit result, falling back to the displayed pattern.
        tt, bgsub = self.results['data_BGsub']
        _, fit = self.results['data_fit']
        raw = None
        if 'data_raw' in self.results:
            raw = self.results['data_raw'][1]
        elif getattr(self, 'current_valueInt', None) is not None:
            raw = self.current_valueInt

        fit_path = os.path.join(save_dir, f'{stem}_fit{ext}')
        try:
            rt.write_fit_curve_file(fit_path, frame_idx, self.results,
                                    self.phases, tt, raw, bgsub, fit,
                                    delimiter,
                                    sourceName=self._source_name(frame_idx),
                                    WL=getattr(self, 'WL', np.nan))
            print(f"Saved fit to: {fit_path}")
        except Exception as e:
            print(f"  Could not save fit curve: {e}")

        latt_path = os.path.join(save_dir, f'{stem}_lattParam{ext}')
        try:
            rt.write_latt_param_file(latt_path, frame_idx, self.results,
                                     self.phases, delimiter)
        except Exception as e:
            print(f"  Could not save lattice parameters: {e}")

        fitp_path = os.path.join(save_dir, f'{stem}_fitParam{ext}')
        try:
            WL = float(self.results.get('_WL', getattr(self, 'WL', np.nan)))
            tt_min = float(self.results.get(
                '_twoThetaMin', self.ui.doubleSpinBox_twoThetaMin.value()))
            tt_max = float(self.results.get(
                '_twoThetaMax', self.ui.doubleSpinBox_twoThetaMax.value()))
            rt.write_fit_param_file(fitp_path, frame_idx, self.results,
                                    self.phases, WL, tt_min, tt_max,
                                    delimiter,
                                    sourceName=self._source_name(frame_idx))
        except Exception as e:
            print(f"  Could not save reflection table: {e}")






    def update_processed_plot(self):
        self.plot_processed.clear()
        self._ensure_1d_crosshair()   # clear() drops the crosshair
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
        self._pp_xy = (twoTheta, valueIntBGsub)   # for the crosshair readout
        self._update_crosshair_readout()

        # --- wavelength label (top-left) ---
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

        # Cached sequential-fit curve for this frame
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
        # clear() above already removed the old tick items
        self._tick_items = []
        self._draw_tickmarks()

    def _draw_tickmarks(self):
        """Draw one vertical tick per reflection for every 'Show'-ticked
        phase, at the 2-theta of phases[...]_unit_cell. Items are kept in
        self._tick_items so _redraw_tickmarks() can swap them without
        redrawing the data / fit curves (used by the live JCPDS editor)."""
        if not hasattr(self, '_tick_items'):
            self._tick_items = []
        if not hasattr(self, 'phases') or self.phases is None:
            return
        if getattr(self, 'WL', None) is None:
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
                    self._tick_items.append(line)
            except Exception:
                pass  # phase not fully loaded yet

    def _redraw_tickmarks(self):
        """Replace only the tick marks in the pattern panel."""
        for it in getattr(self, '_tick_items', []):
            try:
                self.plot_processed.removeItem(it)
            except Exception:
                pass
        self._tick_items = []
        self._draw_tickmarks()

    def save_2d_plot(self):
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

    def update_2d_plot(self):
        """
        Re-render the 2D heatmap from the accumulated all_patterns_BGsub array.
        Each row is a pattern; x-axis is 2-theta; y-axis is pattern index.

        IMPORTANT: we REUSE the persistent self.ui.img_item that the LUT widget
        is bound to. Creating a new ImageItem on every refresh would detach
        the LUT from the image (the contrast slider would do nothing).

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
        transformed = self._apply_2d_intensity_scaling(arr)
        numPatterns, numPoints = transformed.shape

        # ImageItem expects (x, y) = (n_points, n_patterns)
        img_data = transformed.T

        # Clear only the overlay items (fitted-peak scatter dots etc.);
        # keep the persistent ImageItem alive so the LUT stays connected.
        self._clear_2d_overlays()

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
        self._replot_2d_peak_overlays()

    def _clear_2d_overlays(self):
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

    def _replot_2d_peak_overlays(self):
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


    def plot_2d_from_raw(self):
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
        UC_at_P = eos.find_uc_at_p(UC0, P, V0, K0, K0P)
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

    def _default_phase_color(self, phaseIndex):
        """Return the default QColor for a given phase index, cycling
        through _DEFAULT_PHASE_PALETTE."""
        name = self._DEFAULT_PHASE_PALETTE[phaseIndex % len(self._DEFAULT_PHASE_PALETTE)]
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
            self._clear_2d_overlays()
            self._replot_2d_peak_overlays()

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
        # in the list of paths handed to load_jcpds.
        resolved_paths = []
        for path in file_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".cif":
                try:
                    jcpds_path = jcpds.cif_to_jcpds(
                        path,
                        jcpdsPath=None,    # default: same dir, .jcpds extension
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
        new_phases = jcpds.load_jcpds(resolved_paths)  # new batch

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
                onStateChanged=self.update_processed_plot)
            self.ui.tableWidget_JCPDSTable.setCellWidget(row, 1, wrap_show)

            # Column 2: Use (centred checkbox)
            # Whenever the user ticks/unticks 'Use' for a phase, rebuild
            # the pressure-guess tab widget so a tab appears/disappears
            # for that phase. We use a lambda so the int that
            # stateChanged passes through doesn't collide with any
            # positional args on _rebuild_pressure_guess_tabs.
            wrap_use = self._centered_checkbox(
                checked=False,
                onStateChanged=lambda _state: self._rebuild_pressure_guess_tabs())
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
    def _fitted_uc_for(self, phaseIndex, frame):
        """Fitted unit cell of `phaseIndex` in `frame`, or None. The
        single-pattern fit takes priority over the batch cache."""
        res = getattr(self, 'results', None)
        if (res is not None and getattr(self, 'results_frame_index', None) == frame):
            uc = res.get(f'phase_{phaseIndex}_unit_cell_fit')
            if uc is not None:
                return np.asarray(uc, dtype=float)
        fr = (getattr(self, 'sequentialFitResults', {}) or {}).get(f'frame_{frame}')
        if fr and f'phase_{phaseIndex}' in fr:
            return np.asarray(fr[f'phase_{phaseIndex}'], dtype=float)
        return None

    def _pressure_guess_rows(self, phaseIndex):
        """Rows (frame, P, fix, overrides) of a phase's pressure-guess
        tab, from the live table if it exists, else from the stored data."""
        tbl = getattr(self, '_phase_pg_tables', {}).get(phaseIndex)
        if tbl is not None:
            return self._snapshot_table_rows(tbl)
        return list(getattr(self, '_phase_pg_data', {}).get(phaseIndex, []))

    def _current_pressure_for_phase(self, phaseIndex, frame=None):
        """Best estimate of the pressure of a phase in the displayed
        pattern. Returns (P_GPa, human-readable source). Priority:
            1. fitted cell of this pattern  -> P from the BM3 EoS
            2. pressure-guess table (interpolated at this pattern)
            3. 'Pressure' spinbox of the JCPDS table
        """
        if frame is None:
            frame = self.ui.spinBox_PatternNumber.value()
        try:
            comp = self.phases[f'phase_{phaseIndex}_compression_constants']
            V0, K0, K0P = float(comp[0]), float(comp[1]), float(comp[2])
        except Exception:
            V0 = K0 = K0P = None

        uc_fit = self._fitted_uc_for(phaseIndex, frame)
        if uc_fit is not None and K0:
            try:
                V_fit = cryst.unit_cell_volume(uc_fit)
                P = float(eos.bm3_pressure(V_fit, V0, K0, K0P)) if V_fit else float('nan')
                if np.isfinite(P):
                    return P, f"from the fit of pattern {frame}"
            except Exception:
                pass

        rows = self._pressure_guess_rows(phaseIndex)
        if rows:
            rows = sorted(rows, key=lambda r: r[0])
            fpts = np.array([float(r[0]) for r in rows])
            Ppts = np.array([float(r[1]) for r in rows])
            P = float(np.interp(float(frame), fpts, Ppts))
            return P, f"pressure guess, pattern {frame}"

        spin = self.ui.tableWidget_JCPDSTable.cellWidget(phaseIndex, 3)
        P = float(spin.value()) if isinstance(spin, QtWidgets.QDoubleSpinBox) else 0.0
        return P, "JCPDS table"

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

        # v0.1.4 items 4+5: the editor shows d / 2theta at the CURRENT
        # pressure and pushes every edit back here so the tick marks in the
        # pattern panel move live. We snapshot the phase first so Cancel
        # can restore it exactly.
        P_cur, P_src = self._current_pressure_for_phase(row)
        keys = ('crystal_system', 'unit_cell', 'unit_cell_0', 'HKL',
                'compression_constants')
        snapshot = {k: (np.array(self.phases[f'phase_{row}_{k}'], copy=True)
                        if k != 'crystal_system'
                        else self.phases[f'phase_{row}_{k}'])
                    for k in keys if f'phase_{row}_{k}' in self.phases}

        def _preview(pv, _row=row):
            self.phases[f'phase_{_row}_crystal_system'] = pv['crystal_system']
            self.phases[f'phase_{_row}_unit_cell'] = np.asarray(pv['unit_cell_P'], dtype=float)
            self.phases[f'phase_{_row}_HKL'] = np.asarray(pv['HKL'], dtype=float).reshape(-1, 4)
            self._redraw_tickmarks()

        context = {
            'WL': getattr(self, 'WL', None),
            'pressure': P_cur,
            'pressure_source': P_src,
            'preview_callback': _preview,
        }
        dlg = jcpdsedit.JCPDSEditDialog(self, phase_dict, context=context)
        accepted = (dlg.exec_() == QtWidgets.QDialog.Accepted
                    and dlg.result_phase is not None)
        if not accepted:
            # Cancel: put the phase back exactly as it was
            for k, v in snapshot.items():
                self.phases[f'phase_{row}_{k}'] = v
            self._redraw_tickmarks()
            return

        # --- Update in-memory phases ---
        rp = dlg.result_phase
        self.phases[f'phase_{row}_crystal_system']        = rp['crystal_system']
        self.phases[f'phase_{row}_unit_cell_0']           = np.array(rp['unit_cell_0'], dtype=float)
        # Working cell = the edited cell at the current pressure, i.e.
        # exactly what the user saw while editing (ticks don't jump on OK)
        self.phases[f'phase_{row}_unit_cell']             = np.array(
            rp.get('unit_cell_P', rp['unit_cell_0']), dtype=float)
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

        # (The editor already returns the cell at the current pressure.)

        # Refresh both dependent plots
        if hasattr(self, 'current_twoTheta'):
            self.update_processed_plot()
        if hasattr(self, 'plotWidget_2D'):
            self._clear_2d_overlays()
            self._replot_2d_peak_overlays()

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

        # DO NOT move the spinbox / redraw the 1D plot on every frame.
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
        twoTheta, valueIntRaw, valueBG_frame, valueIntBGsub = bg_result
        # Keep self.current_* in sync in case other code reads them
        self.current_twoTheta       = twoTheta
        self.current_valueInt       = valueIntRaw
        self.current_valueBG        = valueBG_frame
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

        # Reset every phase's per-frame flags at the start of each frame
        # ("fix lattice" + the v0.1.4 per-frame overrides). The
        # pressureGuess branch below sets them again for phases whose tab
        # row at the matched frame asks for it. Other modes leave them
        # cleared, so initialise_* falls back to its normal "vary=True
        # with maxShift bounds" behaviour.
        fpt.clear_phase_override_keys(
            self.phases, range(int(self.phases.get('phases_Number', 0))))

        # Frame-wide fit-input overrides (sigma is shared between phases)
        frame_sig_guess = None
        frame_sig_bounds = None

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
                    V_prev  = cryst.unit_cell_volume(prev_UC)
                    P_prev  = eos.bm3_pressure(V_prev, V0, K0, K0P) if V_prev else P_jcpds
                    UC_start = eos.scale_uc_at_p(prev_UC, P_prev + deltaP, UC0, K0, K0P)
                else:
                    UC_start = eos.find_uc_at_p(UC0, P_jcpds, V0, K0, K0P)

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
                    # in pawley_toolbox.py builds an `frame_arr` via
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

                UC_start = eos.find_uc_at_p(UC0, P_target, V0, K0, K0P)

                # ---- per-frame overrides from 'Edit' ----
                # Same strict exact-frame matching as 'Fix'.
                ov = next((r[3] for r in rows
                           if int(r[0]) == int(frame) and len(r) > 3
                           and fpt.has_overrides(r[3])), None)
                if ov is not None and UC_start is not None:
                    cs = self.phases[f'phase_{phase_index}_crystal_system']
                    UC_start = fpt.apply_lattice_overrides(UC_start, cs, ov)
                    fixed = fpt.fixed_lattice_names(ov)
                    if fixed:
                        self.phases[f'phase_{phase_index}_fix_params'] = fixed
                        if fix_this:
                            print(f"  phase {phase_index}: single parameters "
                                  f"fixed ({', '.join(fixed)}) -> whole-lattice "
                                  f"'Fix' ignored for frame {frame}")
                        fix_this = False
                    fit_ov = ov.get('fit', {})
                    if 'max_shift' in fit_ov:
                        self.phases[f'phase_{phase_index}_max_shift'] = float(fit_ov['max_shift'])
                    if 'amp_bounds' in fit_ov:
                        self.phases[f'phase_{phase_index}_amp_bounds'] = float(fit_ov['amp_bounds'])
                    if 'amp_prefactor' in fit_ov:
                        # fit_pawley builds ampGuess from the GLOBAL prefactor;
                        # rescale this phase relative to it.
                        glob = self._seq_amp_prefactor or 1.0
                        self.phases[f'phase_{phase_index}_amp_scale'] = \
                            float(fit_ov['amp_prefactor']) / float(glob)
                    if 'sig_guess' in fit_ov and frame_sig_guess is None:
                        frame_sig_guess = float(fit_ov['sig_guess'])
                    if 'sig_bounds' in fit_ov and frame_sig_bounds is None:
                        frame_sig_bounds = float(fit_ov['sig_bounds'])
                    print(f"  phase {phase_index}, frame {frame}: overrides "
                          f"-> {fpt.summarise(ov)}")

                # Stamp per-phase fix flag onto the phases dict; the
                # initialise_parameters function reads it for THIS
                # frame and it is cleared again at the next frame.
                self.phases[f'phase_{phase_index}_fix_lattice'] = fix_this

            else:
                if frame == 0 or \
                   f'frame_{frame - 1}' not in self.sequentialFitResults or \
                   f'phase_{phase_index}' not in self.sequentialFitResults.get(
                                                    f'frame_{frame - 1}', {}):
                    UC_start = eos.find_uc_at_p(UC0, P_jcpds, V0, K0, K0P)
                else:
                    UC_start = self.sequentialFitResults[f'frame_{frame - 1}'][f'phase_{phase_index}']

            if UC_start is not None:
                self.phases[f'phase_{phase_index}_unit_cell'] = np.array(UC_start)

        # ---- Run the Pawley fit for this frame ----
        try:
            results = exc.fit_pawley(
                twoTheta,
                valueIntBGsub,
                self.phases,
                self.WL,
                frame=frame,
                twoThetaMin=self._seq_tt_min,
                twoThetaMax=self._seq_tt_max,
                sigGuess=(frame_sig_guess if frame_sig_guess is not None
                          else self._seq_sig_guess),
                sigmaMode=self._seq_sigma_mode,
                sigmaBounds=(frame_sig_bounds if frame_sig_bounds is not None
                             else self._seq_sigma_bounds),
                ampBounds=self._seq_amp_bounds,
                ampPrefactor=self._seq_amp_prefactor,
                maxShift=self._seq_max_shift,
                valueIntRaw=valueIntRaw,
                profile=self._seq_profile,
                etaGuess=self._seq_eta,
                cagliotiUVW=self._seq_caglioti_uvw,
            )
        except Exception as e:
            print(f"  Fit failed for frame {frame}: {e}")
            self._seq_current_frame += 1
            self._update_progress_dialog(self._seq_current_frame)
            return

        # ---- Store results ----
        self.sequentialFitResults[f'frame_{frame}']      = {}
        self.sequentialFitResultsPeaks[f'frame_{frame}'] = {}

        # cache fit curve for replay when the user scrolls back
        try:
            tt_fit, bestFit = results['data_fit']
            _,      bgsub   = results['data_BGsub']
            self.sequentialFitResults[f'frame_{frame}']['_data_fit']   = (
                np.asarray(tt_fit), np.asarray(bestFit))
            self.sequentialFitResults[f'frame_{frame}']['_data_BGsub'] = (
                np.asarray(tt_fit), np.asarray(bgsub))
        except (KeyError, TypeError, ValueError):
            pass

        # Raw (pre-BG) intensities on the fit grid: needed for the BG and
        # Int columns of the fit-curve files written by Save all.
        try:
            tt_raw, I_raw = results['data_raw']
            self.sequentialFitResults[f'frame_{frame}']['_data_raw'] = (
                np.asarray(tt_raw), np.asarray(I_raw))
        except (KeyError, TypeError, ValueError):
            pass

        # Cache fit-quality stats so the 1D-plot label can show them when
        # the user scrolls back to a fitted frame (Rp/Rwp/chi2).
        for stat in tuple(stats.PROFILE_STAT_KEYS) + ('weighting', 'chisqr',
                                                   'redchi', 'nfev'):
            if stat in results:
                self.sequentialFitResults[f'frame_{frame}'][stat] = results[stat]

        # Cache the per-peak fit context so save_all_results can later
        # build *_fitParam.txt files. We copy the same five keys that
        # fit_pawley stamps on its return value; all start with '_' so they
        # don't collide with phase_* entries and are skipped by the
        # existing per-phase loop in save_all_results.
        for k in ('_lmfit_params', '_sigma_mode', '_profile', '_errors_fallback',
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
            # pawley_toolbox.unit_cell_errors). Stored under two keys:
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
            # Volume / pressure with errors as computed by the fit
            for suffix in ('_V_fit', '_V_error', '_P_fit', '_P_error'):
                key = f'phase_{phase}{suffix}'
                if key in results:
                    self.sequentialFitResults[f'frame_{frame}'][key] = results[key]
            self.phases[f'phase_{phase}_unit_cell'] = UC_fit
        if results.get('_errors_fallback'):
            self._seq_fallback_frames.append(frame)

        # ---- cheap live feedback ----
        # Don't redraw the whole processed plot. Just add this frame's peak
        # dots to the 2D plot incrementally (no full 2D rebuild).
        self._overlay_peaks_for_frame(frame)

        # Also redraw the 1D fit plot with the just-fitted curve so
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

        self._pp_xy = (tt_fit, bgsub)        # for the crosshair readout
        self._ensure_1d_crosshair()   # clear() drops the crosshair
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

        # Re-apply the wavelength/statistics label for the current frame
        try:
            self._update_wavelength_label()
        except Exception:
            pass

    def _overlay_peaks_for_frame(self, frame):
        """Add peak dots on the 2D plot for just one frame. Cheap per-frame
        update used during batch fitting (vs full _replot_2d_peak_overlays)."""
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

        # Don't let the last frame's per-frame flags leak into later fits
        if getattr(self, 'phases', None) is not None:
            fpt.clear_phase_override_keys(
                self.phases, range(int(self.phases.get('phases_Number', 0))))

        if cancelled:
            print("Refinement cancelled.")
        else:
            print("Refinement complete.")
        if getattr(self, '_seq_fallback_frames', None):
            self._show_errors_fallback_warning(self._seq_fallback_frames)
            self._seq_fallback_frames = []

        # Refresh: full 2D overlay (shows every fitted frame, not just
        # those accumulated during the run) and processed plot for the
        # currently selected frame.
        try:
            self._clear_2d_overlays()
            self._replot_2d_peak_overlays()
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
        """Save all sequential-fit results:
          <phase>_sequential_results.csv : one row per fitted frame
                                           (columns LATT_COLUMNS)
          fit_curves/<file>_fit.txt       : profile incl. BG and residual
          fit_curves/<file>_lattParam.txt
          fit_curves/<file>_fitParam.txt
        """
        if not self.sequentialFitResults:
            print("No sequential fit results to save.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select directory to save results")
        if not save_dir:
            return

        import re
        frame_keys = sorted(self.sequentialFitResults.keys(),
                            key=lambda k: int(k.split('_')[1]))

        # -------- aggregated per-phase table --------
        phase_rows = {}
        for frame_key in frame_keys:
            frame_num = int(frame_key.split('_')[1])
            frame_res = self.sequentialFitResults[frame_key]
            fitted = sorted(int(m.group(1)) for k in frame_res
                            for m in [re.match(r'^phase_(\d+)$', k)] if m)
            for phase_idx in fitted:
                cs = rt.phase_cell_summary(frame_res, phase_idx, self.phases)
                if cs is None:
                    continue
                phase_rows.setdefault(phase_idx, []).append(
                    [float(frame_num), cs['P'], cs['dP'], cs['V'], cs['dV']]
                    + cs['UC'] + cs['err']
                    + [frame_res.get(k, np.nan) for k in
                       ('Rp', 'Rwp', 'chi2', 'chi2_red')])

        for phase_idx, rows in phase_rows.items():
            phase_name = self.phases.get(f'phase_{phase_idx}_name', f'phase_{phase_idx}')
            fname = os.path.join(save_dir, f'{phase_name}_sequential_results.csv')
            np.savetxt(fname, np.array(rows, dtype=float), delimiter=',',
                       header=','.join(rt.LATT_COLUMNS), comments='',
                       fmt='%.6g')
            print(f"Saved: {fname}")

        # -------- per-frame files --------
        curves_dir = os.path.join(save_dir, 'fit_curves')
        os.makedirs(curves_dir, exist_ok=True)
        n_saved = 0
        for frame_key in frame_keys:
            frame_num = int(frame_key.split('_')[1])
            frame_res = self.sequentialFitResults[frame_key]
            if '_data_fit' not in frame_res or '_data_BGsub' not in frame_res:
                continue
            tt_fit, bestFit = frame_res['_data_fit']
            _, bgsub = frame_res['_data_BGsub']

            # Raw intensity: cached with the fit; otherwise
            # recompute from the BG cache with the batch BG settings.
            raw = None
            if '_data_raw' in frame_res:
                raw = frame_res['_data_raw'][1]
            else:
                try:
                    if hasattr(self, '_seq_bg_method'):
                        bg_res = self.data_manager.get_background(
                            self.file_paths[frame_num], self._seq_bg_method,
                            self._seq_bg_params)
                        if bg_res is not None:
                            raw = bg_res[1]
                except Exception:
                    raw = None

            try:
                base = os.path.splitext(os.path.basename(self.file_paths[frame_num]))[0]
            except (IndexError, AttributeError):
                base = f'frame_{frame_num:04d}'

            try:
                rt.write_fit_curve_file(
                    os.path.join(curves_dir, f'{base}_fit.txt'), frame_num,
                    frame_res, self.phases, tt_fit, raw, bgsub, bestFit, '\t',
                    sourceName=self._source_name(frame_num),
                    WL=getattr(self, 'WL', np.nan))
                n_saved += 1
            except Exception as e:
                print(f"  Could not save fit curve for frame {frame_num}: {e}")

            try:
                rt.write_latt_param_file(
                    os.path.join(curves_dir, f'{base}_lattParam.txt'),
                    frame_num, frame_res, self.phases, '\t')
            except Exception as e:
                print(f"  Could not save lattParam for frame {frame_num}: {e}")

            try:
                WL = float(frame_res.get('_WL', getattr(self, 'WL', np.nan)))
                tt_min = float(frame_res.get(
                    '_twoThetaMin', self.ui.doubleSpinBox_twoThetaMin.value()))
                tt_max = float(frame_res.get(
                    '_twoThetaMax', self.ui.doubleSpinBox_twoThetaMax.value()))
                rt.write_fit_param_file(
                    os.path.join(curves_dir, f'{base}_fitParam.txt'),
                    frame_num, frame_res, self.phases, WL, tt_min, tt_max, '\t',
                    sourceName=self._source_name(frame_num))
            except Exception as e:
                print(f"  Could not save fitParam for frame {frame_num}: {e}")

        print(f"Saved {n_saved} fit-curve file(s) to: {curves_dir}")
        print("All results saved.")







# ----------------------------------------------------------------------------
# Application-wide stylesheet: white outline for checkbox indicators
# (better contrast on the dark theme).
# ----------------------------------------------------------------------------
def _apply_app_stylesheet(app):
    """White outline for QCheckBox indicators, applied to the whole
    application (main window and all dialogs). Nothing else is restyled.

    The indicator size is taken from the active style (Fusion) so it keeps
    its current size; only the colour changes. Once a
    stylesheet styles ::indicator, Qt no longer draws the native tick, so a
    tick pixmap is rendered here with QPainter (no SVG plugin needed, which
    keeps PyInstaller builds working).
    """
    import tempfile
    from PyQt5.QtWidgets import QStyle
    from PyQt5.QtGui import QPixmap, QPainter, QPen
    from PyQt5.QtCore import QPointF

    style = app.style()
    ind = style.pixelMetric(QStyle.PM_IndicatorWidth) or 14
    BORDER = '#ffffff'
    BG = 'rgb(61, 61, 61)'      # same as the main-window background

    # Render a white tick once and store it in the temp dir.
    tick_path = os.path.join(tempfile.gettempdir(), f'exodus_tick_{ind}.png')
    try:
        if not os.path.isfile(tick_path):
            scale = 4                              # supersample, then let Qt scale
            pm = QPixmap(ind * scale, ind * scale)
            pm.fill(Qt.transparent)
            qp = QPainter(pm)
            qp.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor('#ffffff'))
            pen.setWidthF(1.8 * scale)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            qp.setPen(pen)
            w = ind * scale
            qp.drawPolyline(QPointF(0.22 * w, 0.52 * w),
                            QPointF(0.42 * w, 0.72 * w),
                            QPointF(0.78 * w, 0.30 * w))
            qp.end()
            pm.save(tick_path, 'PNG')
        tick_url = tick_path.replace(os.sep, '/')
        tick_rule = f'image: url("{tick_url}");'
    except Exception as e:
        print(f"[style] could not create checkbox tick image: {e}")
        tick_rule = ''

    qss = f"""
    QCheckBox::indicator {{
        width: {ind - 2}px;
        height: {ind - 2}px;
        border: 1px solid {BORDER};
        border-radius: 2px;
        background-color: {BG};
    }}
    QCheckBox::indicator:checked {{
        {tick_rule}
    }}
    QCheckBox::indicator:disabled {{
        border: 1px solid #8a8a8a;
    }}
    """
    app.setStyleSheet(qss)

#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#





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
    _apply_app_stylesheet(app)

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
    window.setWindowTitle('Exodus v0.1.4')
    window.setWindowIcon(app_icon)   # belt and braces

    window.show()
    sys.exit(app.exec_())