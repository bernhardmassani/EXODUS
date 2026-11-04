# export_2d_toolbox.py for EXODUS v0.1.4b
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
# "Save 2D Plot" pop-up: choose what goes into the exported image of the 2D
# plot, then render it OFF-SCREEN (the 2D plot in the main window is never
# touched, so zoom / crosshair / dot sizes there stay as they are).
#
# Options
#   * green dashed crosshair lines on / off
#   * axes on / off (off = only the data rectangle is saved)
#   * fitted-peak symbols on / off, their shape and size
#   * wavelength label (lambda = ... A) on / off, corner, font size
#   * x (2theta) and y (pattern) range
#   * image size in pixels, axis font size
#
# The main window hands over a plain `source` dict (see
# MainWindow._export_2d_source); this module never reads the main window.
#
#   source = {
#     'image':      2D ndarray exactly as shown on the ImageItem,
#     'lut':        colour look-up table (N x 3/4 uint8) or None,
#     'levels':     (low, high) of the LUT,
#     'transform':  QTransform of the ImageItem (2theta / pattern geometry),
#     'overlays':   [{'x': array, 'y': array, 'color': QColor}, ...],
#     'crosshair':  {'x': 2theta or None, 'y': pattern line or None},
#     'WL':         wavelength in A or None,
#     'full_range': (x0, x1, y0, y1)  - the whole data set,
#     'view_range': (x0, x1, y0, y1)  - what the 2D plot shows now,
#     'labels':     (bottom label, left label),
#     'background': (r, g, b),
#     'default_name': file name suggestion without extension,
#   }
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


# ============================================================================
#                                CONSTANTS
# ============================================================================
CROSSHAIR_COLOR = (0, 230, 0)

# label shown in the pop-up -> pyqtgraph symbol code
SYMBOLS = {
    'Circle': 'o',
    'Square': 's',
    'Diamond': 'd',
    'Triangle': 't',
    'Plus': '+',
    'Cross': 'x',
    'Star': 'star',
    'Vertical bar': '|',
}

CORNERS = ('Top left', 'Top right', 'Bottom left', 'Bottom right')

DEFAULT_SETTINGS = {
    'crosshair': False,
    'axes': True,
    'peaks': True,
    'symbol': 'Circle',
    'symbol_size': 4.0,
    'wl': False,
    'wl_corner': 'Top left',
    'wl_font': 12,
    'width': 1600,
    'height': 1200,
    'axis_font': 12,
    'range_mode': 'view',      # 'view' | 'full' | 'custom'
    'x0': None, 'x1': None, 'y0': None, 'y1': None,
}

FILE_FILTER = ("PNG Image (*.png);;TIFF Image (*.tiff *.tif);;"
               "JPEG Image (*.jpg *.jpeg);;All Files (*)")


def _vertical_bar_symbol():
    """'|' marker (a short vertical tick, like the 1D tick marks)."""
    path = QtGui.QPainterPath()
    path.moveTo(0.0, -0.5)
    path.lineTo(0.0, 0.5)
    return path


# ============================================================================
#                                RENDERING
# ============================================================================
def render_2d_export(source, settings, outPath):
    """Render the 2D plot described by `source` with `settings` into
    `outPath` (format from the extension). Raises on failure."""
    from pyqtgraph.exporters import ImageExporter

    s = dict(DEFAULT_SETTINGS)
    s.update(settings or {})
    W, H = int(s['width']), int(s['height'])
    bg = tuple(source.get('background', (61, 61, 61)))
    fg = (255, 255, 255)

    w = pg.PlotWidget(background=bg)
    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.setFixedSize(W, H)
    pi = w.getPlotItem()
    vb = pi.getViewBox()
    vb.setMouseEnabled(x=False, y=False)
    pi.hideButtons()

    # ---- image (same data, LUT, levels and geometry as on screen) ----------
    img = pg.ImageItem()
    img.setImage(np.asarray(source['image']), autoLevels=False)
    if source.get('lut') is not None:
        img.setLookupTable(np.asarray(source['lut']))
    lv = source.get('levels')
    if lv is not None:
        img.setLevels(lv)
    if source.get('transform') is not None:
        img.setTransform(source['transform'])
    pi.addItem(img)

    # ---- fitted-peak symbols -----------------------------------------------
    if s['peaks']:
        code = SYMBOLS.get(s['symbol'], 'o')
        sym = _vertical_bar_symbol() if code == '|' else code
        line_like = code in ('+', 'x', '|')
        for ov in source.get('overlays', []):
            col = QtGui.QColor(ov['color'])
            kw = dict(x=np.asarray(ov['x'], float), y=np.asarray(ov['y'], float),
                      size=float(s['symbol_size']), symbol=sym)
            if line_like:     # outline-only symbols need a pen
                kw.update(pen=pg.mkPen(col, width=max(1.0, s['symbol_size'] / 6.0)),
                          brush=pg.mkBrush(None))
            else:
                kw.update(pen=None, brush=pg.mkBrush(col))
            pi.addItem(pg.ScatterPlotItem(**kw))

    # ---- green crosshair ---------------------------------------------------
    ch = source.get('crosshair') or {}
    if s['crosshair']:
        pen = pg.mkPen(CROSSHAIR_COLOR, width=max(2.0, W / 600.0),
                       style=Qt.DashLine)
        if ch.get('x') is not None:
            pi.addItem(pg.InfiniteLine(pos=float(ch['x']), angle=90, pen=pen),
                       ignoreBounds=True)
        if ch.get('y') is not None:
            pi.addItem(pg.InfiniteLine(pos=float(ch['y']), angle=0, pen=pen),
                       ignoreBounds=True)

    # ---- axes ----------------------------------------------------------------
    if s['axes']:
        font = QtGui.QFont()
        font.setPointSize(int(s['axis_font']))
        style = {'color': '#ffffff', 'font-size': f"{int(s['axis_font'])}pt"}
        bottom, left = source.get('labels', ('Two Theta (°)', 'Pattern No.'))
        pi.setLabel('bottom', bottom, **style)
        pi.setLabel('left', left, **style)
        for nm in ('bottom', 'left'):
            ax = pi.getAxis(nm)
            ax.setPen(pg.mkPen(fg))
            ax.setTextPen(pg.mkPen(fg))
            ax.setStyle(tickFont=font)
        pi.layout.setContentsMargins(6, 6, 12, 6)
        target = pi
    else:
        pi.hideAxis('bottom')
        pi.hideAxis('left')
        pi.layout.setContentsMargins(0, 0, 0, 0)
        w.setContentsMargins(0, 0, 0, 0)
        target = vb

    # ---- range ----------------------------------------------------------------
    x0, x1, y0, y1 = (float(s['x0']), float(s['x1']),
                      float(s['y0']), float(s['y1']))
    vb.setRange(xRange=(min(x0, x1), max(x0, x1)),
                yRange=(min(y0, y1), max(y0, y1)), padding=0)

    w.show()
    QtWidgets.QApplication.processEvents()

    # ---- wavelength label (pinned to a corner of the data area) ------------
    if s['wl'] and source.get('WL') is not None:
        corner = s['wl_corner']
        anchor = (1 if 'right' in corner else 0, 1 if 'Bottom' in corner else 0)
        lbl = pg.TextItem(anchor=anchor, color=fg)
        lbl.setHtml(f'<div style="font-size:{int(s["wl_font"])}pt; color:#ffffff;">'
                    f'λ = {float(source["WL"]):.4f} Å</div>')
        lbl.setParentItem(vb)
        m = 8
        xpos = vb.width() - m if anchor[0] else m
        ypos = vb.height() - m if anchor[1] else m
        lbl.setPos(xpos, ypos)
        lbl.setZValue(2000)
        QtWidgets.QApplication.processEvents()

    exp = ImageExporter(target)
    p = exp.parameters()
    try:
        p['background'] = QtGui.QColor(*bg)
    except Exception:
        pass
    # 1 exported pixel per pixel of the off-screen widget
    p['width'] = int(round(target.boundingRect().width())) or W
    ok = exp.export(outPath)
    w.close()
    w.deleteLater()
    if ok is False or not os.path.isfile(outPath):
        raise IOError(f"could not write {outPath}")
    return outPath


# ============================================================================
#                                 POP-UP
# ============================================================================
class Export2DDialog(QtWidgets.QDialog):
    """Options for the 2D-plot export (see module header)."""

    def __init__(self, parent, source, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Save 2D plot")
        self.source = source
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(settings or {})
        self.saved_path = None
        self._build()
        self._load(self.settings)

    # ---- layout ---------------------------------------------------------------
    @staticmethod
    def _dspin(lo, hi, dec, step):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setSingleStep(step)
        return sp

    @staticmethod
    def _ispin(lo, hi, step=1):
        sp = QtWidgets.QSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        return sp

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        # -- content -----------------------------------------------------------
        g1 = QtWidgets.QGroupBox("Content")
        f1 = QtWidgets.QGridLayout(g1)
        self.chk_cross = QtWidgets.QCheckBox("Green dashed crosshair lines")
        has_ch = any(v is not None for v in (self.source.get('crosshair') or {}).values())
        self.chk_cross.setEnabled(has_ch)
        if not has_ch:
            self.chk_cross.setToolTip("No 2θ / pattern selected in the 2D plot.")
        self.chk_axes = QtWidgets.QCheckBox("Axes and labels "
                                            "(untick: save only the data area)")
        self.spin_axis_font = self._ispin(6, 48)
        self.chk_wl = QtWidgets.QCheckBox("Wavelength label")
        self.chk_wl.setEnabled(self.source.get('WL') is not None)
        if self.source.get('WL') is None:
            self.chk_wl.setToolTip("No PONI / wavelength loaded.")
        self.cmb_corner = QtWidgets.QComboBox()
        self.cmb_corner.addItems(CORNERS)
        self.spin_wl_font = self._ispin(6, 72)
        f1.addWidget(self.chk_cross, 0, 0, 1, 4)
        f1.addWidget(self.chk_axes, 1, 0, 1, 2)
        f1.addWidget(QtWidgets.QLabel("Font (pt)"), 1, 2)
        f1.addWidget(self.spin_axis_font, 1, 3)
        f1.addWidget(self.chk_wl, 2, 0)
        f1.addWidget(self.cmb_corner, 2, 1)
        f1.addWidget(QtWidgets.QLabel("Font (pt)"), 2, 2)
        f1.addWidget(self.spin_wl_font, 2, 3)
        lay.addWidget(g1)

        # -- symbols ---------------------------------------------------------
        g2 = QtWidgets.QGroupBox("Fitted-peak symbols")
        f2 = QtWidgets.QGridLayout(g2)
        self.chk_peaks = QtWidgets.QCheckBox("Show")
        self.chk_peaks.setEnabled(bool(self.source.get('overlays')))
        if not self.source.get('overlays'):
            self.chk_peaks.setToolTip("No batch-fit results to show.")
        self.cmb_symbol = QtWidgets.QComboBox()
        self.cmb_symbol.addItems(list(SYMBOLS))
        self.spin_symbol = self._dspin(0.5, 50.0, 1, 0.5)
        f2.addWidget(self.chk_peaks, 0, 0)
        f2.addWidget(QtWidgets.QLabel("Shape"), 0, 1)
        f2.addWidget(self.cmb_symbol, 0, 2)
        f2.addWidget(QtWidgets.QLabel("Size (px)"), 0, 3)
        f2.addWidget(self.spin_symbol, 0, 4)
        lay.addWidget(g2)

        # -- range -----------------------------------------------------------
        g3 = QtWidgets.QGroupBox("Plot range")
        f3 = QtWidgets.QGridLayout(g3)
        self.spin_x0 = self._dspin(-1e4, 1e4, 3, 0.1)
        self.spin_x1 = self._dspin(-1e4, 1e4, 3, 0.1)
        self.spin_y0 = self._dspin(-1e7, 1e7, 1, 1.0)
        self.spin_y1 = self._dspin(-1e7, 1e7, 1, 1.0)
        f3.addWidget(QtWidgets.QLabel("2θ from"), 0, 0)
        f3.addWidget(self.spin_x0, 0, 1)
        f3.addWidget(QtWidgets.QLabel("to"), 0, 2)
        f3.addWidget(self.spin_x1, 0, 3)
        f3.addWidget(QtWidgets.QLabel("Pattern from"), 1, 0)
        f3.addWidget(self.spin_y0, 1, 1)
        f3.addWidget(QtWidgets.QLabel("to"), 1, 2)
        f3.addWidget(self.spin_y1, 1, 3)
        b_view = QtWidgets.QPushButton("Current view")
        b_full = QtWidgets.QPushButton("Full data")
        b_view.clicked.connect(lambda: self._set_range(self.source['view_range']))
        b_full.clicked.connect(lambda: self._set_range(self.source['full_range']))
        f3.addWidget(b_view, 0, 4)
        f3.addWidget(b_full, 1, 4)
        lay.addWidget(g3)

        # -- image size ----------------------------------------------------------
        g4 = QtWidgets.QGroupBox("Image size")
        f4 = QtWidgets.QHBoxLayout(g4)
        self.spin_w = self._ispin(100, 10000, 50)
        self.spin_h = self._ispin(100, 10000, 50)
        f4.addWidget(QtWidgets.QLabel("Width (px)"))
        f4.addWidget(self.spin_w)
        f4.addSpacing(12)
        f4.addWidget(QtWidgets.QLabel("Height (px)"))
        f4.addWidget(self.spin_h)
        f4.addStretch(1)
        lay.addWidget(g4)

        # -- buttons -----------------------------------------------------------
        bb = QtWidgets.QDialogButtonBox()
        self.btn_save = bb.addButton("Save...", QtWidgets.QDialogButtonBox.AcceptRole)
        bb.addButton(QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self.chk_peaks.toggled.connect(self._refresh_enabled)
        self.chk_wl.toggled.connect(self._refresh_enabled)
        self.chk_axes.toggled.connect(self._refresh_enabled)

    def _refresh_enabled(self, *_a):
        on = self.chk_peaks.isChecked() and self.chk_peaks.isEnabled()
        self.cmb_symbol.setEnabled(on)
        self.spin_symbol.setEnabled(on)
        on = self.chk_wl.isChecked() and self.chk_wl.isEnabled()
        self.cmb_corner.setEnabled(on)
        self.spin_wl_font.setEnabled(on)
        self.spin_axis_font.setEnabled(self.chk_axes.isChecked())

    def _set_range(self, rng):
        x0, x1, y0, y1 = rng
        self.spin_x0.setValue(x0)
        self.spin_x1.setValue(x1)
        self.spin_y0.setValue(y0)
        self.spin_y1.setValue(y1)

    # ---- settings <-> widgets -------------------------------------------
    def _load(self, s):
        self.chk_cross.setChecked(bool(s['crosshair']) and self.chk_cross.isEnabled())
        self.chk_axes.setChecked(bool(s['axes']))
        self.chk_peaks.setChecked(bool(s['peaks']) and self.chk_peaks.isEnabled())
        self.cmb_symbol.setCurrentText(s['symbol'] if s['symbol'] in SYMBOLS else 'Circle')
        self.spin_symbol.setValue(float(s['symbol_size']))
        self.chk_wl.setChecked(bool(s['wl']) and self.chk_wl.isEnabled())
        self.cmb_corner.setCurrentText(s['wl_corner'] if s['wl_corner'] in CORNERS
                                       else CORNERS[0])
        self.spin_wl_font.setValue(int(s['wl_font']))
        self.spin_axis_font.setValue(int(s['axis_font']))
        self.spin_w.setValue(int(s['width']))
        self.spin_h.setValue(int(s['height']))
        # Range: last custom range only if it still overlaps the data,
        # otherwise the current view of the 2D plot.
        if s.get('range_mode') == 'custom' and None not in (
                s['x0'], s['x1'], s['y0'], s['y1']):
            self._set_range((s['x0'], s['x1'], s['y0'], s['y1']))
        else:
            self._set_range(self.source['view_range'])
        self._refresh_enabled()

    def _collect(self):
        rng = (self.spin_x0.value(), self.spin_x1.value(),
               self.spin_y0.value(), self.spin_y1.value())
        view = tuple(round(v, 3) for v in self.source['view_range'])
        mode = 'view' if tuple(round(v, 3) for v in rng) == view else 'custom'
        return {
            'crosshair': self.chk_cross.isChecked() and self.chk_cross.isEnabled(),
            'axes': self.chk_axes.isChecked(),
            'peaks': self.chk_peaks.isChecked() and self.chk_peaks.isEnabled(),
            'symbol': self.cmb_symbol.currentText(),
            'symbol_size': self.spin_symbol.value(),
            'wl': self.chk_wl.isChecked() and self.chk_wl.isEnabled(),
            'wl_corner': self.cmb_corner.currentText(),
            'wl_font': self.spin_wl_font.value(),
            'width': self.spin_w.value(),
            'height': self.spin_h.value(),
            'axis_font': self.spin_axis_font.value(),
            'range_mode': mode,
            'x0': rng[0], 'x1': rng[1], 'y0': rng[2], 'y1': rng[3],
        }

    # ---- save -------------------------------------------------------------
    def _on_save(self):
        s = self._collect()
        if s['x0'] == s['x1'] or s['y0'] == s['y1']:
            QtWidgets.QMessageBox.warning(self, "Save 2D plot",
                                          "The plot range is empty.")
            return
        path, flt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 2D plot", self.source.get('default_name', '2D_plot'),
            FILE_FILTER)
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.png', '.tif', '.tiff', '.jpg', '.jpeg'):
            path += ('.tiff' if 'TIFF' in flt else
                     '.jpg' if 'JPEG' in flt else '.png')
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            render_2d_export(self.source, s, path)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Save 2D plot",
                                           f"Could not save the 2D plot:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        print(f"2D plot saved to: {path}")
        self.settings = s
        self.saved_path = path
        self.accept()


def open_export_dialog(parent, source, settings=None):
    """Show the pop-up (modal). Returns the settings used if an image was
    saved (the caller keeps them for the next export), else None."""
    dlg = Export2DDialog(parent, source, settings)
    if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.saved_path:
        return dlg.settings
    return None
