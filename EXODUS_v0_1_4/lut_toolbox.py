# lut_toolbox.py for EXODUS v0.1.4
#
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Bernhard Massani
#
# Adapted from Dioptas - MIT License,
# Copyright (c) 2014-2019 GSECARS, University of Chicago;
# Copyright (c) 2015-2018 University of Cologne;
# Copyright (c) 2019-2020 DESY.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. It is distributed WITHOUT ANY WARRANTY; see the LICENSE
# file or <https://www.gnu.org/licenses/> for details.
#
# ----------------------------------------------------------------------------
#
# Colour look-up-table (contrast / colour-map) control for the 2D plot,
# modelled on the horizontal HistogramLUTItem of Dioptas
# (dioptas/widgets/plot_widgets/HistogramLUTItem.py).
#
# Differences to Dioptas
# ----------------------
# * Dioptas keeps its levels on a LOG intensity scale. EXODUS already
#   transforms the image itself (Lin / Log / Sqrt buttons) and its
#   background-subtracted data contain negative values, so the levels here
#   are linear in whatever the 2D plot currently shows.
# * No colormap pop-up / normalisation label: EXODUS has its own colormap
#   and intensity-scale buttons.
#
# Drop-in replacement for the parts of pg.HistogramLUTWidget that EXODUS
# uses: setImageItem(), setLevels(), getLevels(), .gradient, .region, .item.
#
#
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import numpy as np
import pyqtgraph as pg
import pyqtgraph.functions as fn
from pyqtgraph.graphicsItems.GradientEditorItem import GradientEditorItem
from pyqtgraph.graphicsItems.GraphicsWidget import GraphicsWidget
from pyqtgraph.graphicsItems.LinearRegionItem import LinearRegionItem
from pyqtgraph.Point import Point
from PyQt5 import QtCore, QtWidgets

# ---- look (Dioptas defaults) -------------------------------------------------
HIST_LINE = (50, 150, 50)        # green histogram outline
HIST_FILL = (100, 100, 200)      # blue fill
HIST_HEIGHT = 30                 # px, histogram strip
WIDGET_HEIGHT = 62               # px, gradient + histogram
BACKGROUND = (61, 61, 61)        # same as the EXODUS plots
N_BINS = 1500


class DioptasHistogramLUTItem(GraphicsWidget):
    """Horizontal histogram / gradient LUT in the Dioptas style."""

    sigLookupTableChanged = QtCore.pyqtSignal(object)
    sigLevelsChanged = QtCore.pyqtSignal(object)
    sigLevelChangeFinished = QtCore.pyqtSignal(object)

    def __init__(self, image=None):
        GraphicsWidget.__init__(self)
        self.lut = None
        self.imageItem = None

        self.layout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(0)

        self.vb = pg.ViewBox()
        self.vb.setMouseEnabled(x=False, y=False)
        self.vb.setMenuEnabled(False)
        self.vb.setFixedHeight(HIST_HEIGHT)

        self.gradient = GradientEditorItem()
        self.gradient.setOrientation('top')
        self.gradient.loadPreset('grey')

        self.region = LinearRegionItem([0, 1], LinearRegionItem.Vertical)
        self.region.setZValue(1000)

        self.layout.addItem(self.gradient, 0, 0)
        self.layout.addItem(self.vb, 1, 0)
        self.gradient.setFlag(self.gradient.ItemStacksBehindParent)
        self.vb.setFlag(self.gradient.ItemStacksBehindParent)
        self.vb.addItem(self.region)

        self.plot = pg.PlotDataItem()
        self.plot.setPen(pg.mkPen(color=HIST_LINE, width=1))
        self.plot.setFillLevel(0.0)
        self.plot.setFillBrush(HIST_FILL)
        self.vb.addItem(self.plot)

        self.gradient.sigGradientChanged.connect(self.gradientChanged)
        self.region.sigRegionChanged.connect(self.regionChanging)
        self.region.sigRegionChangeFinished.connect(self.regionChanged)
        self.vb.sigRangeChanged.connect(self.update)

        # The histogram strip never zooms / pans (as in Dioptas)
        self.vb.mouseClickEvent = self._ignore
        self.vb.mouseDragEvent = self._ignore
        self.vb.mouseDoubleClickEvent = self._ignore
        self.vb.wheelEvent = self._ignore

        if image is not None:
            self.setImageItem(image)

    @staticmethod
    def _ignore(*_args, **_kw):
        pass

    # ---- connector lines (the Dioptas "trapezoid") ----------------------
    def paint(self, p, *args):
        rgn = self.getLevels()
        cy = self.vb.viewRect().center().y()
        p1 = self.vb.mapFromViewToItem(self, Point(rgn[0], cy))
        p2 = self.vb.mapFromViewToItem(self, Point(rgn[1], cy))
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        for pen in (fn.mkPen('k', width=3), self.region.lines[0].pen):
            p.setPen(pen)
            p.drawLine(p1, gradRect.bottomLeft())
            p.drawLine(p2, gradRect.bottomRight())
            p.drawLine(gradRect.bottomLeft(), gradRect.topLeft())
            p.drawLine(gradRect.bottomRight(), gradRect.topRight())

    # ---- image link -------------------------------------------------------
    def setImageItem(self, img):
        """Link to an ImageItem. Safe to call repeatedly (EXODUS re-links on
        every 2D refresh)."""
        if self.imageItem is not None and self.imageItem is not img:
            try:
                self.imageItem.sigImageChanged.disconnect(self.imageChanged)
            except (TypeError, RuntimeError):
                pass
        if self.imageItem is not img:
            img.sigImageChanged.connect(self.imageChanged)
        self.imageItem = img
        img.setLookupTable(self.getLookupTable)
        self.regionChanging()
        self.imageChanged()

    def gradientChanged(self):
        if self.imageItem is not None:
            if self.gradient.isLookupTrivial():
                self.imageItem.setLookupTable(None)
            else:
                self.imageItem.setLookupTable(self.getLookupTable)
        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def getLookupTable(self, img=None, n=None, alpha=None):
        if n is None:
            n = 256 if (img is not None and img.dtype == np.uint8) else 512
        if self.lut is None:
            self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut

    # ---- levels -------------------------------------------------------
    def regionChanging(self):
        if self.imageItem is not None:
            self.imageItem.setLevels(self.getLevels())
        self.sigLevelsChanged.emit(self)
        self.update()

    def regionChanged(self):
        self.sigLevelChangeFinished.emit(self)

    def getLevels(self):
        a, b = self.region.getRegion()
        return (min(a, b), max(a, b))

    def setLevels(self, mn, mx=None):
        if mx is None:            # also accept setLevels([mn, mx])
            mn, mx = mn
        mn, mx = float(mn), float(mx)
        self.region.setRegion([mn, mx])
        if mx > mn:
            # Show the levels plus a margin, so both edges can be grabbed
            pad = 0.02 * (mx - mn)
            self.vb.setXRange(mn - pad, mx + pad, padding=0)
        if self.imageItem is not None:
            self.imageItem.setLevels((mn, mx))

    # ---- histogram ------------------------------------------------------
    def imageChanged(self, *_args, **_kw):
        if self.imageItem is None:
            return
        data = self.imageItem.image
        if data is None:
            return
        x, y = histogram_data(data)
        if x is None:
            self.plot.clear()
            return
        self.plot.setData(x, y)
        self.vb.setYRange(0, float(np.max(y)) * 1.05, padding=0)


def histogram_data(data):
    """(bin left edges, log(counts)) of the finite pixels, empty bins
    dropped - same representation as Dioptas, but on a linear x-axis."""
    d = np.asarray(data, dtype=np.float64).ravel()
    if d.size > 2_000_000:                    # subsample huge stacks
        d = d[:: int(np.ceil(d.size / 2_000_000))]
    d = d[np.isfinite(d)]
    if d.size == 0 or d.min() == d.max():
        return None, None
    hist, edges = np.histogram(d, bins=N_BINS)
    keep = hist > 0
    return edges[:-1][keep], np.log(hist[keep]) + 1.0   # +1: single counts visible


class DioptasLUTWidget(pg.GraphicsView):
    """QWidget wrapper (drop-in for pg.HistogramLUTWidget in EXODUS)."""

    def __init__(self, parent=None, image=None):
        pg.GraphicsView.__init__(self, parent, background=BACKGROUND)
        self.item = DioptasHistogramLUTItem(image)
        self.setCentralItem(self.item)
        self.setFixedHeight(WIDGET_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)
        self.setToolTip("Drag the region edges in the histogram to set the "
                        "contrast. Right-click the colour bar for presets; "
                        "click below it to add colour ticks.")

    # Delegate the API EXODUS uses
    def __getattr__(self, attr):
        if attr in ('setImageItem', 'setLevels', 'getLevels', 'gradient',
                    'region', 'getLookupTable', 'sigLevelsChanged',
                    'sigLevelChangeFinished', 'sigLookupTableChanged'):
            return getattr(self.item, attr)
        raise AttributeError(attr)
