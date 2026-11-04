# EXODUS changelog

## v0.1.4b (hot fix)

Files changed: EXODUS_main.py, EXODUS_core.py, results_toolbox.py,
CHANGELOG.md. New file: export_2d_toolbox.py. All other files are unchanged
from v0.1.4.

### Fixed
- 'Save All' wrote only the tick-mark columns (and the lattParam rows /
  fitParam blocks) of the phases refined in the LAST frame of the batch run
  for every frame. The writers used the global phases['phases_used'], which
  the batch loop overwrites frame by frame. Every fit now stores its own
  phase list ('_phases_used') and the writers use it, so each frame is
  saved with its own phases. Same fix for 'Save' after a batch run.
- 2D plot was black in the .exe (and in the first run of a fresh Spyder
  kernel), also with no data loaded. The pyqtgraph background option was
  set in MainWindow.__init__, AFTER EXODUS_GUI_ui had created the 2D plot.
  It is now set at import time, before the window is built, and the 2D
  plot is styled explicitly. Not a PyInstaller problem.
- Cancelling a batch run now goes through the normal end-of-run path
  (per-frame overrides cleared, warnings shown, 2D overlay refreshed).
- A frame whose fit raises no longer leaves an empty entry in the
  batch results.
- The arrows of the pressure-guess phase tabs did nothing: Qt's scroll
  arrows only slide the tab headers, and every relayout of the panel slid
  them back to the selected tab. The arrows now select the previous / next
  phase tab.
- A batch run did not replace an earlier single fit ('Fit JCPDS(s)') of the
  same frame: that older fit kept priority for its frame in the pattern
  panel, the fit statistics label, Plot Data, EoS Fit and the 'Fitted'
  column of the row 'Edit' pop-up. A lattice parameter fixed in the batch
  run therefore looked refined there. A batch run now discards the earlier
  single fit.
- Lattice parameters held fixed in a fit are listed in the console
  ("Frame N, phase X: lattice parameters held fixed -> b") and in the
  header of the *_fitParam file.
- Pressure Guess needed at least two rows in one tab and otherwise did
  nothing except print a console message. One row is now enough (that
  pressure is used for every frame of the phase); with no rows at all a
  pop-up says so.

### Changed
- Pressure-guess rows now apply in every fit mode. A row overrides the
  start of EXACTLY its frame in Single, Sequential, Pressure Walk and
  Pressure Guess fits: its pressure sets the start cell, its 'Fix' holds the
  whole lattice and its 'Edit' overrides set / fix single parameters and fit
  inputs. Frames without a row behave as before: Sequential starts from the
  previous frame's cell, Pressure Walk from the previous pressure + Delta P
  (so the frame after a row starts from that row's fit), the single fit from
  the JCPDS-table pressure. Only Pressure Guess interpolates between rows
  (constant beyond the first / last row; one row = that pressure
  throughout). The rows used are printed in the console at the start of
  every batch run. (The header comment of fix_parameters_toolbox.py still
  says "Pressure Guess only".)
- Covariance failure: lattice-parameter errors that cannot be obtained are
  set to 0 (EXODUS_core.COVARIANCE_FAILURE_ERROR), as are the V and P errors
  that depend on them and the per-reflection errors in *_fitParam. The
  sqrt(reduced chi^2) fallback is kept, commented out, in fit_pawley.
  New pop-up: "Covariance matrix failed for the following frames. No fitting
  error can be obtained. Errors set to 0." Output-file headers updated.
- Batch fits: a phase the fit LOST in a frame (all refined amplitudes ~0,
  no reflection in the window, or a lattice parameter on its max-shift
  bound) is flagged ('_phase_flags', console, file header, pop-up at the end
  of the run). Its lost cell is no longer used as the start for the next
  frame in Sequential / Pressure Walk mode; the last good cell (or the
  JCPDS-table pressure) is used instead, so one bad frame no longer drops
  the phase for the rest of the run.
- 'Save 2D Plot' opens an export pop-up (export_2d_toolbox): crosshair on /
  off, axes on / off (data area only), fitted-peak symbols on / off with
  shape and size, wavelength label with corner and font size, 2theta and
  pattern range (current view / full data / custom), image size in pixels,
  axis font size. PNG, TIFF or JPEG. Rendered off-screen; the options are
  remembered for the next export.

## v0.1.4

### New
- Pseudo-Voigt peak profile (Gaussian / Pseudo Voigt checkboxes) with one
  global mixing parameter eta, held fixed at the value in the GUI.
- Caglioti peak broadening, FWHM^2 = U tan^2(theta) + V tan(theta) + W, with
  refined global U, V, W (start values in the GUI).
- Faster fitting: every reflection is evaluated only on the data points near
  its centre (+-3 FWHM Gaussian, +-30 FWHM pseudo-Voigt).
- Fewer frames without a covariance matrix:
  * exactly overlapping reflections (same d for every cell of the crystal
    system, e.g. (300)/(221) in cubic) share one refined amplitude, split
    in the ratio of their JCPDS intensities (Pawley constraint);
  * the fit window is limited to the 2theta range of the data, so no
    reflection beyond the last data point is refined;
  * if lmfit's covariance is missing or singular, it is rebuilt from the
    Jacobian without the parameters the data cannot determine (reported in
    the console) and all other parameters keep proper errors.
- Pop-up warning when a lattice parameter still has no error from the
  covariance and its error falls back to sqrt(chi^2) (one pop-up per batch
  run).
- EoS Fit: the K0'' implied by BM2, BM3, Vinet, natural strain and
  Murnaghan is shown for every model; refined / fixed / implied parameters
  are labelled identically in the error column, statistics, legend and CSV.

### Changed
- Layout: profile box (Gaussian / Pseudo Voigt, eta) in the left column;
  peak-broadening box (incl. Caglioti U, V, W) and the amplitude / sigma
  box in a new row below the JCPDS table; new start-up window sizes.
- The former 'caglioti' sigma mode (sigma0 / cos(theta)) is now called
  'scherrer'; 'caglioti' is the U, V, W law above.
- sigma(V) is computed once per fit (full covariance) and stored with the
  results, so output files, Plot Data and EoS Fit use the same value.
- EoS Fit curve, residuals and saved P_fit use the exact fitted parameters
  (not the rounded spin-box values).
- unit_cell_volume() returns None for an invalid cell.
- d-spacings for all crystal systems from the reciprocal metric tensor.
- Code reorganised into *_toolbox.py modules (see the header of
  EXODUS_main.py); functions in snake_case, arguments in camelCase; the
  refinement is named Pawley throughout. Retired code is kept, commented
  out, in legacy_toolbox.py.

### Fixed
- 'Save' (single fit) refused to save when the pattern had been selected
  with the spin box, 2D plot or keys instead of the file list (it compared
  the fit with a stale pattern index).
- CIF -> JCPDS: R-centred structures (given on hexagonal axes by pymatgen)
  were labelled RHOMBOHEDRAL, which EXODUS reads as the rhombohedral
  setting; they are now written as HEXAGONAL.
- Background of patterns not yet in the cache ignored the 'Width' spin box
  (fixed 0.35 deg).
- Orthorhombic crystal system spelled 'ORTHORHOMBIC' (legacy 'ORTHOROMBIC'
  still accepted on load).
