# data_toolbox.py for EXODUS v0.1.4
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
# Loading of experiment files and in-memory caching of patterns (no Qt):
#
#   * load_poni     wavelength from a pyFAI/Dioptas .poni file
#   * load_data      one 1D pattern (.fxye, .xy, .dat, .chi), cropped to a
#                   2theta window
#   * DataManager   two-layer cache (raw patterns, background-subtracted
#                   patterns) with parallel pre-loading and parallel
#                   background computation for the 2D stack
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import bisect
import os

import numpy as np
import pandas as pd

import background_toolbox as bgt


# ============================================================================
#                             PONI (WAVELENGTH)
# ============================================================================
def load_poni(poniFile):
    file_name = poniFile
    file_path = str(file_name)

    # Loads file
    df = pd.read_csv(file_path, header=0, delimiter = ' ')

    # Drops rows with null value
    #df.dropna(axis=0)

    # Find the row where 'Wavelength:' is present
    wavelength = df[df.apply(lambda row: row.astype(str).str.contains('Wavelength:').any(), axis=1)]

    # Initialise so a missing 'Wavelength:' line produces a clear error
    # rather than UnboundLocalError when we try to return WL below.
    WL = None

    # Finds the wavelength and returns it in Angstroms
    # PONI files store wavelength in metres; 1 m = 1e10 A. The literal
    # below is written as 10e9 (= 1.0 * 10**10) for historical reasons,
    # but it IS the correct conversion factor.
    for index, row in wavelength.iterrows():
        WL = row.iloc[1]
        WL = float(WL) * 1e10  # m -> A

    if WL is None:
        raise ValueError(
            f"PONI file {file_path!r} contains no 'Wavelength:' line. "
            "Cannot determine wavelength."
        )
    return WL


# ============================================================================
#                                1D PATTERNS
# ============================================================================
def _truncate_range(twoTheta, valueInt, errorInt, twoThetaMin, twoThetaMax):
    '''
    Helper: clips a (twoTheta, intensity, error) triplet to the given range.
    Inputs may be lists or arrays; outputs are numpy arrays. twoTheta MUST be
    monotonically increasing for bisect to work (which is true for all the
    XRD file formats we currently support).
    '''
    twoTheta = list(twoTheta)
    valueInt = list(valueInt)
    errorInt = list(errorInt)

    indexLow = bisect.bisect(twoTheta, twoThetaMin)
    indexHigh = bisect.bisect(twoTheta, twoThetaMax)

    twoTheta = np.array(twoTheta[indexLow:indexHigh])
    valueInt = np.array(valueInt[indexLow:indexHigh])
    errorInt = np.array(errorInt[indexLow:indexHigh])

    return twoTheta, valueInt, errorInt


def _load_fxye(filePath):
    '''
    GSAS-II *.fxye format. ~23-line header, then tab-separated columns:
        col 1 = 2theta (centideg) -- needs *0.01 to get degrees
        col 2 = intensity
        col 3 = error/sigma

    Some GSAS-II writers prefix every data line with a leading tab, which
    pandas reads as an empty column 0 (so the real columns sit at 1, 2, 3).
    Other writers don't, and the columns are at 0, 1, 2. We auto-detect
    which layout this file uses by looking at the data row width.
    '''
    df = pd.read_csv(filePath, header=23, delimiter='\t')
    n_cols = df.shape[1]
    if n_cols >= 4:
        # Leading-tab layout: empty col 0, then 2theta/I/sigma at 1/2/3.
        col_tt, col_I, col_e = 1, 2, 3
    elif n_cols == 3:
        # Plain 3-column layout: 2theta/I/sigma at 0/1/2.
        col_tt, col_I, col_e = 0, 1, 2
    else:
        raise ValueError(
            f"fxye file {filePath!r} has {n_cols} columns after the "
            f"header; expected 3 (twoTheta, intensity, sigma) or 4 "
            f"(with a leading empty column)."
        )
    twoTheta = [i * 0.01 for i in df.iloc[:, col_tt].values.tolist()]
    valueInt = df.iloc[:, col_I].values.tolist()
    errorInt = df.iloc[:, col_e].values.tolist()
    return twoTheta, valueInt, errorInt


def _load_xy(filePath):
    '''
    DIOPTAS / pyFAI *.xy format. Whitespace-delimited 2-column file
    (twoTheta_deg, intensity) preceded by an arbitrary number of comment
    lines starting with '#'. Errors are not stored - we synthesise
    sqrt(|I|) as a sensible default (Poisson counting statistics).
    '''
    data = np.loadtxt(filePath, comments='#')
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


def _load_dat(filePath):
    '''
    Plain *.dat format. Whitespace-delimited 2-column file
    (twoTheta_deg, intensity), normally with no header. Some producers
    add '#' comment lines, so np.loadtxt with comments='#' covers both.
    Errors are synthesised as sqrt(|I|).
    '''
    data = np.loadtxt(filePath, comments='#')
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


def _load_chi(filePath):
    '''
    *.chi format (Fit2D / pyFAI). Fixed 4-line header:
        line 1 = source file path
        line 2 = x-axis label (e.g. '2th_deg')
        line 3 = blank
        line 4 = number of points
    followed by whitespace-delimited (twoTheta, intensity) columns.
    Errors are synthesised as sqrt(|I|).
    '''
    data = np.loadtxt(filePath, skiprows=4)
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


# Registry for the loader dispatch. Lookup is case-insensitive; loaders
# return (twoTheta, valueInt, errorInt) BEFORE truncation.
_LOADERS = {
    '.fxye': _load_fxye,
    '.xy':   _load_xy,
    '.dat':  _load_dat,
    '.chi':  _load_chi,
}


def load_data(filePath, twoThetaMin=0, twoThetaMax=30):
    '''
    Load a single XRD pattern and return (twoTheta, intensity, error) as
    np.arrays clipped to [twoThetaMin, twoThetaMax].

    Supported formats (auto-detected by extension):
        *.fxye - GSAS-II  (23-line header, 3 data columns; centideg)
        *.xy   - DIOPTAS  (#-comment header, 2 data columns; deg)
        *.dat  - plain    (no header, 2 data columns; deg)
        *.chi  - Fit2D    (4-line header, 2 data columns; deg)

    For 2-column formats (.xy, .dat, .chi) the error column is synthesised
    as sqrt(|I|) since it is not stored on disk.
    '''
    ext = os.path.splitext(filePath)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file extension '{ext}' for {filePath}. "
            f"Supported: {', '.join(sorted(_LOADERS.keys()))}"
        )
    twoTheta, valueInt, errorInt = loader(filePath)
    return _truncate_range(twoTheta, valueInt, errorInt,
                           twoThetaMin, twoThetaMax)




# ============================================================================
#                               PATTERN CACHE
# ============================================================================
# Minimum number of files for which a thread/process pool is used; below
# this the pool overhead is larger than the gain.
BG_PARALLEL_MIN_FILES = 50


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
        return idx, bgt.compute_background(tt, valI, valE, method, params)
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
            tt, I, e = load_data(path, twoThetaMin=-1e9, twoThetaMax=1e9)
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

    def prefetch_raw(self, filePaths, maxWorkers=None,
                     progressCallback=None, cancelFlag=None):
        """Load every path in filePaths into raw_cache in parallel.

        Threads (not processes) are correct here because np.loadtxt /
        pandas releases the GIL on disk I/O, and we avoid the pickling
        cost a process pool would incur. Order-preserving via map().

        progressCallback(i, n) is called from the main thread after
        each completion, so it's safe to update Qt widgets from there.
        cancelFlag, if provided, is a list whose [0] element is checked
        between completions; set [0]=True to stop early."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Skip files we've already cached.
        todo = [p for p in filePaths if p not in self.raw_cache]
        n = len(todo)
        if n == 0:
            if progressCallback is not None:
                progressCallback(len(filePaths), len(filePaths))
            return

        if maxWorkers is None:
            # I/O scales past CPU count - threads are cheap, disk is the
            # bottleneck. Cap at 32 to be polite on networked drives.
            import os as _os
            maxWorkers = min(32, (_os.cpu_count() or 4) * 2)

        def _worker(p):
            # Worker runs in a thread; touches no Qt objects. Loads the
            # full pattern (no crop) so the raw cache is range-agnostic.
            try:
                tt, I, e = load_data(p, twoThetaMin=-1e9, twoThetaMax=1e9)
                return p, (np.asarray(tt), np.asarray(I), np.asarray(e))
            except Exception as exc_:
                return p, exc_

        done = 0
        with ThreadPoolExecutor(max_workers=maxWorkers) as pool:
            futures = {pool.submit(_worker, p): p for p in todo}
            for fut in as_completed(futures):
                if cancelFlag is not None and cancelFlag[0]:
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
                if progressCallback is not None:
                    # progress is over the WHOLE file_paths list so the
                    # ETA looks right even when most were already cached.
                    already = len(filePaths) - n
                    progressCallback(already + done, len(filePaths))

    # ------------------------------------------------------------------
    # Parallel background subtraction
    # ------------------------------------------------------------------
    def compute_backgrounds_parallel(self, filePaths, method, params,
                                     mode='threads', maxWorkers=None,
                                     progressCallback=None,
                                     cancelFlag=None,
                                     minParallelFiles=BG_PARALLEL_MIN_FILES):
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

        Returns a list of results in the SAME ORDER as filePaths.
        Entries are either (twoTheta, valueInt, valueBG, valueIntBGsub)
        or None for paths that failed.
        """
        n = len(filePaths)
        results = [None] * n

        # Build the cache keys up front. Anything already cached returns
        # immediately and isn't shipped to a worker.
        keys = [(p, method, tuple(sorted(params.items()))) for p in filePaths]
        todo = []   # list of (idx, path)
        for i, key in enumerate(keys):
            cached = self.bg_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                todo.append((i, filePaths[i]))

        # Progress accounts for both already-cached and newly-computed items
        already_done = n - len(todo)

        def _bump(extra):
            if progressCallback is not None:
                progressCallback(already_done + extra, n)

        _bump(0)
        if not todo:
            return results

        # Decide on actual mode. Below the threshold, force serial.
        if len(todo) < minParallelFiles and mode != 'serial':
            mode = 'serial'

        if mode == 'serial':
            for done, (i, path) in enumerate(todo, start=1):
                if cancelFlag is not None and cancelFlag[0]:
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

        if maxWorkers is None:
            import os as _os
            cpu = _os.cpu_count() or 4
            # Threads can over-subscribe; processes shouldn't.
            maxWorkers = cpu if mode == 'processes' else max(2, cpu)

        Pool = ProcessPoolExecutor if mode == 'processes' else ThreadPoolExecutor
        completed = 0

        try:
            with Pool(max_workers=maxWorkers) as pool:
                futures = {pool.submit(_bg_worker, t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    if cancelFlag is not None and cancelFlag[0]:
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
                        path = filePaths[idx]
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
                if cancelFlag is not None and cancelFlag[0]:
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

        # The raw pattern is loaded into raw_cache on first use, so the
        # background is always computed on in-memory, cropped arrays.
        cropped = self.get_cropped(path, params["twoThetaMin"],
                                   params["twoThetaMax"])
        if cropped is None:
            return None
        try:
            twoTheta, valueInt, valueBG, valueIntBGsub = \
                bgt.compute_background(*cropped, method, params)
        except Exception as e:
            print(f"  BG fit failed for {path}: {e}")
            return None

        processed = (twoTheta, valueInt, valueBG, valueIntBGsub)
        self.bg_cache[key] = processed
        return processed
