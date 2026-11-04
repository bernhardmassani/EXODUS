# jcpds_toolbox.py for EXODUS v0.1.4
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
# Reading, writing and creating JCPDS phase files (no Qt):
#
#     load_jcpds(paths)        read one or more JCPDS files into the EXODUS
#                              'phases' dictionary
#     write_jcpds(...)         write a JCPDS file from arrays (JCPDS editor)
#     cif_to_jcpds(cif, ...)   convert a CIF into a JCPDS file (pymatgen)
#
# Three on-disk JCPDS layouts are read: the canonical 'KEY: value' format
# (fully populated or abbreviated) and the legacy four-line 'DAC style'
# format. pymatgen is imported only when a CIF is converted, so EXODUS
# starts without it.
#
# ============================================================================
#                              Imported libraries
# ============================================================================
import math
import os
import re

import numpy as np

import crystallography_toolbox as cryst


# ============================================================================
#                                 CONSTANTS
# ============================================================================
# Standard Cu Kalpha1 wavelength used to compute the reference 2theta
# values (the JCPDS file itself stores d-spacings; the wavelength only
# matters for the internal pattern calculation).
_DEFAULT_WAVELENGTH = 1.5405981  # Angstroms

# Default EoS parameters for a freshly converted CIF (the values the user
# asked for as the standard starting point).
_DEFAULT_K0 = 20.0
_DEFAULT_K0P = 4.0

# Symmetry codes of the legacy four-line ('DAC style') JCPDS format.
_JCPDS_SYMMETRY_CODES = {
    1: 'CUBIC',
    2: 'HEXAGONAL',
    3: 'TETRAGONAL',
    4: 'ORTHORHOMBIC',
    5: 'MONOCLINIC',
    6: 'TRICLINIC',
    7: 'RHOMBOHEDRAL',
}


# ============================================================================
#                                  READING
# ============================================================================
def _parse_jcpds_canonical(lines, fileName):
    """Parse the canonical ``KEY: value`` JCPDS format.

    Handles both the fully populated layout (e.g. Au_350K.jcpds) and the
    abbreviated form (e.g. Ne.jcpds) where B, C, the angles, and the
    less-common EoS parameters may be omitted.
    """
    a = b = c = None
    alpha = beta = gamma = None
    crystal_system = None
    K0 = K0P = DK0DT = DK0PDT = alphaT = 0.0
    HKL = []

    for raw in lines:
        line = raw.strip()
        if not line or ':' not in line:
            continue
        key, _, value = line.partition(':')
        key = key.strip().upper()
        value = value.strip()
        if not value:
            continue

        if key == 'A':
            a = float(value.split()[0])
        elif key == 'B':
            b = float(value.split()[0])
        elif key == 'C':
            c = float(value.split()[0])
        elif key == 'ALPHA':
            alpha = float(value.split()[0])
        elif key == 'BETA':
            beta = float(value.split()[0])
        elif key == 'GAMMA':
            gamma = float(value.split()[0])
        elif key == 'K0':
            K0 = float(value.split()[0])
        elif key == 'K0P':
            K0P = float(value.split()[0])
        elif key == 'DK0DT':
            DK0DT = float(value.split()[0])
        elif key == 'DK0PDT':
            DK0PDT = float(value.split()[0])
        elif key == 'SYMMETRY':
            # Normalise: upper-case and map the legacy 'ORTHOROMBIC'
            # spelling onto 'ORTHORHOMBIC'.
            crystal_system = cryst.normalise_crystal_system(value.split()[0])
        elif key == 'ALPHAT':
            alphaT = float(value.split()[0])
        elif key == 'DIHKL':
            parts = re.split(r'[\t ,]+', value)
            if len(parts) < 5:
                continue  # malformed line, skip
            # parts: [d, intensity, h, k, l]
            try:
                Int = float(parts[1])
                H = float(parts[2])
                K = float(parts[3])
                L = float(parts[4])
            except ValueError:
                continue
            HKL.append([H, K, L, Int])

    if a is None:
        raise ValueError(f"JCPDS file '{fileName}' is missing the 'A:' lattice parameter.")
    if crystal_system is None:
        raise ValueError(f"JCPDS file '{fileName}' is missing the 'SYMMETRY:' entry.")

    a, b, c, alpha, beta, gamma = cryst.fill_unit_cell(a, b, c, alpha, beta, gamma, crystal_system)
    return (a, b, c, alpha, beta, gamma, crystal_system,
            K0, K0P, alphaT, DK0DT, DK0PDT, HKL)


def _parse_jcpds_legacy_dac(lines, fileName):
    """Parse the legacy four-line ("DAC style") JCPDS format.

    Layout::

        Line 1: free-form comment / phase name
        Line 2: <sym_code> <a> [<b>] [<c>] [<alpha>] [<beta>] [<gamma>] <K0> <K0P> <alphaT>
        Line 3: column header (``d (A)  I/I0  h  k  l``)
        Line 4+: ``d  I/I0  h  k  l`` per reflection

    The number of lattice values on the parameter line depends on the
    symmetry code: cubic uses 1, hex/tetragonal use 2, orthorhombic 3,
    rhombohedral 2 (a, alpha), monoclinic 4 (a, b, c, beta), and
    triclinic 6 (a, b, c, alpha, beta, gamma).
    """
    # Clean lines: keep raw lines, but strip the trailing newline.
    cleaned = [ln.rstrip('\n').rstrip('\r') for ln in lines]
    # Skip leading blank lines.
    idx = 0
    while idx < len(cleaned) and not cleaned[idx].strip():
        idx += 1
    if idx >= len(cleaned):
        raise ValueError(f"JCPDS file '{fileName}' is empty.")

    # Line 2 = parameter line.
    if idx + 1 >= len(cleaned):
        raise ValueError(f"JCPDS file '{fileName}' is too short to be a legacy JCPDS.")

    param_line = cleaned[idx + 1].strip()
    tokens = re.split(r'[\t ,]+', param_line)
    if len(tokens) < 4:
        raise ValueError(
            f"JCPDS file '{fileName}' parameter line has too few values: '{param_line}'.")

    try:
        sym_code = int(float(tokens[0]))
    except ValueError as exc:
        raise ValueError(
            f"JCPDS file '{fileName}': could not read symmetry code from '{tokens[0]}'."
        ) from exc

    if sym_code not in _JCPDS_SYMMETRY_CODES:
        raise ValueError(
            f"JCPDS file '{fileName}': unknown symmetry code {sym_code}.")
    crystal_system = _JCPDS_SYMMETRY_CODES[sym_code]

    # Number of lattice tokens expected for each symmetry on the parameter line.
    n_lattice_by_sym = {
        'CUBIC': 1,         # a
        'HEXAGONAL': 2,     # a, c
        'TETRAGONAL': 2,    # a, c
        'RHOMBOHEDRAL': 2,  # a, alpha (rhombohedral setting)
        'ORTHORHOMBIC': 3,  # a, b, c
        'MONOCLINIC': 4,    # a, b, c, beta
        'TRICLINIC': 6,     # a, b, c, alpha, beta, gamma
    }
    n_lat = n_lattice_by_sym[crystal_system]
    # After: sym_code  <n_lat lattice values>  K0  K0P  alphaT
    # That gives a minimum length of 1 + n_lat + 3.
    min_len = 1 + n_lat + 3
    if len(tokens) < min_len:
        raise ValueError(
            f"JCPDS file '{fileName}': parameter line has {len(tokens)} values, "
            f"expected at least {min_len} for {crystal_system}.")

    lat_vals = [float(x) for x in tokens[1:1 + n_lat]]
    K0 = float(tokens[1 + n_lat])
    K0P = float(tokens[2 + n_lat])
    alphaT = float(tokens[3 + n_lat])
    DK0DT = 0.0   # not present in the legacy format
    DK0PDT = 0.0

    a = b = c = None
    alpha = beta = gamma = None
    if crystal_system == 'CUBIC':
        a = lat_vals[0]
    elif crystal_system in ('HEXAGONAL', 'TETRAGONAL'):
        a, c = lat_vals
    elif crystal_system == 'RHOMBOHEDRAL':
        a, alpha = lat_vals
    elif crystal_system == 'ORTHORHOMBIC':
        a, b, c = lat_vals
    elif crystal_system == 'MONOCLINIC':
        a, b, c, beta = lat_vals
    elif crystal_system == 'TRICLINIC':
        a, b, c, alpha, beta, gamma = lat_vals

    a, b, c, alpha, beta, gamma = cryst.fill_unit_cell(a, b, c, alpha, beta, gamma, crystal_system)

    # Skip line 3 (column header) and parse the remaining lines as reflections.
    HKL = []
    for raw in cleaned[idx + 3:]:
        line = raw.strip()
        if not line:
            continue
        # Skip stray comment / header lines.
        if line.lower().startswith('d') and 'i' in line.lower():
            continue
        parts = re.split(r'[\t ,]+', line)
        if len(parts) < 5:
            continue
        try:
            # parts: [d, intensity, h, k, l]
            Int = float(parts[1])
            H = float(parts[2])
            K = float(parts[3])
            L = float(parts[4])
        except ValueError:
            continue
        HKL.append([H, K, L, Int])

    return (a, b, c, alpha, beta, gamma, crystal_system,
            K0, K0P, alphaT, DK0DT, DK0PDT, HKL)


def _detect_jcpds_format(lines):
    """Return 'canonical' or 'legacy' for a list of raw file lines.

    Canonical files use ``KEY: value`` syntax and almost always contain
    a ``SYMMETRY:`` line. The legacy DAC-style format has no colon-keyed
    lines at all, and the second non-blank line is a whitespace-separated
    list of numbers starting with the symmetry code.
    """
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # Comment-only lines that happen to contain a colon (e.g. a
        # citation) shouldn't fool us — but the canonical format always
        # uses uppercase keys, so checking for any of the well-known keys
        # is robust.
        upper = line.upper()
        for key in ('VERSION:', 'COMMENT:', 'SYMMETRY:', 'K0:', 'A:', 'DIHKL:'):
            if upper.startswith(key):
                return 'canonical'
    return 'legacy'


def load_jcpds(JCPDS):
    '''
    Input is one or more JCPDS files and it stores the parameters for the
    phases for this refinement as a dictionary.  ``phase_i`` calls the
    i-th phase, e.g. ``phases['phase_0_unit_cell']``.

    Three on-disk formats are recognised:

    1.  Canonical, fully populated ``KEY: value`` layout
        (e.g. Au_350K.jcpds) - the format produced by ``write_jcpds``.
    2.  Canonical but abbreviated (e.g. Ne.jcpds), where B, C, the
        angles, and the less-common EoS parameters may be omitted and
        are filled in from the symmetry.
    3.  Legacy four-line "DAC style" format (e.g. Mo_400.jcpds), with a
        comment header line, a single parameter line whose layout
        depends on the symmetry code, a column-header line, and one
        ``d  I  h  k  l`` row per reflection.
    '''

    # Ensure we have a list of files
    if isinstance(JCPDS, str):
        JCPDS = [JCPDS]

    phases = {}
    phases_used = []
    i = -1  # so phases['phases_Number'] = i + 1 works even with empty input
    for i, phase in enumerate(JCPDS):
        file_name = phase
        file_path = str(file_name)
        print('Load: \t' + str(file_path))
        phase_i = 'phase_' + str(i)
        phases_used.append(i)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        fmt = _detect_jcpds_format(lines)
        if fmt == 'canonical':
            parsed = _parse_jcpds_canonical(lines, file_name)
        else:
            parsed = _parse_jcpds_legacy_dac(lines, file_name)

        (a, b, c, alpha, beta, gamma, crystal_system,
         K0, K0P, alphaT, DK0DT, DK0PDT, HKL) = parsed

        crystal_system = cryst.normalise_crystal_system(crystal_system)
        if crystal_system not in cryst.CRYSTAL_SYSTEMS:
            raise ValueError(
                f"Unknown SYMMETRY '{crystal_system}' in JCPDS file "
                f"{file_path}. Expected one of: {', '.join(cryst.CRYSTAL_SYSTEMS)}")

        # [V0, K0, K0P, alphaT, DK0DT, DK0PDT]
        compression_constants = np.array([0.0, K0, K0P, alphaT, DK0DT, DK0PDT])
        HKL = np.array(HKL)
        unit_cell = np.array([a, b, c, alpha, beta, gamma])
        V0 = cryst.unit_cell_volume(unit_cell.tolist())
        if V0 is None:
            raise ValueError(f"Invalid unit cell in JCPDS file {file_path}: "
                             f"{unit_cell.tolist()}")
        compression_constants[0] = V0

        phase_name_only = os.path.splitext(os.path.basename(file_name))[0]
        phases[phase_i + '_name'] = phase_name_only
        phases[phase_i + '_crystal_system'] = crystal_system
        phases[phase_i + '_unit_cell'] = unit_cell
        phases[phase_i + '_unit_cell_0'] = unit_cell
        phases[phase_i + '_HKL'] = HKL
        phases[phase_i + '_compression_constants'] = compression_constants

    phases['phases_Number'] = i + 1

    phases_used = np.array(phases_used)
    phases['phases_used'] = phases_used
    phases['phases_Number_array'] = phases_used
    return phases




# ============================================================================
#                                  WRITING
# ============================================================================
def write_jcpds(jcpdsPath,
                unitCell,
                HKL,
                symmetry,
                K0,
                K0P,
                alphaT=0.0,
                DK0DT=0.0,
                DK0PDT=0.0,
                comment=None):
    """
    Write a JCPDS file from existing arrays - used by the JCPDS editor
    dialog to save edits back to disk.

    Parameters
    ----------
    jcpdsPath : str
        Where to write the .jcpds file.
    unitCell : sequence of 6 floats
        [a, b, c, alpha, beta, gamma] in A and degrees.
    HKL : 2D array-like, shape (N, 4)
        Each row is [h, k, l, intensity]. The intensities here are
        intensities (NOT d-spacings); d is recomputed from the unit
        cell + h k l using the supplied symmetry, so editing the
        lattice parameters is enough to update the reflection list.
    symmetry : str
        One of 'CUBIC', 'TETRAGONAL', 'HEXAGONAL', 'RHOMBOHEDRAL',
        'ORTHORHOMBIC', 'MONOCLINIC', 'TRICLINIC' (matches load_jcpds).
        The legacy spelling 'ORTHOROMBIC' is accepted and written out
        with the correct spelling.
    K0, K0P, alphaT, DK0DT, DK0PDT : float
        EoS parameters.
    comment : str, optional
        Free-form comment line. Newlines stripped.
    """
    a, b, c, alpha, beta, gamma = [float(x) for x in unitCell]
    symmetry = cryst.normalise_crystal_system(symmetry)
    volume = cryst.unit_cell_volume(unitCell)
    volume = float('nan') if volume is None else volume

    if comment is None:
        comment = "Edited by EXODUS JCPDS editor"
    comment = str(comment).replace("\n", " ").replace("\r", " ").strip()

    os.makedirs(os.path.dirname(os.path.abspath(jcpdsPath)) or ".", exist_ok=True)

    with open(jcpdsPath, "w") as f:
        f.write(f"VERSION:   4\n")
        f.write(f"COMMENT:   {comment}\n")
        f.write(f"K0:        {K0:.4f}\n")
        f.write(f"K0P:       {K0P:.4f}\n")
        f.write(f"SYMMETRY:  {symmetry}\n")
        f.write(f"A:         {a:.5f}\n")
        f.write(f"B:         {b:.5f}\n")
        f.write(f"C:         {c:.5f}\n")
        f.write(f"ALPHA:     {alpha:.4f}\n")
        f.write(f"BETA:      {beta:.4f}\n")
        f.write(f"GAMMA:     {gamma:.4f}\n")
        f.write(f"VOLUME:    {volume:.4f}\n")
        f.write(f"ALPHAT:    {alphaT:.6e}\n")
        f.write(f"DK0DT:     {DK0DT:.6e}\n")
        f.write(f"DK0PDT:    {DK0PDT:.6e}\n")
        for row in HKL:
            h, k, l, inten = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            d, _tt = cryst.d_and_twotheta(unitCell, symmetry, h, k, l, None)
            if d is None:
                continue
            f.write(f"DIHKL:     {d:.5f}\t{inten:.2f}\t{int(round(h))}\t"
                    f"{int(round(k))}\t{int(round(l))}\n")

    return os.path.abspath(jcpdsPath)




# ============================================================================
#                               CIF CONVERSION
# ============================================================================
def _space_group_to_symmetry(sgNumber, sgSymbol):
    """Map an IT space-group number to the upper-case crystal-system
    label that load_jcpds expects.

    Since EXODUS v0.1.4 the canonical spelling is 'ORTHORHOMBIC'
    (older versions used the misspelling 'ORTHOROMBIC'; load_jcpds still
    accepts it on read).
    """
    if sgNumber <= 2:
        return "TRICLINIC"
    if sgNumber <= 15:
        return "MONOCLINIC"
    if sgNumber <= 74:
        return "ORTHORHOMBIC"
    if sgNumber <= 142:
        return "TETRAGONAL"
    if sgNumber <= 167:
        # 143-167 trigonal: rhombohedral if R-centred, otherwise
        # described on hexagonal axes.
        return "RHOMBOHEDRAL" if str(sgSymbol).startswith("R") else "HEXAGONAL"
    if sgNumber <= 194:
        return "HEXAGONAL"
    return "CUBIC"


def _import_pymatgen():
    """Import pymatgen lazily and raise a helpful error if missing."""
    try:
        from pymatgen.core import Structure
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ImportError as exc:
        raise ImportError(
            "Converting a CIF to a JCPDS requires pymatgen. "
            "Install it with:  pip install pymatgen"
        ) from exc
    return Structure, XRDCalculator


def cif_to_jcpds(cifPath,
                 jcpdsPath=None,
                 K0=_DEFAULT_K0,
                 K0P=_DEFAULT_K0P,
                 wavelength=_DEFAULT_WAVELENGTH,
                 twoThetaRange=(1.0, 90.0),
                 minIntensity=0.0):
    """
    Convert a CIF file to a JCPDS file readable by EXODUS.

    Parameters
    ----------
    cifPath : str
        Path to the input .cif file.
    jcpdsPath : str or None, optional
        Output path for the .jcpds file. If None, the file is written
        next to the CIF with the same base name and a .jcpds extension.
    K0 : float, optional
        Bulk modulus B0 in GPa (default 20).
    K0P : float, optional
        Pressure derivative of B0 (default 4).
    wavelength : float, optional
        X-ray wavelength in A used internally to choose which
        reflections to include via the 2theta range (default Cu Kalpha1).
    twoThetaRange : (float, float), optional
        Range of 2theta over which to compute reflections (degrees).
    minIntensity : float, optional
        Drop reflections weaker than this (intensities are normalised
        to a maximum of 100). Default 0 = keep everything.

    Returns
    -------
    str
        Absolute path to the written JCPDS file.
    """
    Structure, XRDCalculator = _import_pymatgen()

    if not os.path.isfile(cifPath):
        raise FileNotFoundError(f"CIF file not found: {cifPath}")

    if jcpdsPath is None:
        base, _ = os.path.splitext(cifPath)
        jcpdsPath = base + ".jcpds"

    # --- Parse structure ---------------------------------------------------
    structure = Structure.from_file(cifPath)
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    volume = structure.lattice.volume

    sg_symbol, sg_number = structure.get_space_group_info()
    symmetry = _space_group_to_symmetry(sg_number, sg_symbol)
    # EXODUS uses RHOMBOHEDRAL for the rhombohedral setting
    # (a = b = c, alpha = beta = gamma). pymatgen normally returns
    # R-centred structures on hexagonal axes; those are written as
    # HEXAGONAL so the cell and the (hexagonal) hkl stay consistent.
    if symmetry == "RHOMBOHEDRAL" and abs(gamma - 120.0) < 1e-3:
        symmetry = "HEXAGONAL"

    # --- Compute powder pattern -------------------------------------------
    xrd = XRDCalculator(wavelength=wavelength)
    pattern = xrd.get_pattern(structure, two_theta_range=twoThetaRange)

    reflections = []
    seen_hkl = set()
    for two_th, inten, hkl_list in zip(pattern.x, pattern.y, pattern.hkls):
        if inten < minIntensity:
            continue
        # First entry of the equivalent set is enough for a JCPDS row.
        h, k, l = hkl_list[0]['hkl']
        key = (int(h), int(k), int(l))
        if key in seen_hkl:
            continue
        seen_hkl.add(key)
        d = wavelength / (2.0 * math.sin(math.radians(two_th / 2.0)))
        reflections.append((d, float(inten), int(h), int(k), int(l)))

    # Sort by descending d (ascending 2theta) - standard JCPDS layout.
    reflections.sort(key=lambda r: -r[0])

    # --- Write the JCPDS file ---------------------------------------------
    cif_basename = os.path.basename(cifPath)
    os.makedirs(os.path.dirname(os.path.abspath(jcpdsPath)) or ".", exist_ok=True)

    with open(jcpdsPath, "w") as f:
        f.write(f"VERSION:   4\n")
        f.write(f"COMMENT:   Generated by EXODUS from {cif_basename} "
                f"(SG #{sg_number} {sg_symbol})\n")
        f.write(f"K0:        {K0:.4f}\n")
        f.write(f"K0P:       {K0P:.4f}\n")
        f.write(f"SYMMETRY:  {symmetry}\n")
        f.write(f"A:         {a:.5f}\n")
        f.write(f"B:         {b:.5f}\n")
        f.write(f"C:         {c:.5f}\n")
        f.write(f"ALPHA:     {alpha:.4f}\n")
        f.write(f"BETA:      {beta:.4f}\n")
        f.write(f"GAMMA:     {gamma:.4f}\n")
        f.write(f"VOLUME:    {volume:.4f}\n")
        f.write(f"ALPHAT:    0.0\n")
        f.write(f"DK0DT:     0.0\n")
        f.write(f"DK0PDT:    0.0\n")
        # DIHKL value layout expected by load_jcpds, parts after 'DIHKL:' are
        # split on whitespace as: [d, intensity, h, k, l]
        for d, inten, h, k, l in reflections:
            f.write(f"DIHKL:     {d:.5f}\t{inten:.2f}\t{h}\t{k}\t{l}\n")

    return os.path.abspath(jcpdsPath)
