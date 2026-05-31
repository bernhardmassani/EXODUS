# -*- coding: utf-8 -*-
"""
CIF_to_JCPDS_toolbox.py
-----------------------
Convert a CIF file to a JCPDS file in the format used by the EXODUS
toolbox (BatchFit_toolbox.load_JCPDS).

Only one public function is intended for external use:

    cif_to_jcpds(cif_path, jcpds_path=None, K0=20.0, K0P=4.0, ...)

It defers the import of ``pymatgen`` until first call so that EXODUS
can still start without pymatgen installed; only the user who actually
loads a CIF will hit the ImportError.

@author: bmass
"""

import os
import math


# Standard Cu Kalpha1 wavelength used to compute the reference 2theta
# values (the JCPDS file itself stores d-spacings; the wavelength only
# matters for the internal pattern calculation).
_DEFAULT_WAVELENGTH = 1.5405981  # Angstroms

# Default EoS parameters for a freshly converted CIF (the values the user
# asked for as the standard starting point).
_DEFAULT_K0 = 20.0
_DEFAULT_K0P = 4.0


def _space_group_to_symmetry(sg_number, sg_symbol):
    """Map an IT space-group number to the upper-case crystal-system
    label that BatchFit_toolbox.load_JCPDS expects.

    Note: load_JCPDS uses the (mis)spelling 'ORTHOROMBIC' for
    orthorhombic - we keep that here so the file round-trips cleanly.
    """
    if sg_number <= 2:
        return "TRICLINIC"
    if sg_number <= 15:
        return "MONOCLINIC"
    if sg_number <= 74:
        return "ORTHOROMBIC"  # match the spelling used by load_JCPDS
    if sg_number <= 142:
        return "TETRAGONAL"
    if sg_number <= 167:
        # 143-167 trigonal: rhombohedral if R-centred, otherwise
        # described on hexagonal axes.
        return "RHOMBOHEDRAL" if str(sg_symbol).startswith("R") else "HEXAGONAL"
    if sg_number <= 194:
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


def cif_to_jcpds(cif_path,
                 jcpds_path=None,
                 K0=_DEFAULT_K0,
                 K0P=_DEFAULT_K0P,
                 wavelength=_DEFAULT_WAVELENGTH,
                 two_theta_range=(1.0, 90.0),
                 min_intensity=0.0):
    """
    Convert a CIF file to a JCPDS file readable by EXODUS.

    Parameters
    ----------
    cif_path : str
        Path to the input .cif file.
    jcpds_path : str or None, optional
        Output path for the .jcpds file. If None, the file is written
        next to the CIF with the same base name and a .jcpds extension.
    K0 : float, optional
        Bulk modulus B0 in GPa (default 20).
    K0P : float, optional
        Pressure derivative of B0 (default 4).
    wavelength : float, optional
        X-ray wavelength in A used internally to choose which
        reflections to include via the 2theta range (default Cu Kalpha1).
    two_theta_range : (float, float), optional
        Range of 2theta over which to compute reflections (degrees).
    min_intensity : float, optional
        Drop reflections weaker than this (intensities are normalised
        to a maximum of 100). Default 0 = keep everything.

    Returns
    -------
    str
        Absolute path to the written JCPDS file.
    """
    Structure, XRDCalculator = _import_pymatgen()

    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    if jcpds_path is None:
        base, _ = os.path.splitext(cif_path)
        jcpds_path = base + ".jcpds"

    # --- Parse structure ---------------------------------------------------
    structure = Structure.from_file(cif_path)
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    volume = structure.lattice.volume

    sg_symbol, sg_number = structure.get_space_group_info()
    symmetry = _space_group_to_symmetry(sg_number, sg_symbol)

    # --- Compute powder pattern -------------------------------------------
    xrd = XRDCalculator(wavelength=wavelength)
    pattern = xrd.get_pattern(structure, two_theta_range=two_theta_range)

    reflections = []
    seen_hkl = set()
    for two_th, inten, hkl_list in zip(pattern.x, pattern.y, pattern.hkls):
        if inten < min_intensity:
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
    cif_basename = os.path.basename(cif_path)
    os.makedirs(os.path.dirname(os.path.abspath(jcpds_path)) or ".", exist_ok=True)

    with open(jcpds_path, "w") as f:
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
        # DIHKL value layout expected by load_JCPDS, parts after 'DIHKL:' are
        # split on whitespace as: [d, intensity, h, k, l]
        for d, inten, h, k, l in reflections:
            f.write(f"DIHKL:     {d:.5f}\t{inten:.2f}\t{h}\t{k}\t{l}\n")

    return os.path.abspath(jcpds_path)


def write_jcpds(jcpds_path,
                unit_cell,
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
    jcpds_path : str
        Where to write the .jcpds file.
    unit_cell : sequence of 6 floats
        [a, b, c, alpha, beta, gamma] in A and degrees.
    HKL : 2D array-like, shape (N, 4)
        Each row is [h, k, l, intensity]. The intensities here are
        intensities (NOT d-spacings); d is recomputed from the unit
        cell + h k l using the supplied symmetry, so editing the
        lattice parameters is enough to update the reflection list.
    symmetry : str
        One of 'CUBIC', 'TETRAGONAL', 'HEXAGONAL', 'RHOMBOHEDRAL',
        'ORTHOROMBIC', 'MONOCLINIC', 'TRICLINIC' (matches load_JCPDS).
    K0, K0P, alphaT, DK0DT, DK0PDT : float
        EoS parameters.
    comment : str, optional
        Free-form comment line. Newlines stripped.
    """
    # Local import to keep module load light.
    import EoS_toolbox as EoS  # noqa: F401  (used implicitly via volume calc)

    a, b, c, alpha, beta, gamma = [float(x) for x in unit_cell]
    volume = _unit_cell_volume(unit_cell)

    if comment is None:
        comment = "Edited by EXODUS JCPDS editor"
    comment = str(comment).replace("\n", " ").replace("\r", " ").strip()

    os.makedirs(os.path.dirname(os.path.abspath(jcpds_path)) or ".", exist_ok=True)

    with open(jcpds_path, "w") as f:
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
            d = _d_spacing(a, b, c, alpha, beta, gamma, h, k, l, symmetry)
            if d is None or not math.isfinite(d) or d <= 0:
                continue
            f.write(f"DIHKL:     {d:.5f}\t{inten:.2f}\t{int(round(h))}\t"
                    f"{int(round(k))}\t{int(round(l))}\n")

    return os.path.abspath(jcpds_path)


def _unit_cell_volume(unit_cell):
    """Standalone volume calculation - duplicate of EoS_toolbox.unitCellVolume
    so this module can be imported without a hard dep on EoS_toolbox at
    module-load time (we still import it lazily in write_jcpds for clarity).

    Returns NaN on invalid geometry (matches EoS.unitCellVolume).
    """
    a, b, c = unit_cell[0], unit_cell[1], unit_cell[2]
    alp = math.radians(unit_cell[3])
    bet = math.radians(unit_cell[4])
    gam = math.radians(unit_cell[5])
    radicand = (
        1 - math.cos(alp) ** 2 - math.cos(bet) ** 2 - math.cos(gam) ** 2
        + 2 * math.cos(alp) * math.cos(bet) * math.cos(gam)
    )
    if radicand <= 0:
        return float('nan')
    try:
        v = a * b * c * math.sqrt(radicand)
    except (ValueError, TypeError):
        return float('nan')
    return v


def _d_spacing(a, b, c, alpha, beta, gamma, h, k, l, symmetry):
    """Compute the d-spacing for a (hkl) reflection in any of the seven
    crystal systems supported by load_JCPDS. Returns None on failure.

    The formulae match those in BatchFit_toolbox.reflection_List so that
    a CIF round-tripped through this module produces the same peak
    positions as a hand-written JCPDS file.
    """
    try:
        alp = math.radians(alpha)
        bet = math.radians(beta)
        gam = math.radians(gamma)
        if symmetry == "CUBIC":
            return a / math.sqrt(h * h + k * k + l * l)
        if symmetry == "TETRAGONAL":
            return math.sqrt(1.0 / ((h * h + k * k) / a ** 2 + l * l / c ** 2))
        if symmetry == "HEXAGONAL":
            return 1.0 / math.sqrt(
                4.0 / 3.0 * (h * h + h * k + k * k) / a ** 2 + l * l / c ** 2
            )
        if symmetry == "RHOMBOHEDRAL":
            sa = math.sin(alp)
            ca = math.cos(alp)
            num = (h * h + k * k + l * l) * sa ** 2 + 2 * (h * k + k * l + h * l) * (ca ** 2 - ca)
            den = a ** 2 * (1 - 3 * ca ** 2 + 2 * ca ** 3)
            return 1.0 / math.sqrt(num / den)
        if symmetry == "ORTHOROMBIC":
            return 1.0 / math.sqrt(h * h / a ** 2 + k * k / b ** 2 + l * l / c ** 2)
        if symmetry == "MONOCLINIC":
            sb = math.sin(bet)
            cb = math.cos(bet)
            return 1.0 / math.sqrt(
                (1.0 / sb ** 2)
                * (h * h / a ** 2 + k * k * sb ** 2 / b ** 2 + l * l / c ** 2
                   - 2 * h * l * cb / (a * c))
            )
        if symmetry == "TRICLINIC":
            # Standard metric-tensor S_ij form (Cullity & Stock, App. 3 / ITC).
            # Previous (buggy) form was missing one lattice-constant factor on
            # every term under the square root (e.g. h^2*a^2*sin^2(alpha)
            # instead of h^2*b^2*c^2*sin^2(alpha)). The off-diagonal terms
            # similarly missed the third lattice constant. The correct form
            # below matches BatchFit_toolbox.reflection_List exactly so a
            # CIF round-trip reproduces the same peak positions.
            v = _unit_cell_volume([a, b, c, alpha, beta, gamma])
            sa, ca = math.sin(alp), math.cos(alp)
            sb, cb = math.sin(bet), math.cos(bet)
            sg, cg = math.sin(gam), math.cos(gam)
            S11 = b * b * c * c * sa * sa
            S22 = a * a * c * c * sb * sb
            S33 = a * a * b * b * sg * sg
            S12 = a * b * c * c * (ca * cb - cg)
            S23 = a * a * b * c * (cb * cg - ca)
            S13 = a * b * b * c * (cg * ca - cb)
            inv_d2 = (1.0 / (v * v)) * (
                S11 * h * h + S22 * k * k + S33 * l * l
                + 2.0 * S12 * h * k + 2.0 * S23 * k * l + 2.0 * S13 * h * l
            )
            return 1.0 / math.sqrt(inv_d2)
    except (ValueError, ZeroDivisionError):
        return None
    return None
