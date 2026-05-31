"""
Created on Sat Feb  1 23:32:35 2025
@ Dr Bernhard Massani 
-*- coding: utf-8 -*-
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import os
import glob
import bisect
import EoS_toolbox as EoS
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
from lmfit import Parameters, minimize, fit_report
from lmfit import Minimizer, Parameters, conf_interval
from lmfit.confidence import conf_interval
from lmfit.printfuncs import report_ci
import re # for splitting JCPDS/HKL
from matplotlib.colors import LogNorm

###################
#   OS Functions  #   
###################

# LEGACY: Not called from GUI - kept for standalone script use
# LEGACY: create_folder commented out - not used by GUI
# def create_folder(folder_path):
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#         print(f"\n Analysis Folder '{folder_path}' created successfully.")
#     else:
#         #print(f"\n Analysis Folder '{folder_path}' already exists.")
#         pass
#     return str(folder_path) 

################
#   Load Poni  #   
################

def load_poni(poni_file):
    file_name = poni_file
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


#################
#   Load JCPDS  #   
#################

# Mapping of numeric symmetry codes used by the legacy four-line JCPDS
# format ("DAC style") to the canonical symmetry strings used throughout
# the rest of the code base. The codes follow the convention used by the
# old DAC/PolyXtal JCPDS files.
_JCPDS_SYMMETRY_CODES = {
    1: 'CUBIC',
    2: 'HEXAGONAL',
    3: 'TETRAGONAL',
    4: 'ORTHOROMBIC',   # spelling kept consistent with the rest of the code base
    5: 'MONOCLINIC',
    6: 'TRICLINIC',
    7: 'RHOMBOHEDRAL',
}


def _default_angles(symmetry):
    """Return default (alpha, beta, gamma) for a given crystal system.

    Used when the JCPDS file omits the angle entries (e.g. the abbreviated
    canonical format for cubic phases such as Ne.jcpds).
    """
    sym = (symmetry or '').upper()
    if sym == 'HEXAGONAL':
        return 90.0, 90.0, 120.0
    # CUBIC, TETRAGONAL, ORTHOROMBIC, RHOMBOHEDRAL (when treated as hex
    # setting), and the safe fallback all use 90/90/90. Truly triclinic
    # or monoclinic files are expected to specify their angles explicitly.
    return 90.0, 90.0, 90.0


def _fill_unit_cell(a, b, c, alpha, beta, gamma, symmetry):
    """Fill in missing lattice parameters using symmetry constraints.

    Abbreviated canonical JCPDS files often only supply A (cubic) or A
    and C (hex/tetragonal). This expands them to a full six-parameter
    cell so the downstream code sees a consistent shape.
    """
    sym = (symmetry or '').upper()
    if sym == 'CUBIC':
        if b is None:
            b = a
        if c is None:
            c = a
    elif sym in ('HEXAGONAL', 'TETRAGONAL'):
        if b is None:
            b = a
        # c must be supplied for these systems
    elif sym == 'RHOMBOHEDRAL':
        # Rhombohedral in the rhombohedral setting: a = b = c, alpha = beta = gamma
        if b is None:
            b = a
        if c is None:
            c = a
        if beta is None and alpha is not None:
            beta = alpha
        if gamma is None and alpha is not None:
            gamma = alpha
    # ORTHOROMBIC, MONOCLINIC, TRICLINIC: caller is responsible for
    # providing all three a/b/c; we only fall back on angles below.

    if alpha is None or beta is None or gamma is None:
        da, db, dg = _default_angles(sym)
        if alpha is None:
            alpha = da
        if beta is None:
            beta = db
        if gamma is None:
            gamma = dg
    return a, b, c, alpha, beta, gamma


def _parse_jcpds_canonical(lines, file_name):
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
            crystal_system = value.split()[0].upper()
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
        raise ValueError(f"JCPDS file '{file_name}' is missing the 'A:' lattice parameter.")
    if crystal_system is None:
        raise ValueError(f"JCPDS file '{file_name}' is missing the 'SYMMETRY:' entry.")

    a, b, c, alpha, beta, gamma = _fill_unit_cell(a, b, c, alpha, beta, gamma, crystal_system)
    return (a, b, c, alpha, beta, gamma, crystal_system,
            K0, K0P, alphaT, DK0DT, DK0PDT, HKL)


def _parse_jcpds_legacy_DAC(lines, file_name):
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
        raise ValueError(f"JCPDS file '{file_name}' is empty.")

    # Line 2 = parameter line.
    if idx + 1 >= len(cleaned):
        raise ValueError(f"JCPDS file '{file_name}' is too short to be a legacy JCPDS.")

    param_line = cleaned[idx + 1].strip()
    tokens = re.split(r'[\t ,]+', param_line)
    if len(tokens) < 4:
        raise ValueError(
            f"JCPDS file '{file_name}' parameter line has too few values: '{param_line}'.")

    try:
        sym_code = int(float(tokens[0]))
    except ValueError as exc:
        raise ValueError(
            f"JCPDS file '{file_name}': could not read symmetry code from '{tokens[0]}'."
        ) from exc

    if sym_code not in _JCPDS_SYMMETRY_CODES:
        raise ValueError(
            f"JCPDS file '{file_name}': unknown symmetry code {sym_code}.")
    crystal_system = _JCPDS_SYMMETRY_CODES[sym_code]

    # Number of lattice tokens expected for each symmetry on the parameter line.
    n_lattice_by_sym = {
        'CUBIC': 1,         # a
        'HEXAGONAL': 2,     # a, c
        'TETRAGONAL': 2,    # a, c
        'RHOMBOHEDRAL': 2,  # a, alpha (rhombohedral setting)
        'ORTHOROMBIC': 3,   # a, b, c
        'MONOCLINIC': 4,    # a, b, c, beta
        'TRICLINIC': 6,     # a, b, c, alpha, beta, gamma
    }
    n_lat = n_lattice_by_sym[crystal_system]
    # After: sym_code  <n_lat lattice values>  K0  K0P  alphaT
    # That gives a minimum length of 1 + n_lat + 3.
    min_len = 1 + n_lat + 3
    if len(tokens) < min_len:
        raise ValueError(
            f"JCPDS file '{file_name}': parameter line has {len(tokens)} values, "
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
    elif crystal_system == 'ORTHOROMBIC':
        a, b, c = lat_vals
    elif crystal_system == 'MONOCLINIC':
        a, b, c, beta = lat_vals
    elif crystal_system == 'TRICLINIC':
        a, b, c, alpha, beta, gamma = lat_vals

    a, b, c, alpha, beta, gamma = _fill_unit_cell(a, b, c, alpha, beta, gamma, crystal_system)

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


def load_JCPDS(JCPDS):
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
            parsed = _parse_jcpds_legacy_DAC(lines, file_name)

        (a, b, c, alpha, beta, gamma, crystal_system,
         K0, K0P, alphaT, DK0DT, DK0PDT, HKL) = parsed

        # [V0, K0, K0P, alphaT, DK0DT, DK0PDT]
        compression_constants = np.array([0.0, K0, K0P, alphaT, DK0DT, DK0PDT])
        HKL = np.array(HKL)
        unit_cell = np.array([a, b, c, alpha, beta, gamma])
        V0 = EoS.unitCellVolume(unit_cell.tolist())
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




# LEGACY: update_phases_used commented out - not used by GUI
# def update_phases_used(phases_used, phases):
#     '''
#     Replaces the phases used in the phaes dictionary
#     '''
#     phases['phases_used'] = np.array(phases_used)


def specify_phases(phases, phase_rules, frame):
    '''
    Used to specify which phases are loaded for a given frame.
    '''
    for condition, phase_list in phase_rules:
        if condition(frame):
            phases['phases_used'] = phase_list
            break
    else:
        raise ValueError(f"Unexpected frame number: {frame}")
        
    
    # if frame <= 48:
    #     phases['phases_used'] = [0]
    # elif frame == 49:
    #     phases['phases_used'] = [0,2]
    # elif frame == 50:    
    #     phases['phases_used'] = [0,1,2]
    # elif frame > 50:
    #     phases['phases_used'] = [0,1]
    # else:
    #     raise ValueError("Unexpected frame number: " + str(frame))

    return phases


def update_unitCell(newUnitCell, phases, phaseNumber):
    '''
    Takes a new unit cell and replaces the values in the phaes dictionary 
    '''
    newUnitCell = np.array(newUnitCell)     # makes sure it is a np.array
    
    key = 'phase_' + str(phaseNumber) + '_unit_cell'
    if key in phases:
        phases[key] = newUnitCell
    else:
        print('Error. No matching unit cell to update.')
    # try: 
    #     phases['phase_'+ str(phaseNumber)+'_unit_cell'] = newUnitCell
    # except:
    #     print('Error. No matching unit cell to update.')


# LEGACY: Not called from GUI or EXODUS_core
# LEGACY: update_HKL commented out - not used by GUI
# def update_HKL(newHKL, phases, phaseNumber):
#     '''
#     Takes a new HKL-array and replaces the values in the phases dictionary 
#     HKL array has the form [[H, K, L, Int], [...], ...]    
#     '''
#     newHKL = np.array(newHKL)       # makes sure it is a np.array

#     #HKL_item = [float(H),float(K),float(L),float(Int)]
#     #HKL.append(HKL_item)    
#     key = 'phase_' + str(phaseNumber) + '_HKL'
#     if key in phases:
#         phases[key] = newHKL
#     else:
#         print('Error. No matching unit cell to update.')


################
#   Load Data  #   
################

def extract_number(filename):
    """Extracts the numeric part from filenames like 'run1_93' safely.

    Returns float('inf') for any filename that doesn't end in an integer
    suffix (e.g. 'metadata.txt', '.xy', or empty); these get sorted to
    the end and then filtered out by load_frame.
    """
    base = os.path.basename(filename)
    parts = base.split("_")

    try:
        return int(parts[-1].split(".")[0])  # Extract number before any extension
    except (ValueError, IndexError, AttributeError):
        return float('inf')  # If no valid number, send to the end

# LEGACY: Not called from GUI - use len(file_paths) instead
# LEGACY: load_frameNumber commented out - not used by GUI
# def load_frameNumber(data_path):
#     # Get all files in the folder and filter out non-numeric ones
#     files = sorted(glob.glob(os.path.join(data_path, "*")), key=extract_number)
#     return len(files)

def load_frame(frame, data_path, twoThetaMin = 0, twoThetaMax = 30):
    # Get all files in the folder and filter out non-numeric ones
    files = sorted(glob.glob(os.path.join(data_path, "*")), key=extract_number)
    if files:
        print(f"Found {len(files)} files in the directory.")# + str(data_path))
        #num_files = len(glob.glob(data_path + "/*"))  # Counts all files
        #print(f"Number of files: {num_files}")
    else:
        print("No data files found in the directory!")
            
    # Remove files where extract_number() returned 'inf' (non-numeric suffix)
    files = [f for f in files if extract_number(f) != float('inf')]

    total_files = len(files)

    if total_files == 0:
        print("No valid numbered files found!")
        return None

    if 1 <= frame <= total_files:
        file_path = files[frame - 1]  # Convert to 0-based index
        #print(f"Loading file {frame}/{total_files}: {file_path}")
        # Open and read file content
        twoTheta, valueInt, errorInt = loadData(file_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)
        #print("File loaded successfully!")

        return twoTheta, valueInt, errorInt
    else:
        print(f"Error: Frame {frame} is out of range. Total files: {total_files}")
        return None

def load_frame_fast(frame, data_path, twoThetaMin = 0, twoThetaMax = 30):
    # Get all files in the folder and filter out non-numeric ones
    files = sorted(glob.glob(os.path.join(data_path, "*")), key=extract_number)
            
    # Remove files where extract_number() returned 'inf' (non-numeric suffix)
    files = [f for f in files if extract_number(f) != float('inf')]
    total_files = len(files)

    if total_files == 0:
        print("No valid numbered files found!")
        return None

    if 1 <= frame <= total_files:
        file_path = files[frame - 1]  # Convert to 0-based index
        # Open and read file content
        twoTheta, valueInt, errorInt = loadData(file_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)
        return twoTheta, valueInt, errorInt
    else:
        print(f"Error: Frame {frame} is out of range. Total files: {total_files}")
        return None


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


def _load_fxye(file_path):
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
    df = pd.read_csv(file_path, header=23, delimiter='\t')
    n_cols = df.shape[1]
    if n_cols >= 4:
        # Leading-tab layout: empty col 0, then 2theta/I/sigma at 1/2/3.
        col_tt, col_I, col_e = 1, 2, 3
    elif n_cols == 3:
        # Plain 3-column layout: 2theta/I/sigma at 0/1/2.
        col_tt, col_I, col_e = 0, 1, 2
    else:
        raise ValueError(
            f"fxye file {file_path!r} has {n_cols} columns after the "
            f"header; expected 3 (twoTheta, intensity, sigma) or 4 "
            f"(with a leading empty column)."
        )
    twoTheta = [i * 0.01 for i in df.iloc[:, col_tt].values.tolist()]
    valueInt = df.iloc[:, col_I].values.tolist()
    errorInt = df.iloc[:, col_e].values.tolist()
    return twoTheta, valueInt, errorInt


def _load_xy(file_path):
    '''
    DIOPTAS / pyFAI *.xy format. Whitespace-delimited 2-column file
    (twoTheta_deg, intensity) preceded by an arbitrary number of comment
    lines starting with '#'. Errors are not stored - we synthesise
    sqrt(|I|) as a sensible default (Poisson counting statistics).
    '''
    data = np.loadtxt(file_path, comments='#')
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


def _load_dat(file_path):
    '''
    Plain *.dat format. Whitespace-delimited 2-column file
    (twoTheta_deg, intensity), normally with no header. Some producers
    add '#' comment lines, so np.loadtxt with comments='#' covers both.
    Errors are synthesised as sqrt(|I|).
    '''
    data = np.loadtxt(file_path, comments='#')
    twoTheta = data[:, 0]
    valueInt = data[:, 1]
    errorInt = np.sqrt(np.abs(valueInt))
    return twoTheta.tolist(), valueInt.tolist(), errorInt.tolist()


def _load_chi(file_path):
    '''
    *.chi format (Fit2D / pyFAI). Fixed 4-line header:
        line 1 = source file path
        line 2 = x-axis label (e.g. '2th_deg')
        line 3 = blank
        line 4 = number of points
    followed by whitespace-delimited (twoTheta, intensity) columns.
    Errors are synthesised as sqrt(|I|).
    '''
    data = np.loadtxt(file_path, skiprows=4)
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


def loadData(file_path, twoThetaMin=0, twoThetaMax=30):
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
    ext = os.path.splitext(file_path)[1].lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file extension '{ext}' for {file_path}. "
            f"Supported: {', '.join(sorted(_LOADERS.keys()))}"
        )
    twoTheta, valueInt, errorInt = loader(file_path)
    return _truncate_range(twoTheta, valueInt, errorInt,
                           twoThetaMin, twoThetaMax)


#################
# Conversations #   
#################

rad_to_deg = 180 / np.pi
deg_to_rad = np.pi / 180
colours = [
    'red', 'blue', 'orange', 'purple', 'yellow', 'green', 
    'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender',
    'brown', 'beige', 'maroon', 'navy', 'olive', 'coral',
    'turquoise', 'gold', 'silver', 'indigo', 'violet', 'crimson'
]

# LEGACY: Formula noted as wrong in docstring - do not use
# LEGACY: twoTheta_to_q commented out - not used by GUI
# def twoTheta_to_q(twoTheta, WL):
#     '''
#     Input twoTheta (deg); output Q; Wrong Formula - fix before use
#     '''
#     return  4 * math.pi / WL *math.sin(twoTheta/2)   

def dspacing_to_twoTheta(d, WL):
    '''
    Input dspacing; output twoTheta (deg)
    '''
    return 2 * np.arcsin(WL / (2 * d)) * rad_to_deg

def twoTheta_to_dspacing(twoTheta, WL):
    '''
    Input twoTheta (deg); output dspacing.

    Returns +inf at 2theta=0 (the mathematically correct limit d -> inf as
    sin(theta) -> 0). The np.errstate suppresses the cosmetic
    'divide by zero' RuntimeWarning that np.sin(0) triggers; the inf is
    intentional and is used by initialise_parameters_LB as the d_max
    sentinel for an "all-pass" 2theta filter.
    '''
    with np.errstate(divide='ignore'):
        return 1 / (2 * np.sin(twoTheta / rad_to_deg / 2) / WL)

##################
# Background Fit #   
##################

def backgroundFit_arrays(twoTheta, valueInt, errorInt,
                         peakSearchAuto=True, prominence=0.10, height=0.08,
                         order=6, excludePeakList=[], peakwidth=0.35):
    """Array-in / array-out variant of backgroundFit. Identical logic,
    but operates on pre-loaded arrays (e.g. from DataManager.get_cropped)
    instead of a file path. backgroundFit() is now a thin wrapper."""
    twoTheta = np.array(twoTheta)
    valueInt = np.array(valueInt)
    errorInt = np.array(errorInt)

    twoTheta_unaltered = np.array(twoTheta)
    valueInt_unaltered = np.array(valueInt)
    errorInt_unaltered = np.array(errorInt)

    peaks, _ = find_peaks(valueInt, height=max(valueInt) * height,
                          threshold=None, distance=None, prominence=prominence,
                          width=None, wlen=None, rel_height=0.5, plateau_size=None)

    if excludePeakList:
        peaks = excludePeakList

    peak_list = []
    twoTheta_removed = twoTheta
    valueInt_removed = valueInt
    for i, peak in enumerate(twoTheta[peaks]):
        peak_list.append(peak)
        peakmin = bisect.bisect(twoTheta, peak - peakwidth)
        peakmax = bisect.bisect(twoTheta, peak + peakwidth)
        print(f'Peak {i} ({peak} deg) removed in the range: '
              f'{round(peak - peakwidth, 1)} - {round(peak + peakwidth, 1)}')
        twoTheta[peakmin:peakmax] = np.nan
        valueInt[peakmin:peakmax] = np.nan
        twoTheta_removed = twoTheta[np.isfinite(twoTheta)]
        valueInt_removed = valueInt[np.isfinite(valueInt)]

    con_fit = np.polyfit(twoTheta_removed, valueInt_removed, order)
    polyFit = np.poly1d(con_fit)

    return (twoTheta_unaltered, valueInt_unaltered, errorInt_unaltered,
            polyFit(twoTheta_unaltered), peak_list)


def backgroundFit(data_path, twoThetaMin = 0, twoThetaMax = 30, 
                  peakSearchAuto = True, prominence=0.10, height = 0.08, 
                  order=6, excludePeakList=[], peakwidth = 0.35,
                  frame=None, plotBG=False):
    '''
    Finds peaks based on prominence=0.1 and height = 0.3 - these values can be varied
    Fits a polynomial of i-th order (standardwise 6th order) to the background
        Returns original data and the respective BG-polynomial. When called a BG-subtraced array must be created
    '''
    twoTheta, valueInt, errorInt = loadData(
        data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
    return backgroundFit_arrays(
        twoTheta, valueInt, errorInt,
        peakSearchAuto=peakSearchAuto, prominence=prominence, height=height,
        order=order, excludePeakList=excludePeakList, peakwidth=peakwidth)



def backgroundFit_ALS_arrays(twoTheta, valueInt, errorInt,
                             lam=1e5, p=0.01, niter=10):
    """Array-in / array-out variant of backgroundFit_ALS. Identical maths,
    but no disk I/O - call this when you already have the cropped pattern
    in memory (e.g. via DataManager.get_cropped). The original disk-based
    backgroundFit_ALS is now a thin wrapper around this."""
    twoTheta = np.asarray(twoTheta)
    valueInt = np.asarray(valueInt)
    errorInt = np.asarray(errorInt)

    L = len(valueInt)
    if L < 3:
        # Not enough points for the second-derivative matrix
        return twoTheta, valueInt, errorInt, np.zeros_like(valueInt), valueInt.copy()

    # Second derivative matrix.
    # - dtype=float silences the int->float FutureWarning in scipy.sparse.diags.
    # - .tocsc() up-front so D.T @ D, the additions below, and the spsolve
    #   call all happen in CSC format and we don't get a SparseEfficiencyWarning.
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L), dtype=float).tocsc()
    D = lam * (D.T @ D)

    # Initialize weights
    w = np.ones(L)
    for _ in range(int(niter)):
        # W is built as CSC each iteration (cheap; just a diagonal),
        # so Z = W + D stays CSC and spsolve doesn't warn.
        W = sp.diags(w, 0, format='csc')
        Z = (W + D).tocsc()
        z = splinalg.spsolve(Z, w * valueInt)
        # Update weights (asymmetric)
        w = p * (valueInt > z) + (1 - p) * (valueInt < z)

    BGInt = z
    valueIntBGsub = valueInt - BGInt
    return twoTheta, valueInt, errorInt, BGInt, valueIntBGsub


def backgroundFit_ALS(data_path, twoThetaMin=0, twoThetaMax=30,
                      lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares background subtraction
    lam : float,Smoothness parameter (10^4 – 10^7 typical)
    p : float, Asymmetry parameter (0.001 – 0.1 typical)
    niter : int, Iterations (10–20 typical)
    """

    # Load data
    twoTheta, valueInt, errorInt = loadData(data_path,
                                            twoThetaMin=twoThetaMin,
                                            twoThetaMax=twoThetaMax)
    return backgroundFit_ALS_arrays(twoTheta, valueInt, errorInt,
                                    lam=lam, p=p, niter=niter)







def subtract_BG(twoTheta, valueInt, valueBG):    
    valueIntBGsub = valueInt - valueBG
    twoTheta = np.array(twoTheta)
    valueIntBGsub = np.array(valueIntBGsub)
    
    return twoTheta, valueInt, valueBG, valueIntBGsub


######################
###  Pressure Help ###
######################

def pressure_interpolation(framePressureGuess, plotPressureGuess = False):
    '''
    Takes Pressure points as an array [[frame, Pressure],[],...] and intrapolates the pressures in between
    '''
    # Extract x and y values from framePressure
    frame = np.array([point[0] for point in framePressureGuess])
    P = np.array([point[1] for point in framePressureGuess])
    
    
    framePressureGuess = np.array(framePressureGuess)
    # Create an interpolation function
    interp_func = interp1d(frame, P, kind='linear', fill_value="extrapolate")
    
    # Generate new x values for interpolation
    frame_interp = np.linspace(int(framePressureGuess[0, 0]), int(framePressureGuess[-1, 0]), int(framePressureGuess[-1, 0])) #Start, stop, samples
    frame_interp = np.array(frame_interp)
     
    # Perform interpolation
    P_interp = interp_func(frame_interp)
    P_interp = np.array(P_interp)   
    
    # Plot the original data and interpolated values
    if plotPressureGuess == True:
        plt.plot(frame, P, 'o', label='Original Data')
        plt.plot(frame_interp, P_interp, '-', label='Interpolated Data')
        plt.legend()
        plt.show()
    
    return frame_interp, P_interp


# LEGACY: Superseded by EoS.find_UC_at_P / update_phases()
# LEGACY: unit_cell_P_guess commented out - not used by GUI
# def unit_cell_P_guess(unitCell_0, frame, framePressureGuess, phases, phase):
#     '''
#     Guesses unit cell volume based on Pressure input
#     It accepts one unit cell [a, b, c, alp, bet, gam], and returns one unit cell.
#     '''
#     frame_interp, P_interp = pressure_interpolation(framePressureGuess, plotPressureGuess=False)
#     for frame_i, frame_interpolated in enumerate(frame_interp):
#         if frame_i == frame:  
#             unitCell_guess = EoS.find_UC_at_P(unitCell_0, 
#                                             float(P_interp[frame_i]), 
#                                             V0 = float(phases['phase_'+str(phase)+'_compression_constants'][0]), 
#                                             K0 = float(phases['phase_'+str(phase)+'_compression_constants'][1]), 
#                                             K0P = float(phases['phase_'+str(phase)+'_compression_constants'][2])    )
#             print('Unit Cell ' + str(phase) + ' guess: \n' + str(unitCell_guess)) 
#     return unitCell_guess

def update_phases(frame, phases, phase, framePressureGuess, 
                  deltaP, resultsFit,
                  resultsFitUC = [0,0,0,0,0,0], 
                  method = 'pressureHelper'):
    
    if method == 'sequential':
        update_unitCell(resultsFitUC, phases, phase)
        print('Unit Cell ' + str(phase) + ' guess: \n' + str(resultsFitUC)) 
        return resultsFitUC
            
    elif method == 'pressureHelper':
        # Get fitted unit cell and calculate volume
        unitCell_fit = phases['phase_'+str(phase)+'_unit_cell']
        
        # Get EoS parameters
        unitCell_0 = phases[f'phase_{phase}_unit_cell_0']
        V0 = float(phases[f'phase_{phase}_compression_constants'][0])
        K0 = float(phases[f'phase_{phase}_compression_constants'][1])
        K0P = float(phases[f'phase_{phase}_compression_constants'][2])
        
        P_fit = EoS.BM3_EOS(EoS.unitCellVolume(phases['phase_'+str(phase)+'_unit_cell']), 
                                               V0, K0, K0P)
        
        # Creates an interpolation for the pressures and therefore the unitcell volume
        frame_interp, P_interp = pressure_interpolation(framePressureGuess, plotPressureGuess=False)
        
        # Scales UC based on last UC - more reliable than doing it from UC0
        unitCell_guess = EoS.scale_UC_at_P(unitCell_fit, P_interp[frame], 
                                           unitCell_0, K0, K0P)
        # Scales UC based on last UC0
        # unitCell_new = unit_cell_P_guess(phases['phase_'+str(phase)+'_unit_cell'], 
        #                                          frame, framePressureGuess, phases, phase)
        
        print('Unit Cell ' + str(phase) + ' guess: \n' + str(unitCell_guess)) 
                
        update_unitCell(unitCell_guess, phases, phase)
        return unitCell_guess 
   
    elif method == 'pressureWalk':
        # Get fitted unit cell and calculate volume
        unitCell_fit = resultsFit[f'phase_{phase}_unit_cell_fit']

        # Get EoS parameters
        unitCell_0 = phases[f'phase_{phase}_unit_cell_0']
        V0 = float(phases[f'phase_{phase}_compression_constants'][0])
        K0 = float(phases[f'phase_{phase}_compression_constants'][1])
        K0P = float(phases[f'phase_{phase}_compression_constants'][2])

        # Calculate current pressure using BM3
        P_fit = EoS.BM3_EOS(EoS.unitCellVolume(resultsFit['phase_'+ str(phase) +'_unit_cell_fit']), 
                                               V0, K0, K0P)
        #print(f"  Phase {phase}: P_fit = {P_fit:.2f} GPa")

        #UC_fits[phase].append(UC_fit)

        # Predict next frame's Pressure
        P_next = P_fit + deltaP
        #print('Pressure of the fit is' + str(P_fit) + ' and pressure of the next frame is ' + str(P_next))
        
        # Predict next frame's unit cell
        unitCell_next = EoS.scale_UC_at_P(unitCell_fit, P_next, 
                                          unitCell_0, K0, K0P)
        # unitCell_next = EoS.find_UC_at_P(unitCell_0, 
        #                        P_next, V0, K0, K0P)

        #print('Nex UC prediction for phase '+ str(phase) +' from pressure walk:')
        #print(unitCell_next)
        
        # Update unit cell for next frame
        phases[f'phase_{phase}_unit_cell'] = unitCell_next
        
        print('Unit Cell ' + str(phase) + ' guess: \n' + str(unitCell_next)) 
        return unitCell_next

    # Loades the phases dictionary; irrelevant now since phaes are loaded in function itself
    # phases = load_JCPDS(JCPDS)
    # Updates the phases used for this refinement
    # update_unitCell(newUnitCell, phases, phaseNumber)  
    # update_phases_used(phases_used, phases) 
    # update_HKL(newHKL, phases, phaseNumber)  



###########################
###   Le Bail Fitting   ###    
###########################

# --- Internal cache of per-phase HKL stencils for the LB hot loop -----------
# Speeds up LB_fit_Model by avoiding the per-call int->float cast and
# crystal-system dispatch inside reflection_List. Keyed by id(HKL_array)
# AND by crystal_system to be safe; entries are evicted via
# _LB_clear_geom_cache() (called automatically when LB_fit_Model is asked
# to do so via the optional `_clear_geom_cache` flag).
_LB_geom_cache = {}


def _LB_phase_stencil(HKL, crystal_system):
    """Return float64 H, K, L columns plus a fast-path discriminator."""
    key = (id(HKL), crystal_system)
    cached = _LB_geom_cache.get(key)
    if cached is not None:
        return cached
    HKL_arr = np.asarray(HKL)
    H = HKL_arr[:, 0].astype(np.float64, copy=False)
    K = HKL_arr[:, 1].astype(np.float64, copy=False)
    L = HKL_arr[:, 2].astype(np.float64, copy=False)
    if crystal_system == 'CUBIC':
        # d = a / sqrt(H^2+K^2+L^2)
        denom = H*H + K*K + L*L
        cached = ('CUBIC', H, K, L, denom)
    elif crystal_system in ('TETRAGONAL', 'HEXAGONAL'):
        if crystal_system == 'HEXAGONAL':
            ab_term = (4.0/3.0) * (H*H + H*K + K*K)
        else:
            ab_term = H*H + K*K
        c_term = L*L
        cached = (crystal_system, H, K, L, ab_term, c_term)
    else:
        # Other systems use the slower path through reflection_List.
        cached = (crystal_system, H, K, L)
    _LB_geom_cache[key] = cached
    return cached


def _LB_clear_geom_cache():
    """Empty the per-phase HKL stencil cache. Call when phases dict
    changes (e.g. new JCPDS loaded) so stale stencils aren't reused."""
    _LB_geom_cache.clear()


def reflection_List(unit_cell, HKL, WL, twoThetaMin = 5, twoThetaMax = 30, crystal_system = 'CUBIC'):
    '''
    Creates a reflection list (in d and in twoTheta) based on the input phase.
    It only creates ONE list for ONE phase.

    NumPy-vectorised over reflections. Returns the same two ndarrays as the
    legacy implementation: dk_reflection_list and twoTheta_reflection_list,
    both of shape (n_visible, 4) with columns [value, H, K, L].
    '''
    a, b, c = unit_cell[0], unit_cell[1], unit_cell[2]
    alp = math.radians(unit_cell[3])
    bet = math.radians(unit_cell[4])
    gam = math.radians(unit_cell[5])

    # Cast HKL once to float64 so all downstream maths is vectorised.
    HKL_arr = np.asarray(HKL)
    H = HKL_arr[:, 0].astype(np.float64, copy=False)
    K = HKL_arr[:, 1].astype(np.float64, copy=False)
    L = HKL_arr[:, 2].astype(np.float64, copy=False)

    if crystal_system == 'CUBIC':
        dk = a / np.sqrt(H*H + K*K + L*L)
    elif crystal_system == 'TETRAGONAL':
        dk = 1.0 / np.sqrt((H*H + K*K) / (a*a) + L*L / (c*c))
    elif crystal_system == 'HEXAGONAL':
        dk = 1.0 / np.sqrt((4.0/3.0) * (H*H + H*K + K*K) / (a*a)
                           + L*L / (c*c))
    elif crystal_system == 'RHOMBOHEDRAL':
        # *** BUG FIX (audit pass) ***
        # The previous form had ' + 2.0 * (H*K + K*L + H*L) * cos_a*cos_a - cos_a'
        # i.e. the '- cos_a' was OUTSIDE the (HK+KL+HL) bracket. The correct
        # expression factors (cos^2(alpha) - cos(alpha)) inside the cross
        # term. Reference: Cullity & Stock, App. 3.
        cos_a = math.cos(alp)
        sin_a = math.sin(alp)
        num = ((H*H + K*K + L*L) * sin_a*sin_a
               + 2.0 * (H*K + K*L + H*L) * (cos_a*cos_a - cos_a))
        den = a*a * (1.0 - 3.0*cos_a*cos_a + 2.0*cos_a*cos_a*cos_a)
        dk = 1.0 / np.sqrt(num / den)
    elif crystal_system == 'ORTHOROMBIC':
        dk = 1.0 / np.sqrt(H*H / (a*a) + K*K / (b*b) + L*L / (c*c))
    elif crystal_system == 'MONOCLINIC':
        sin_b = math.sin(bet)
        cos_b = math.cos(bet)
        dk = 1.0 / np.sqrt(
            (1.0 / (sin_b*sin_b)) *
            (H*H / (a*a) + K*K * sin_b*sin_b / (b*b) + L*L / (c*c)
             - 2.0 * H * L * cos_b / (a * c))
        )
    elif crystal_system == 'TRICLINIC':
        # *** BUG FIX (audit pass) ***
        # The previous form was missing one lattice-constant factor on
        # every term under the square root (e.g. h^2*a^2*sin^2(beta)
        # instead of h^2*b^2*c^2*sin^2(alpha)). Result was wrong by
        # roughly an order of magnitude even in the orthorhombic limit.
        # Re-implemented using the standard S_ij metric-tensor form.
        ca, cb, cg = math.cos(alp), math.cos(bet), math.cos(gam)
        sa, sb, sg = math.sin(alp), math.sin(bet), math.sin(gam)
        V_cell = EoS.unitCellVolume(unit_cell)
        S11 = b*b * c*c * sa*sa
        S22 = a*a * c*c * sb*sb
        S33 = a*a * b*b * sg*sg
        S12 = a * b * c*c * (ca*cb - cg)
        S23 = a*a * b * c * (cb*cg - ca)
        S13 = a * b*b * c * (cg*ca - cb)
        inv_d2 = (1.0 / (V_cell*V_cell)) * (
            S11*H*H + S22*K*K + S33*L*L
            + 2.0*S12*H*K + 2.0*S23*K*L + 2.0*S13*H*L)
        dk = 1.0 / np.sqrt(inv_d2)
    else:
        raise ValueError(f"Unknown crystal_system: {crystal_system}")

    d_min = twoTheta_to_dspacing(twoThetaMax, WL)
    d_max = twoTheta_to_dspacing(twoThetaMin, WL)
    mask = (dk > d_min) & (dk < d_max)

    dk_in = dk[mask]
    twoTh_in = dspacing_to_twoTheta(dk_in, WL)
    H_in, K_in, L_in = H[mask], K[mask], L[mask]

    twoTheta_reflection_list = np.column_stack((twoTh_in, H_in, K_in, L_in))
    dk_reflection_list = np.column_stack((dk_in, H_in, K_in, L_in))
    return dk_reflection_list, twoTheta_reflection_list


def initialise_parameters_LB(phases, WL, twoThetaMin=5, twoThetaMax=40,
                             sigma_mode='caglioti', ampGuess=1, sigGuess=0.06,
                             sigmaBounds=2.0, ampBounds=3.0, maxShift=0.5):
    """
    ampBounds : upper bound multiplier for amplitudes. Each amplitude is
                bounded by [0, amp_init * ampBounds]. Default 3.0 reproduces
                the previous hard-coded behaviour. Mirrors sigmaBounds.
    maxShift  : fractional constraint on lattice parameters (0.0-1.0).
                0.5 = ±50% (default, original behaviour).
                0.05 = ±5% (tight, useful when unit cell is well-known).
    """
    '''
    Defines the parameters for the fit. The params are a dictionary based on unit cell dimensions and HKL values.
    '''
    print('Initialising Parameters')

    # If the user typed 0 in the "Lattice Bounds %" spinbox, the
    # intent is "fix the lattice for every active phase". lmfit
    # forbids min == max on a parameter being added with vary=True
    # though, so we widen the bounds for the params.add() calls below
    # and then set vary=False on every lattice parameter in the
    # post-loop fix-lattice block.
    fix_lattice_globally = (maxShift is not None and maxShift <= 0)
    bounds_shift = 0.5 if fix_lattice_globally else maxShift
    
    # Defining the various parameters
    params = Parameters()
    # Parameters are bounded between min value of 1.0 and max value of 3.0 
    # add(name, value=None, vary=True, min=-inf, max=inf, expr=None, brute_step=None)

    # BG offset bounds scale with the intensity range (ampGuess is normally
    # max(valueIntBGsub) * ampPrefactor at the call site). Using ±0.05*ampGuess
    # keeps the residual BG correction at ~5% of the peak-amplitude scale,
    # so it works whether the data are raw counts in the thousands or
    # normalised to ~1. Falls back to ±0.5 if ampGuess is non-positive
    # (e.g. an empty/zero pattern) so we never set min=max=0 which lmfit
    # would reject.
    
    # This would vary BG freely, but assesments suggest that after ALS, the
    # baseline is already near 0. Freely varying it is a bad idea.
    # bg_bound = 0.05 * abs(ampGuess) if ampGuess and ampGuess > 0 else 0.5
    # params.add('BG', value = 0.0, min = -bg_bound, max = bg_bound)
    # params['BG'].set(vary=True)
    
    params.add('BG', value = 0.0, min= -.5, max=.5)
    # params['BG'].set(vary=True)
    
    params.add('WL', value = WL, min= WL*0.95, max= WL*1.15)
    params['WL'].set(vary=False)
    
    # phasesNumber = phases['phases_Number']
    # update_phases_used([0,2], phases)
    
    # Use bounds_shift for the bounds of every params.add call below,
    # so that a user request of maxShift=0 doesn't blow up at construction
    # time. The actual fixing happens in the post-loop block via vary=False.
    maxShift = bounds_shift

    for i, phase in enumerate(phases['phases_used']):
        # print(i)
        # print(phase)
        # print(phases['phase_'+str(phase)+'_unit_cell'][0])
        if phases['phase_'+str(phase)+'_crystal_system'] == 'CUBIC':
            params.add('a_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][0], 
                        min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                        max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), expr='a_'+str(phase))  # 'b' is always equal to 'a'
            params.add('c_'+str(phase), expr='a_'+str(phase))
            params.add('alp_'+str(phase), value =  phases['phase_'+str(phase)+'_unit_cell'][3], 
                        min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift), 
                        max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params['alp_'+str(phase)].set(vary=False)
            params.add('bet_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][4], 
                        min = phases['phase_'+str(phase)+'_unit_cell'][4]*(1-maxShift), 
                        max = phases['phase_'+str(phase)+'_unit_cell'][4]*(1+maxShift))
            params['bet_'+str(phase)].set(vary=False)
            params.add('gam_'+str(phase), value =  phases['phase_'+str(phase)+'_unit_cell'][5], 
                        min = phases['phase_'+str(phase)+'_unit_cell'][5]*(1-maxShift), 
                        max = phases['phase_'+str(phase)+'_unit_cell'][5]*(1+maxShift))
            params['gam_'+str(phase)].set(vary=False)
        elif phases['phase_'+str(phase)+'_crystal_system'] == 'TETRAGONAL':
            params.add('a_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][0], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), expr='a_'+str(phase))  # 'b' is always equal to 'a'
            params.add('c_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][2], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][2]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][2]*(1+maxShift))
            params.add('alp_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][3], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params['alp_'+str(phase)].set(vary=False)
            params.add('bet_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][4], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][4]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][4]*(1+maxShift))
            params['bet_'+str(phase)].set(vary=False)
            params.add('gam_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][5], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][5]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][5]*(1+maxShift))
            params['gam_'+str(phase)].set(vary=False)
        elif phases['phase_'+str(phase)+'_crystal_system'] == 'HEXAGONAL':   
            params.add('a_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][0], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), expr='a_'+str(phase))  # 'b' is always equal to 'a'
            params.add('c_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][2], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][2]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][2]*(1+maxShift))
            params.add('alp_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][3], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params['alp_'+str(phase)].set(vary=False)
            params.add('bet_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][4], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][4]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][4]*(1+maxShift))
            params['bet_'+str(phase)].set(vary=False)
            params.add('gam_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][5], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][5]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][5]*(1+maxShift))
            params['gam_'+str(phase)].set(vary=False)
        elif phases['phase_'+str(phase)+'_crystal_system'] == 'RHOMBOHEDRAL':
            params.add('a_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][0], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), expr='a_'+str(phase))  # 'b' is always equal to 'a'
            params.add('c_'+str(phase), expr='a_'+str(phase))  # 'c' is always equal to 'a'
            params.add('alp_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][3], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params.add('bet_'+str(phase), expr='alp_'+str(phase))  # 'b' is always equal to 'a'
            params.add('gam_'+str(phase), expr='alp_'+str(phase))  # 'b' is always equal to 'a'
        elif phases['phase_'+str(phase)+'_crystal_system'] == 'ORTHOROMBIC':
            params.add('a_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][0], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][1], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][1]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][1]*(1+maxShift))
            params.add('c_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][2], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][2]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][2]*(1+maxShift))
            params.add('alp_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][3], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params['alp_'+str(phase)].set(vary=False)
            params.add('bet_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][4], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][4]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][4]*(1+maxShift))
            params['bet_'+str(phase)].set(vary=False)
            params.add('gam_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][5], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][5]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][5]*(1+maxShift))
            params['gam_'+str(phase)].set(vary=False)
        elif phases['phase_'+str(phase)+'_crystal_system'] == 'MONOCLINIC':
            params.add('a_'+str(phase), value =phases['phase_'+str(phase)+'_unit_cell'][0], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][1], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][1]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][1]*(1+maxShift))
            params.add('c_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][2], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][2]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][2]*(1+maxShift))
            params.add('alp_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][3], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params['alp_'+str(phase)].set(vary=False)
            params.add('bet_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][4],
                        min = phases['phase_'+str(phase)+'_unit_cell'][4]*(1-maxShift), 
                        max = phases['phase_'+str(phase)+'_unit_cell'][4]*(1+maxShift))
            #params['bet_'+str(phase)].set(vary=False)
            params.add('gam_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][5], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][5]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][5]*(1+maxShift))   
            params['gam_'+str(phase)].set(vary=False)
        else:
            params.add('a_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][0], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][0]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][0]*(1+maxShift))
            params.add('b_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][1], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][1]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][1]*(1+maxShift))
            params.add('c_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][2], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][2]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][2]*(1+maxShift))
            params.add('alp_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][3], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][3]*(1-maxShift),
                       max = phases['phase_'+str(phase)+'_unit_cell'][3]*(1+maxShift))
            params.add('bet_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][4], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][4]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][4]*(1+maxShift))
            params.add('gam_'+str(phase), value = phases['phase_'+str(phase)+'_unit_cell'][5], 
                       min = phases['phase_'+str(phase)+'_unit_cell'][5]*(1-maxShift), 
                       max = phases['phase_'+str(phase)+'_unit_cell'][5]*(1+maxShift))
            #params.add('sig1', value = 0.027,  min= 0.0, max= 0.05)
         
        # ---- Per-frame "Fix lattice" hook (item 2) ---------------------
        # Two paths trigger lattice-fixing for THIS phase on THIS call:
        #   1. phases[f'phase_{X}_fix_lattice'] is True
        #      (set per-frame by EXODUS_main._process_next_frame in
        #      pressureGuess mode when a row's 'Fix' checkbox is ticked)
        #   2. fix_lattice_globally - the user typed 0 in the "Lattice
        #      Bounds %" spinbox in the GUI, signalling "fix every
        #      phase" for this fit.
        # In either case we set vary=False on every lattice parameter
        # (a, b, c, alp, bet, gam) for that phase. The values stay at
        # whatever was just placed in phases[...]_unit_cell, so the
        # fit only refines amplitudes / sigmas / background / wavelength.
        if phases.get(f'phase_{phase}_fix_lattice', False) or fix_lattice_globally:
            for nm in ('a', 'b', 'c', 'alp', 'bet', 'gam'):
                key = f'{nm}_{phase}'
                if key in params:
                    # Some keys are constrained via expr= (e.g. CUBIC's
                    # b/c -> a). lmfit forbids vary=False on an
                    # expression-bound parameter; skip those quietly.
                    if getattr(params[key], 'expr', None):
                        continue
                    params[key].set(vary=False)

        dk_reflection_list, twoTheta_reflection_list = ([] for i in range(2))
        # `dk_reflection_list` / `twoTheta_reflection_list` are produced
        # here for use elsewhere in this module (peak overlays etc.).
        # They used to be built with a `twoThetaMax + 5` buffer to
        # provide "phantom" amplitude params for reflections that might
        # drift into the fit window during refinement; that buffer is
        # no longer needed because amplitudes (and per-peak sigmas)
        # are now keyed by HKL index and a parameter is created for
        # EVERY HKL entry, with out-of-window ones set to vary=False -
        # see the amp-creation block below.
        dk_reflection_list, twoTheta_reflection_list = reflection_List(phases['phase_'+str(phase)+'_unit_cell'], 
                                                                       phases['phase_'+str(phase)+'_HKL'], 
                                                                       WL, twoThetaMin, twoThetaMax,
                                                                       crystal_system = phases['phase_'+str(phase)+'_crystal_system']
                                                                       )

        # ----------------------------------------------------------------
        # Amplitude parameters MUST be keyed by HKL index, not by
        # filtered-list ordinal. LB_fit_Model and _add_phase_contribution
        # look up `amp_{phase}_{HKL_index+1}` (their `visible_indices` are
        # positions in the FULL HKL array, see line `mask = (dk > d_min) ...`
        # in LB_fit_Model). Previously this loop used `j` from
        # `enumerate(dk_reflection_list)` -- the ordinal in the FILTERED
        # list -- which only happens to coincide with HKL index when every
        # HKL entry falls inside the fit window. As soon as one or more
        # HKL reflections sit outside, the keys diverge and the residual
        # KeyErrors on something like 'amp_1_7'. The fix below uses the
        # HKL-index keying scheme consistently.
        #
        # We create a parameter for EVERY HKL entry. Reflections outside
        # the fit window are added with vary=False so they don't bloat
        # the optimisation but they're still present in the dict and get
        # a sensible value should the lattice shift bring them into view
        # at runtime.
        HKL_full = np.asarray(phases[f'phase_{phase}_HKL'], dtype=float)
        n_hkl = HKL_full.shape[0]

        # Compute 2theta for ALL HKL entries at the initial unit cell, in
        # HKL order, using the same physics as reflection_List but with
        # an "all-pass" 2theta filter so we get one entry per HKL row.
        _, tt_all = reflection_List(
            phases[f'phase_{phase}_unit_cell'],
            HKL_full, WL,
            twoThetaMin=0.0, twoThetaMax=180.0,
            crystal_system=phases[f'phase_{phase}_crystal_system'])

        # tt_all rows are [twoTheta, H, K, L] for every HKL whose 2theta
        # is finite in (0, 180]. Build an HKL-index aligned 2theta array,
        # padding any entries that didn't pass (e.g. forbidden reflections
        # producing NaN d-spacing) with +inf so they're flagged as
        # out-of-window without raising on comparison.
        tt_by_hkl_idx = np.full(n_hkl, np.inf)
        if tt_all.size:
            # Match each tt_all row back to its HKL index by H,K,L lookup.
            # Build a hashable lookup from the HKL array first.
            hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): i
                        for i, (h, k, l) in enumerate(HKL_full[:, :3])}
            for row in tt_all:
                key = (int(round(row[1])), int(round(row[2])), int(round(row[3])))
                idx = hkl_keys.get(key)
                if idx is not None:
                    tt_by_hkl_idx[idx] = row[0]

        in_window  = (tt_by_hkl_idx >= twoThetaMin) & (tt_by_hkl_idx <= twoThetaMax)
        n_visible  = int(in_window.sum())

        # Creates Amplitudes - one per HKL index, keys aligned with
        # the visible_indices values that LB_fit_Model will look up.
        for hkl_idx in range(n_hkl):
            amp_init = ampGuess / 100.0 * HKL_full[hkl_idx, 3]
            upper = amp_init * ampBounds if amp_init > 0 else ampGuess * ampBounds
            # Guard against min==max==0 which lmfit rejects. Happens when
            # ampGuess <= 0 (e.g. an empty pattern, or the user passed 0).
            # A nominal small upper bound lets the parameter be added; it
            # will be pinned at 0 by `min=0` until the optimiser pushes it
            # up if there's any signal.
            if upper <= 0:
                upper = 1.0
            params.add(
                f'amp_{phase}_{hkl_idx+1}',
                value=amp_init,
                min=0.0,
                max=upper)  # generous upper bound
            # Only let amps for in-window reflections vary at the start.
            # Out-of-window reflections start frozen so the Jacobian
            # stays full rank; if the lattice shifts and they drift into
            # view, LB_fit_Model still finds the parameter (it's just
            # frozen at its starting value, which is a reasonable proxy
            # for an "average" amplitude).
            # if not in_window[hkl_idx]:
            #     params[f'amp_{phase}_{hkl_idx+1}'].set(vary=False)
            
            # Freeze amp if either: out-of-window, OR JCPDS intensity is zero/negative.  
            # A reflection with zero structure factor has no amplitude to fit and would
            # otherwise contribute a zero column to the Jacobian, making J^T J singular
            # and covar unavailable.
            if (not in_window[hkl_idx]) or (HKL_full[hkl_idx, 3] <= 0.0):
                params[f'amp_{phase}_{hkl_idx+1}'].set(vary=False)
            
        # Calls the sigma setup function
        setup_sigma_params(params, sigma_mode,
            phase, phases[f'phase_{phase}_HKL'], sigGuess, sigmaBounds=sigmaBounds,
            in_window_mask=in_window)

        # If maxShift==0, freeze all lattice parameters for this phase
        if maxShift == 0:
            for lp in ['a_', 'b_', 'c_', 'alp_', 'bet_', 'gam_']:
                key = lp + str(phase)
                if key in params and params[key].expr is None:
                    params[key].set(vary=False)
    
        # # ---- mode selection (must be mutually exclusive) ----
        # sigma_fixed = False
        # sigma_per_phase = False
        # sigma_caglioti = True
        # sigma_separate = False  # (fixed spelling)
        
        # modes = [sigma_fixed, sigma_per_phase, sigma_caglioti, sigma_separate]
        # if sum(modes) != 1:
        #     raise ValueError("Exactly one sigma mode must be True")
        

        # if sigma_fixed:
        #     # Uses only one sigma for all peaks
        #     # Fastest alternative
        #     for j, _ in enumerate(dk_reflection_list):
        #         params.add(
        #             f'sig_{phase}_{j+1}',
        #             value=sigGuess,
        #             min=0.0, max=sigGuess * 5)
        
        
        # elif sigma_per_phase:
        #     # Uses a different fixed sigma for every phase
        #     # Not very phyiscal but maybe useful for strained samples
        #     for j, _ in enumerate(dk_reflection_list):
        #         if phase == 0:
        #             params.add(
        #                 f'sig_{phase}_{j+1}',
        #                 value=sigGuess,
        #                 min=0.0, max=sigGuess * 5)
        #         else:
        #             params.add(
        #                 f'sig_{phase}_{j+1}',
        #                 value=sigGuess,
        #                 min=0.0, max=sigGuess * 2)
        

        # elif sigma_caglioti:
        #     # Uses a 1 paramter Caglioti approximation of 1/cos(theta) 
        #     # Only on sigma is used
        #     for j, _ in enumerate(dk_reflection_list):
        #         theta = twoTheta_reflection_list[j][0] / 2 * deg_to_rad
        #         if phase == 0:
        #             params.add(
        #                 f'sig_{phase}_{j+1}',
        #                 value=sigGuess,
        #                 min=0.0, max=sigGuess * 2)
        #         else:
        #             params.add(
        #                 f'sig_{phase}_{j+1}',
        #                 value=sigGuess / np.cos(theta),
        #                 min=0.0, max=sigGuess * 2)
        

        # elif sigma_separate:
        #     # Creates a seperate sigma for every single reflection
        #     # NOT recommended. Will increase computation time massivley.
        #     for j, hkl in enumerate(dk_reflection_list):
        #         if phase == 0:
        #             params.add(
        #                 f'sig_{phase}_{j+1}',
        #                 value=sigGuess,
        #                 min=0.0, max=sigGuess * 5)
        #         else:
        #             params.add(
        #                 f'sig_{phase}_{j+1}',
        #                 value=sigGuess,
        #                 min=0.0, max=sigGuess * 2)
        
    return params


def setup_sigma_params(params, sigma_mode, phase,
                       HKL, sigGuess, sigmaBounds=2.0,
                       in_window_mask=None):
    """
    Set up sigma (peak-width) parameters for one phase.

    Parameters
    ----------
    params : lmfit.Parameters
        The parameter set to add to (modified in place).
    sigma_mode : {'fixed', 'per_phase', 'caglioti', 'separate'}
    phase : int
        Phase index (used in parameter names).
    HKL : np.ndarray
        The full HKL array for the phase, shape (n_hkl, >=4). Only its
        first dimension matters here - we use it as the index space for
        per-peak sigmas in 'separate' mode.
    sigGuess : float
        Initial sigma value.
    sigmaBounds : float
        Upper bound multiplier (default 2.0). Max = sigGuess * sigmaBounds.
        Controlled by the 'Sigma Bounds' spinbox in the GUI.
    in_window_mask : np.ndarray of bool or None
        Per-HKL-index bool array marking which reflections fall inside
        the actual fit window. Used in 'separate' mode to freeze sigmas
        for out-of-window peaks (vary=False) so they don't introduce
        zero-Jacobian columns. If None, every sigma starts varying.

    Notes
    -----
    'separate' mode used to key sigmas by filtered-list ordinal
    (`for j, _ in enumerate(dk_reflection_list)`). _add_phase_contribution
    looks up sigmas by HKL index (`pv[f'sig_{phase}_{int(idx)+1}']`),
    so the keys diverged whenever any HKL entry sat outside the fit
    window. A fallback at runtime quietly substituted the highest-
    index sigma when a key was missing, hiding the bug at the cost of
    using the wrong sigma. This function now keys by HKL index, the
    same way amplitudes do, and the fallback in _add_phase_contribution
    becomes a redundant safety net rather than a routine path.
    """
    sig_max = sigGuess * sigmaBounds
    HKL_arr = np.asarray(HKL)
    n_hkl = HKL_arr.shape[0]

    # GLOBAL FIXED (one sigma total, not angle-dependent)
    if sigma_mode == "fixed":
        if 'sig_global' not in params:
            params.add('sig_global',
                       value=sigGuess, min=0.0, max=sig_max)

    # ONE SIGMA PER PHASE
    elif sigma_mode == "per_phase":
        params.add(f'sig_{phase}',
                   value=sigGuess, min=0.0, max=sig_max)

    # CAGLIOTI 1/cos(theta) - one global base sigma, angle-dependent width
    elif sigma_mode == "caglioti":
        if 'sig_global' not in params:
            params.add('sig_global',
                       value=sigGuess, min=0.0, max=sig_max)

    # ONE SIGMA PER PEAK (most flexible, slowest)
    elif sigma_mode == "separate":
        for hkl_idx in range(n_hkl):
            params.add(f'sig_{phase}_{hkl_idx+1}',
                       value=sigGuess, min=0.0, max=sig_max)
            # Freeze sigmas for reflections outside the fit window so
            # they don't introduce zero-Jacobian columns. They stay
            # in the dict so _add_phase_contribution always finds them.
            if in_window_mask is not None and not in_window_mask[hkl_idx]:
                params[f'sig_{phase}_{hkl_idx+1}'].set(vary=False)

    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")
    
    
def get_sigma(params, sigma_mode, phase, j, cen, deg_to_rad):
    # Matching sigma usage in the Model function
    if sigma_mode == "fixed":
        return params['sig_global'].value

    elif sigma_mode == "per_phase":
        return params[f'sig_{phase}'].value

    elif sigma_mode == "caglioti":
        theta = cen / 2 * deg_to_rad
        sigma0 = params['sig_global'].value
        return sigma0 / np.cos(theta)
        #return sigma0 *(1 + 25* np.tan(theta)**2)

    elif sigma_mode == "separate":
        # Guard: if a reflection shifted into the window during fitting and j exceeds
        # the number of sigma params created at initialisation, reuse the last one.
        key = f'sig_{phase}_{j+1}'
        if key not in params:
            # Find the highest created index for this phase
            existing = [k for k in params if k.startswith(f'sig_{phase}_')]
            if existing:
                key = sorted(existing, key=lambda k: int(k.rsplit('_',1)[-1]))[-1]
            else:
                raise KeyError(f"No sigma parameters found for phase {phase}")
        return params[key].value

    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")











# Define the fitting function
def LB_fit_Model(params, x, y, phases, twoThetaMin=0, twoThetaMax=30,
                 sigma_mode='caglioti'):
    """
    Vectorised residual.
    - One params.valuesdict() call (avoids per-parameter lmfit / asteval
      lookups in the hot loop).
    - Per-phase HKL stencils are cached (CUBIC/TETRAGONAL/HEXAGONAL fast
      path) so we skip the int->float cast and crystal-system dispatch
      that reflection_List does.
    - Peak superposition is built via NumPy broadcasting rather than a
      Python loop over reflections.
    """
    pv = params.valuesdict()
    y_fit = np.full_like(x, pv['BG'])
    WL = pv['WL']

    d_min = twoTheta_to_dspacing(twoThetaMax, WL)
    d_max = twoTheta_to_dspacing(twoThetaMin, WL)

    for phase in phases['phases_used']:
        HKL = phases[f'phase_{phase}_HKL']
        cs = phases[f'phase_{phase}_crystal_system']
        a = pv[f'a_{phase}']

        stencil = _LB_phase_stencil(HKL, cs)
        if stencil[0] == 'CUBIC':
            _, H, K, L, denom = stencil
            dk = a / np.sqrt(denom)
        elif stencil[0] in ('TETRAGONAL', 'HEXAGONAL'):
            _, H, K, L, ab_term, c_term = stencil
            c = pv[f'c_{phase}']
            dk = 1.0 / np.sqrt(ab_term / (a*a) + c_term / (c*c))
        else:
            # Fall back through the public reflection_List for less common
            # crystal systems (rhombohedral/orthorhombic/monoclinic/triclinic).
            unit_cell = (a, pv[f'b_{phase}'], pv[f'c_{phase}'],
                         pv[f'alp_{phase}'], pv[f'bet_{phase}'], pv[f'gam_{phase}'])
            _, twoTh_full = reflection_List(
                unit_cell, HKL, WL,
                crystal_system=cs,
                twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
            n_refl = len(twoTh_full)
            if n_refl == 0:
                continue
            cens = twoTh_full[:, 0]
            # Recover the original HKL-array indices for the reflections that
            # passed reflection_List's window filter. Amplitudes and sigmas
            # are keyed by HKL index in the FULL HKL array (see the long
            # comment block in initialise_parameters_LB around line 1095).
            # Previously this fallback passed np.arange(n_refl), the
            # filtered-list ordinal, which collided with the HKL index
            # whenever any reflection sat outside the fit window -- giving
            # wrong amps/sigmas to the fit for non-fast-path crystal
            # systems (rhombohedral/ortho/mono/triclinic).
            HKL_full = np.asarray(HKL)
            hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): i
                        for i, (h, k, l) in enumerate(HKL_full[:, :3])}
            visible_indices = np.empty(n_refl, dtype=np.int64)
            for r, row in enumerate(twoTh_full):
                key = (int(round(row[1])), int(round(row[2])), int(round(row[3])))
                visible_indices[r] = hkl_keys.get(key, -1)
            # Drop any rows whose HKL didn't round-trip back to the source
            # array (shouldn't happen with integer HKLs, but guard anyway
            # so a stray -1 doesn't index amp_<phase>_0).
            ok = visible_indices >= 0
            if not ok.all():
                cens = cens[ok]
                visible_indices = visible_indices[ok]
                if cens.size == 0:
                    continue
            _add_phase_contribution(pv, phase, cens, visible_indices,
                                    sigma_mode, x, y_fit)
            continue

        # Filter to reflections inside the fit window (fast path)
        mask = (dk > d_min) & (dk < d_max)
        if not mask.any():
            continue
        dk_in = dk[mask]
        cens = dspacing_to_twoTheta(dk_in, WL)
        visible_indices = np.flatnonzero(mask)

        _add_phase_contribution(pv, phase, cens, visible_indices,
                                sigma_mode, x, y_fit)

    return y_fit - y


def _add_phase_contribution(pv, phase, cens, visible_indices, sigma_mode, x, y_fit):
    """Add this phase's Gaussian peak superposition to y_fit (in place)."""
    n_refl = len(cens)
    if n_refl == 0:
        return

    # Amplitudes -- indexed by visible-reflection ordinal at INIT time.
    # If a reflection drifted out of the window during the fit,
    # visible_indices skips its slot, which keeps the indexing stable.
    amps = np.fromiter(
        (pv[f'amp_{phase}_{int(idx)+1}'] for idx in visible_indices),
        dtype=np.float64, count=n_refl)

    if sigma_mode == 'caglioti':
        sigs = pv['sig_global'] / np.cos(cens / 2 * deg_to_rad)
    elif sigma_mode == 'fixed':
        sigs = np.full(n_refl, pv['sig_global'])
    elif sigma_mode == 'per_phase':
        sigs = np.full(n_refl, pv[f'sig_{phase}'])
    elif sigma_mode == 'separate':
        # match legacy get_sigma fallback to highest-index sig if a peak
        # drifted into the window
        available_keys = [k for k in pv if k.startswith(f'sig_{phase}_')]
        if not available_keys:
            raise KeyError(f"No sigma parameters found for phase {phase}")
        last_key = max(available_keys, key=lambda k: int(k.rsplit('_', 1)[-1]))
        sigs = np.fromiter(
            (pv.get(f'sig_{phase}_{int(idx)+1}', pv[last_key])
             for idx in visible_indices),
            dtype=np.float64, count=n_refl)
    else:
        raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

    # Broadcast (n_refl, n_x): each reflection contributes a Gaussian.
    diff = (x[None, :] - cens[:, None]) / sigs[:, None]
    y_fit += (amps[:, None] * np.exp(-0.5 * diff * diff)).sum(axis=0)









def unitCell_fit(result, phase):
    '''
    Creates a unit cell np.array from the results
    '''
    unitCell_fit = [result.params['a_'+str(phase)].value, 
                    result.params['b_'+str(phase)].value, 
                    result.params['c_'+str(phase)].value,
                    result.params['alp_'+str(phase)].value, 
                    result.params['bet_'+str(phase)].value, 
                    result.params['gam_'+str(phase)].value
                    ]
    #print('\nUnit Cell '+ str(phase) + ' fitted: \n' + str(unitCell_fit))
    unitCell_fit = np.array(unitCell_fit)
    return unitCell_fit


def create_ticks_guess(phases, twoThetaMin, twoThetaMax, WL):
    '''
    Returns a list of list of 2theta reflections arrays for each phase.
    '''
    tickArray = []
    for phase in phases['phases_used']:
        _, twoTheta_reflection_list = reflection_List(phases[f'phase_{phase}_unit_cell'],
                                                      phases['phase_' + str(phase) + '_HKL'],
                                                      WL,
                                                      twoThetaMin=twoThetaMin,
                                                      twoThetaMax=twoThetaMax,
                                                      crystal_system=phases['phase_' + str(phase) + '_crystal_system']
                                                      )
        # Take only the first column (2Theta values) and append as a separate list
        tickArray.append(twoTheta_reflection_list[:, 0].tolist())
    return tickArray  # list of lists


def create_ticks(result, phases, twoThetaMin, twoThetaMax, WL):
    '''
    Returns a list of list of 2theta reflections arrays for each phase.
    '''
    tickArray = []
    for phase in phases['phases_used']:
        _, twoTheta_reflection_list = reflection_List(unitCell_fit(result, phase),
                                                      phases['phase_' + str(phase) + '_HKL'],
                                                      WL,
                                                      twoThetaMin=twoThetaMin,
                                                      twoThetaMax=twoThetaMax,
                                                      crystal_system=phases['phase_' + str(phase) + '_crystal_system']
                                                      )
        # Take only the first column (2Theta values) and append as a separate list
        tickArray.append(twoTheta_reflection_list[:, 0].tolist())
    return tickArray  # list of lists




def create_guessComponents(twoTheta, phases, params, twoThetaMin, twoThetaMax, 
                           sigma_mode = 'caglioti', printInputComponents = False):
    ''' 
    Creates the components for the overall envelope function and returns the envelope.

    Amplitudes and per-peak sigmas are keyed by the index of the reflection
    in the FULL HKL array (the same scheme used by initialise_parameters_LB
    and LB_fit_Model). Previously this looped `for j, _ in enumerate(...)`
    over the WINDOW-FILTERED reflection list and looked up `amp_<phase>_<j+1>`
    -- which only happens to coincide with the HKL index when every HKL row
    falls inside the fit window. As soon as any row was filtered out, the
    displayed/saved per-peak components used the wrong amplitudes.
    '''
    components = {}
    components['BG'] = params['BG'].value  # always present even if no phases
    for i, phase in enumerate(phases['phases_used']):
        HKL = phases['phase_'+str(phase)+'_HKL']
        dk_reflection_list, twoTheta_reflection_list = reflection_List(
              [params['a_'+str(phase)].value, params['b_'+str(phase)].value, params['c_'+str(phase)].value,
               params['alp_'+str(phase)].value, params['bet_'+str(phase)].value, params['gam_'+str(phase)].value],
              HKL, WL=params['WL'].value, crystal_system=phases['phase_'+str(phase)+'_crystal_system'],
              twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax
              )
        # Build HKL-triplet -> full-array-index lookup so we can recover
        # the original HKL index for each window-passing reflection.
        HKL_full = np.asarray(HKL)
        hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): idx
                    for idx, (h, k, l) in enumerate(HKL_full[:, :3])}
        for j, dk_reflection in enumerate(dk_reflection_list):
            cen = twoTheta_reflection_list[j][0]
            row = twoTheta_reflection_list[j]
            hkl_idx = hkl_keys.get(
                (int(round(row[1])), int(round(row[2])), int(round(row[3]))), j)
            amp = params['amp_'+str(phase) + '_' + str(hkl_idx+1)].value
            sig = get_sigma(params, sigma_mode, phase, hkl_idx, cen, deg_to_rad)
            if printInputComponents == True:
                print('xc '+ str(cen) + '\tamp ' + str(amp) + '\tsig ' + str(sig))
            components['comp_'+str(phase)+'_'+str(hkl_idx+1)] = amp * np.exp(-0.5 * ((twoTheta-cen)/sig)**2)
    # Sum outside the loop so bestFit always exists (even for zero phases)
    bestFit = sum(components.values())
    return bestFit, components



def create_fitComponents(twoTheta, phases, params, result, save_path, frame, valueIntBGsub,
                         twoThetaMin, twoThetaMax, sigma_mode = 'caglioti', 
                         printOutputComponents = False):
    ''' 
    Creates the components for the overall envelope function and returns the envelope.
    Same HKL-index keying as create_guessComponents (see its docstring).
    ''' 
    peak_fit_list = []
    components = {}
    peak_fit = {}
    for i, phase in enumerate(phases['phases_used']):
        HKL = phases['phase_'+str(phase)+'_HKL']  
        dk_reflection_list, twoTheta_reflection_list =  reflection_List([result.params['a_'+str(phase)].value, result.params['b_'+str(phase)].value, result.params['c_'+str(phase)].value,
                        result.params['alp_'+str(phase)].value,result.params['bet_'+str(phase)].value,result.params['gam_'+str(phase)].value], 
                        HKL, result.params['WL'].value, crystal_system = phases['phase_'+str(phase)+'_crystal_system'], 
                        twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)

        # Build HKL-triplet -> full-array-index lookup so amp/sig keys
        # match what initialise_parameters_LB / LB_fit_Model use.
        HKL_full = np.asarray(HKL)
        hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): idx
                    for idx, (h, k, l) in enumerate(HKL_full[:, :3])}

        for j, dk_reflection in enumerate(dk_reflection_list):
            cen = twoTheta_reflection_list[j][0]
            row = twoTheta_reflection_list[j]
            hkl_idx = hkl_keys.get(
                (int(round(row[1])), int(round(row[2])), int(round(row[3]))), j)
            amp = result.params['amp_'+str(phase) + '_' + str(hkl_idx+1)].value
            sig = get_sigma(result.params, sigma_mode, phase, hkl_idx, cen, deg_to_rad)
            if printOutputComponents == True:
                print('xc '+ str(cen) + '\tamp ' + str(amp) + '\tsig ' + str(sig))
            peak_fit_list.append([cen, amp, sig])
            components['comp_'+str(phase)+'_'+str(hkl_idx+1)] = amp * np.exp(-0.5 * ((twoTheta-cen)/sig)**2)
        peak_fit['phase_'+str(phase)] = np.array(peak_fit_list)
        # Create subfolder (if it doesn't exist and save_path is provided)
        if save_path:
            subfolder_path = save_path + "/fits"
            os.makedirs(subfolder_path, exist_ok=True)

    # BG and final sum are OUTSIDE the phase loop so bestFit always exists,
    # even if phases_used is empty.
    components['BG'] = result.params['BG'].value
    bestFit = sum(components.values())
    return bestFit, components
    

# LEGACY: Unused - confidence interval approach not implemented
# LEGACY: residual_for_minimizer commented out - not used by GUI
# def residual_for_minimizer(params, x, y, phases, twoThetaMin=0, twoThetaMax=30):
#     # It should return residual array
#     return LB_fit_Model(params, x, y, phases, twoThetaMin, twoThetaMax,
#                         sigma_mode = 'caglioti')


def calculate_Rw(y_obs, y_calc, weights=None):
    """
    Weighted profile R-factor (Rwp), as defined in the Rietveld refinement
    literature: sqrt( sum(w*(y_obs-y_calc)^2) / sum(w*y_obs^2) ).

    With `weights=None` (the default) we use Poisson counting-statistics
    weights w_i = 1/max(|y_obs_i|, 1) so the result is the conventional
    Rwp. Pass `weights=np.ones_like(y_obs)` to recover the old unweighted
    R-profile (Rp) behaviour. Pass `weights = 1/sigma**2` if you have a
    real per-point variance.

    Note: y_obs here is BG-subtracted, so it can legitimately be small or
    even slightly negative. The clip at 1 stops 1/y blowing up on
    near-zero counts (which would otherwise dominate Rw entirely).
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_calc = np.asarray(y_calc, dtype=float)
    if weights is None:
        # # Gives Rwp - is buggy due to BG subtraction before fit is carried out
        # weights = 1.0 / np.maximum(np.abs(y_obs), 1.0)
        
        # # Uses Rp - while not as good as Rwp, it gives a better goodness of
        # # fit for data which is processed in EXODUS' fashion. 
        weights = np.ones_like(y_obs)
        
        
    numerator = np.sum(weights * (y_obs - y_calc)**2)
    denominator = np.sum(weights * (y_obs)**2)
    if denominator <= 0:
        return float('nan')

    Rw = np.sqrt(numerator / denominator)
    return Rw

def fitStatistics(result, twoTheta, fullFitReport=False, verbose=False):
    """
    Print a summary of the fit and return a global error estimate
    (sqrt(reduced chi-square)) for compatibility with old call sites.

    Per-parameter standard errors (when available) live on
    `result.params[name].stderr`; use `unitCell_errors(result, phase)` to
    pull them out as a 6-element ndarray.

    Notes on the previous bugs this replaces
    ----------------------------------------
    The old version had two cascading problems:
      1. The 'is covariance available?' decision sat under an `elif` of
         `fullFitReport`, so it never actually checked `result.covar`.
      2. The fallback called `result.model.conf_interval()`, but
         `MinimizerResult` (returned by `lmfit.minimize`) has no `.model`
         attribute -- only `ModelResult` does. So the try/except always
         fell into chi^2 fallback, even though stderrs were sitting on
         `result.params`.
    """
    print(f"\nFitting success: {result.success}")

    if not result.success:
        print("Fit was not successful. No error estimates available.")
        return None

    dof = max(1, len(twoTheta) - len(result.var_names))
    chi2_red = result.chisqr / dof
    estimated_error_global = np.sqrt(chi2_red)
    print(f"Reduced Chi-squared: {chi2_red:.6e}")

    has_covar = result.covar is not None
    if has_covar:
        n_with = sum(1 for p in result.params.values()
                     if p.vary and p.stderr is not None)
        n_total = sum(1 for p in result.params.values() if p.vary)
        print(f"Covariance matrix: available "
              f"(stderrs on {n_with}/{n_total} varying params)")
    else:
        print("Covariance matrix: NOT available "
              "(fit may be ill-conditioned or sitting on bounds)")

    if fullFitReport:
        print("\nFit Report")
        print("=" * 50)
        print(f"Chi-square        : {result.chisqr:.4f}")
        print(f"Reduced Chi-square: {result.redchi:.4f}")
        print(f"Akaike info crit  : {result.aic:.4f}")
        print(f"Bayesian info crit: {result.bic:.4f}")
        print(f"nfev              : {result.nfev}")
        print("=" * 50)
        print("\nFitted Parameters:")
        for name, par in result.params.items():
            stderr_str = (f"±{par.stderr:.4g}" if par.stderr is not None
                          else "stderr=N/A")
            on_bound = ""
            if par.vary and par.min > -np.inf and par.max < np.inf:
                tol = 1e-6 * max(abs(par.min), abs(par.max), 1.0)
                if abs(par.value - par.min) < tol:
                    on_bound = "  [AT MIN]"
                elif abs(par.value - par.max) < tol:
                    on_bound = "  [AT MAX]"
            print(f"  {name:<14} = {par.value:>12.6g}  {stderr_str:<14} "
                  f"vary={str(par.vary):<5}{on_bound}")

    return estimated_error_global


def unitCell_errors(result, phase, fallback=None):
    """
    Return a 6-element ndarray [da, db, dc, dalpha, dbeta, dgamma] of
    1-sigma standard errors for the unit-cell parameters of `phase`.

    Constraint expressions (e.g. `b = a` in cubic) are followed: a slaved
    parameter inherits its master's stderr. Fixed parameters get error 0.
    A varying parameter that has no stderr (covar missing or hit a bound)
    returns `fallback` if provided, else NaN.
    """
    names = [f'a_{phase}', f'b_{phase}', f'c_{phase}',
             f'alp_{phase}', f'bet_{phase}', f'gam_{phase}']
    errs = np.zeros(6)
    for i, name in enumerate(names):
        par = result.params[name]
        if par.expr is not None:
            # For EXODUS the expressions are simple identifiers like 'a_0';
            # walk one hop. If it ever gets compound (a*b/c), .stderr on
            # the slave comes from lmfit's expr propagation when present.
            if par.stderr is not None:
                stderr = par.stderr
            else:
                master = result.params.get(par.expr.strip())
                stderr = master.stderr if master is not None else None
        elif not par.vary:
            stderr = 0.0
        else:
            stderr = par.stderr
        errs[i] = stderr if stderr is not None else (
            fallback if fallback is not None else np.nan)
    return errs


def fitStatistics_legacy(result, twoTheta, fullFitReport=False, verbose=False):
    """
    DEPRECATED. Kept as a shim for any external code that imported the old
    name. The original body called result.model.conf_interval(...), which
    is broken (MinimizerResult has no .model attribute -- only ModelResult
    does). Forwards to fitStatistics() instead.
    """
    import warnings
    warnings.warn(
        "fitStatistics_legacy is deprecated; use fitStatistics() instead.",
        DeprecationWarning, stacklevel=2,
    )
    return fitStatistics(result, twoTheta,
                         fullFitReport=fullFitReport, verbose=verbose)



def fit_LB(data_path, phases, poni_file, save_path, 
           frame = 10, twoThetaMin = 0, twoThetaMax = 10, 
           framePressureGuess = [[0,0],[3000,0]],
           prominence = 0.10, height = 0.08, order = 20, 
           ampGuess = 1, sigGuess = 0.05,
           excludePeakList=[], plotLB = True):   
    '''
    Subtracts the BG, and fits the data using the Le Bail method
    '''   
    
    # Load the n-th frame (i.e. diffraction file)
    twoTheta, valueInt, errorInt = load_frame(frame, data_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)    
    
    # Calculates the BG
    try: 
        twoTheta, valueBG = backgroundFit(data_path, frame=frame, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax,
                                             peakSearchAuto = True, plotBG=False, prominence=prominence, 
                                             height = height, order=order, excludePeakList=excludePeakList)   
    except:
        twoTheta = twoTheta
        valueBG = np.zeros_like(twoTheta)
        print('No Background subtraction performed. ')

    # Subtracts the BG and retruns two np.arrays with BG subtracted data
    twoTheta, _, _, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)  

    # Plots Data
    if plotLB == True:
        # Only run once
        plt.figure()
        plt.scatter(twoTheta, valueIntBGsub, marker='.', color='gray', label='Raw Data', s=3)
        
    # Experimental: Checks if there are redundant reflections, i.e. reflections that are 0  or near to 0 in this plot
    # HKL_array[0] = np.delete(HKL_array[0], 1, axis=0)
    # for phase, i in enumerate(HKL_array):
    #     for j in len(phase): 
    #         print(HKL_array[j,3])
            
            
    # Initialises the parameters for the fit, i.e. lattice parameters etc.
    params = initialise_parameters_LB(phases, WL = load_poni(poni_file), ampGuess = max(valueIntBGsub),
                                      twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax, 
                                      sigma_mode = 'caglioti', sigGuess = 0.05)   

    # Evaluate the individual components of the fitted model separately
    guessFit, components = create_guessComponents(twoTheta, phases, params, 
                                            twoThetaMin, twoThetaMax, 
                                            sigma_mode = 'caglioti', printInputComponents = False)
    
    # Plots the initial guess
    if plotLB == True:
        plt.plot(twoTheta, guessFit, c='blue', ls='dotted', lw=1, label='Initial guess') 
    
    
    # Fit data
    try:
        result = minimize(lambda params: LB_fit_Model(params, twoTheta, valueIntBGsub, phases, 
                                                      twoThetaMin, twoThetaMax, 
                                                      sigma_mode = 'caglioti'),
                          params,
                          method='least_squares',
                          #options={'max_nfev': 1000, 'verbose': 2},
                          calc_covar=True,
                          )        
        
        # Will be implemented later to get better errors:
        # # Create Minimizer
        # minimizer = Minimizer(residual_for_minimizer, params, 
        #                       fcn_args=(twoTheta, valueIntBGsub, phases, twoThetaMin, twoThetaMax)
        #                       )
        
        # # Perform minimization
        # result = minimizer.minimize()

        # # Now compute confidence intervals
        # ci = conf_interval(minimizer, result, sigmas=[1, 2], trace=False, maxiter=200)
        # printfuncs.report_ci(ci)
        
    except:
        print('Error in frame ' + str(frame))  
                              
    # Creates the envelope for the best fit and saves the data
    bestFit, components = create_fitComponents(twoTheta, phases, params, result, 
                                               save_path, frame, valueIntBGsub,
                                               twoThetaMin, twoThetaMax, 
                                               sigma_mode = 'caglioti', printOutputComponents = False)
    
    # Fitting Statistics
    printStats = False
    printStats = True
    if printStats == True:
        estimatedError = fitStatistics(result, twoTheta)
        Rw = calculate_Rw(y_obs = valueIntBGsub, y_calc = bestFit, weights=None)
        print(f'Rw for this fit is {Rw:.4g}')
        
    # Creates Tickmarks
    tickArray = create_ticks(result, phases, twoThetaMin, twoThetaMax, WL=load_poni(poni_file))
    
    # Saves the figure 
    if plotLB == True:  
        tickStep = max(valueIntBGsub)/10
        for j in range(len(tickArray)):
            for i in range(len(tickArray[j])):  # Only loop up to the number of reflections for that phase
                plt.scatter(tickArray[j][i], -tickStep - (j+2) * tickStep, 
                            marker='|', linewidths=2, s=100, color=colours[j])
                
        # Adds best fit to the plot        
        plt.plot(twoTheta, bestFit, c='red', ls='-', lw=1, label='Best fit')
        plt.plot(twoTheta, valueIntBGsub-bestFit-tickStep, c='blue', ls='-', lw=1, label='Residuals')
        plt.xlim(left=twoThetaMin)  
        plt.xlim(right=twoThetaMax)
        plt.legend()
        plt.ylabel('Intensity (arb.u.)')
        plt.xlabel('TwoTheta (deg)')
        plt.savefig(save_path + 'fits/' + 'frame_' + str(frame) + '.pdf', format='pdf')  
        plt.show()
        plt.clf()
                                   
    # Saves the fit as a dictionary
    resultsFit = {}
    #print('Following peak positions were calculated: \n' +str(tickArray))
    for i, phase in enumerate(phases['phases_used']):
        resultsFit['phase_'+str(phase)+'_unit_cell_fit'] = unitCell_fit(result, phase)
        print('Unit Cell ' + str(phase) + ' fit: \n' + str(unitCell_fit(result, phase)))
        resultsFit['phase_'+str(i)+'_unit_cell_error'] = np.array([estimatedError,
                                                                   estimatedError, 
                                                                   estimatedError,
                                                                   estimatedError,
                                                                   estimatedError,
                                                                   estimatedError])
        resultsFit['phase_'+str(phase)+'_V_fit'] = float(EoS.unitCellVolume([result.params['a_'+str(phase)].value, result.params['b_'+str(phase)].value, result.params['c_'+str(phase)].value,
                          result.params['alp_'+str(phase)].value,result.params['bet_'+str(phase)].value,result.params['gam_'+str(phase)].value]))
        resultsFit['phase_'+str(phase)+'_P_fit'] = float(EoS.BM3_EOS(resultsFit['phase_'+str(phase)+'_V_fit'], 
                                                        V0 = float(phases['phase_'+str(phase)+'_compression_constants'][0]),
                                                        K0 = float(phases['phase_'+str(phase)+'_compression_constants'][1]), 
                                                        K0P = float(phases['phase_'+str(phase)+'_compression_constants'][2])))
        resultsFit['phase_'+str(phase)+'_BG_fit'] = result.params['BG'].value
        resultsFit['phase_'+str(phase)+'_peakPosition_fit'] = tickArray[i]
        resultsFit['data_BGsub'] = np.array([twoTheta, valueIntBGsub])
        resultsFit['data_fit'] = np.array([twoTheta, bestFit])
        
    print('\n============================================= \n')

    return resultsFit



# LEGACY: plot_LB_fit commented out - not used by GUI
# def plot_LB_fit(data_path, phases, poni_file, save_path,
#                 framePressureGuess, phase_rules = (lambda f: f > 0, [0]),
#                 collectionTime = 1, ylabel = 'Frame',
#                 frameStart = 0, frameEnd = 1, frameStep = 1, 
#                 twoThetaMin = 8, twoThetaMax = 18,
#                 plotLB=False, logPlot = True, CorrNegativeValues = 0.02,
#                 prominence = 0.10, height = 0.08, order = 1, excludePeakList=[],
#                 ampGuess = 1, sigGuess = 0.05,
#                 method = 'pressureWalk', deltaP = 0.5,
#                 offsetStep = 0.05):
#     '''
#     This function creates a plot of the raw data, and the respective fitted data points with a le Bail/Pawley fit. 

#     '''
#     # Initialises plot the data and the fit
#     fig = plt.figure(constrained_layout=True, figsize=(8, 8))
#     spec = fig.add_gridspec(3, 4)

#     # Prints the batch view 
#     ax1 = fig.add_subplot(spec[0:1, :2])   
#     XRD_patterns={'frame':[], 'twoTheta':[], 'valueInt':[]}
#     for i, frame in enumerate(np.arange(frameStart,frameEnd+1,frameStep)):
#         #print(frame, data_path, twoThetaMin, twoThetaMax)
#         twoTheta, valueIntBGsub, errorInt = load_frame_fast(frame, data_path, twoThetaMin, twoThetaMax)
#         timeStamp = [frame*collectionTime]
#         XRD_patterns["frame"].append(timeStamp)

#         # Comment this block out if you don't want a log plot!
#         if logPlot == True:
#             #CorrNegativeValues = 0.02
#             for i in range(len(valueIntBGsub)):  
#                 valueIntBGsub[i] = math.log(valueIntBGsub[i]+CorrNegativeValues)

#         XRD_patterns["twoTheta"].append(twoTheta)
#         XRD_patterns["valueInt"].append(valueIntBGsub)	


#     im = ax1.pcolormesh(XRD_patterns['twoTheta'], 
#                     XRD_patterns['frame'], 
#                     XRD_patterns['valueInt'],
#                     #norm=LogNorm(vmin=1.2e-2, vmax=1e-1),  # Adjust range
#                     #edgecolors='w',
#                     cmap="binary_r")

#     cbar = fig.colorbar(im, ax=ax1)#, 'l')
#     cbar.set_label('Intensity (counts)')    
#     ax1.set(ylabel=ylabel)
#     #ax1.set(xlabel='Two Theta (deg)')
#     ax1.set_aspect('auto')
#     ax1.set_xlim((twoThetaMin, twoThetaMax))
#     ax1.set_ylim((frameStart*collectionTime, frameEnd*collectionTime))

#     # Prints the refined patterns
#     ax2 = fig.add_subplot(spec[1:3, :2])
#     ax2.set(ylabel='Intensity (arb.u.)')
#     ax2.set(xlabel='Two Theta (deg)')
#     ax2.set_xlim((twoThetaMin, twoThetaMax))

#     offset = 0
#     # offsetStep = 0.05

#     # Creates a frame list to process:
#     frameList = []
#     for i in range(frameStart,frameEnd+1,frameStep):
#         frameList.append(i)  

#     # Loads the dictionary of phases and initialises lists for saving data
#     result_lists = {}
#     for i, phase in enumerate(phases['phases_Number_array']):
#         result_lists[f'P_list_{i}'] = []
#         result_lists[f't_list_{i}'] = []
#         result_lists[f'phase_list_{i}'] = [] 

#     # THIS IS THE MAIN LOOP! Loads every frame, and fits it.
#     for i, frame in enumerate(frameList):
#         # Specifies the phases for a frame
#         specify_phases(phases, phase_rules, frame)
#         print('For frame '+ str(frame) +' the phases ' + str(phases['phases_used']) + ' are used')

#         # Pressure Helper; Uses Pessure array to fit data. 
#         if method == 'pressureHelper':
#             for i, phase in enumerate(phases['phases_used']):
#                 # Resets the unit cell to the 0 GPa value
#                 update_unitCell(phases['phase_'+ str(phase) +'_unit_cell_0'], 
#                                 phases, phase)
#                 # Calculates the unit cell for a pressure guess
#                 update_phases(frame, phases, phase, framePressureGuess, 
#                               deltaP=0, resultsFit=[],
#                               resultsFitUC = [0,0,0,0,0,0], 
#                               method = 'pressureHelper')

#         # Fits the data, returns new fit
#         resultsFit = fit_LB(data_path, phases, poni_file, save_path, 
#                                frame, twoThetaMin, twoThetaMax, 
#                                framePressureGuess,
#                                prominence, height, order, ampGuess, sigGuess,
#                                excludePeakList=[], plotLB=False,
#                                )

#             # resultsFit['phase_'+str(phase)+'_unit_cell_fit']
#             # resultsFit['phase_'+str(i)+'_unit_cell_error']
#             # resultsFit['phase_'+str(phase)+'_V_fit']
#             # resultsFit['phase_'+str(phase)+'_P_fit']
#             # resultsFit['phase_'+str(phase)+'_BG_fit']
#             # resultsFit['phase_'+str(phase)+'_peakPosition_fit']
#             # resultsFit['data_BGsub']
#             # resultsFit['data_fit']

#         # Plots the fitted data
#         ax2.scatter(resultsFit['data_BGsub'][0], resultsFit['data_BGsub'][1]+offset, marker = '.', color = 'lightgrey')
#         ax2.plot(resultsFit['data_fit'][0], resultsFit['data_fit'][1]+offset, color = 'red')
#         offset = offset + offsetStep

#         # Creates Markers of the peak position on 2D plot
#         for k, phase in enumerate(phases['phases_used']): 
#             frames = [frame*collectionTime] * len(resultsFit['phase_'+str(phase)+'_peakPosition_fit'])
#             ax1.scatter(resultsFit['phase_'+str(phase)+'_peakPosition_fit'], frames, color=colours[phase], s=0.7)
#             #for j, peakPosition in enumerate(resultsFit['phase_'+str(phase)+'_peakPosition_fit']):

#         # Fill lists for each phase for saving data
#         for i, phase in enumerate(phases['phases_Number_array']):
#             # Only append if this phase exists in resultsFit (handle varying phases_used)
#             try:     
#                 result_lists[f'phase_list_{i}'].append(resultsFit[f'phase_{phase}_unit_cell_fit'])
#                 result_lists[f'P_list_{i}'].append(resultsFit[f'phase_{phase}_P_fit'])  
#                 result_lists[f't_list_{i}'].append(frame * collectionTime)
#             except KeyError:
#                 pass  # Phase data not present for this frame, just skip

#         # Sequential Fitting; uses the last fit as an input for the next iteration
#         if method == 'sequential':
#             for i, phase in enumerate(phases['phases_used']):
#                 update_phases(frame, phases, phase, framePressureGuess, 
#                                  deltaP, resultsFit, 
#                                  resultsFit['phase_'+str(phase)+'_unit_cell_fit'], 
#                                  method = 'sequential')

#         # Walking Pressures; creates a UC guess based on a small pressure increase   
#         if method == 'pressureWalk':
#             for i, phase in enumerate(phases['phases_used']):
#                 update_phases(frame, phases, phase, framePressureGuess, 
#                                  deltaP, resultsFit, 
#                                  resultsFit['phase_'+str(phase)+'_unit_cell_fit'], 
#                                  method = 'pressureWalk',)

#     plt.savefig(save_path + 'Fit.pdf', format='pdf')
#     plt.show() 

#     #print('\n\t It took', round(time.time()-start), 'seconds to run this script.') 

#     # After all frames processed, write files once per phase
#     for i, phase in enumerate(phases['phases_Number_array']):
#         with open(save_path + f'UnitCell_Fit_{phase}.txt', 'w') as f:
#             f.write('time \t Pressure (GPa) \t unit cell (Å)\n')
#             for t, P, UC in zip(result_lists[f't_list_{i}'],
#                                 result_lists[f'P_list_{i}'],
#                                 result_lists[f'phase_list_{i}']):
#                 uc_str = '\t'.join([f'{val:.6f}' for val in UC])
#                 f.write(f'{t:.3f}\t{P:.3f}\t{uc_str}\n')






# LEGACY: index_LB commented out - not used by GUI
# def index_LB(data_path, phases, poni_file, save_path, 
#            frame = 10 , twoThetaMin = 0, twoThetaMax = 30, plotIndex=True, 
#            framePressureGuess=[[0,0],[3000,0]],
#            prominence = 0.10, height = 0.08, order = 20, excludePeakList=[],
#            ampGuess = 1, sigGuess = 0.05):   
#     '''
#     Subtracts the BG and creates an initial model for fitting, with the possible peak positions
#     '''   

#     # Load the n-th frame (i.e. diffraction file)
#     twoTheta, valueInt, errorInt = load_frame(frame, data_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)    

#     # Calculates the BG
#     try: 
#         twoTheta, valueBG = backgroundFit(data_path, frame=frame, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax,
#                                              peakSearchAuto = True, plotBG=False, prominence=prominence, 
#                                              height = height, order=order, excludePeakList=excludePeakList)   
#     except:
#         twoTheta = twoTheta
#         valueBG = np.zeros_like(twoTheta)
#         print('No Background subtraction performed. Two theta range too small.')

#     # Subtracts the BG and retruns two np.arrays with BG subtracted data
#     twoTheta, _, _, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)  

#     # Plots Data
#     if plotIndex == True:
#         # Only run once
#         plt.figure()
#         plt.scatter(twoTheta, valueIntBGsub, marker='.', color='gray', label='Raw Data', s=3)


#     # Initialises the parameters for the fit, i.e. lattice parameters etc.
#     params = initialise_parameters_LB(phases, WL = load_poni(poni_file), ampGuess = max(valueIntBGsub),
#                                       twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax, 
#                                       sigGuess = 0.05)   

#     # Evaluate the individual components of the fitted model separately
#     guessFit, components = create_guessComponents(twoTheta, phases, params, 
#                                             twoThetaMin, twoThetaMax, 
#                                             printInputComponents = False)

#     # Plots the initial guess
#     if plotIndex == True:
#         plt.plot(twoTheta, guessFit, c='blue', ls='dotted', lw=1, label='Initial guess') 
#         tickArray = create_ticks_guess(phases, twoThetaMin, twoThetaMax, WL=load_poni(poni_file))
#         tickStep = max(valueIntBGsub)/10
#         for j in range(len(tickArray)):
#             for i in range(len(tickArray[j])):  # Only loop up to the number of reflections for that phase
#                 plt.scatter(tickArray[j][i], -tickStep - (j+2) * tickStep, 
#                             marker='|', linewidths=2, s=100, color=colours[j])    
#         plt.xlim(left=twoThetaMin)  
#         plt.xlim(right=twoThetaMax)
#         plt.legend()
#         plt.ylabel('Intensity (arb.u.)')
#         plt.xlabel('TwoTheta (deg)')
#         plt.show()















###############################
###   Single Peak Fitting   ###    
###############################
# LEGACY: Only used by single-peak fit (not in GUI)
# LEGACY: find_matching_HKL commented out - not used by GUI
# def find_matching_HKL(reflectionList, HKL_input):
#     """
#     Compares a list or tuple like [h, k, l] with the reflection list of form [[x, H, K, L], ...] 
#     Returns the matching x value(s) for the specified HKL triplet.
#     """
#     reflectionList = np.array(reflectionList, dtype=float)
#     HKL_input = np.array(HKL_input, dtype=float)

#     # Use np.isclose for float-safe comparison
#     matches = np.all(np.isclose(reflectionList[:, 1:], HKL_input), axis=1)
#     matched_values = reflectionList[matches, 0]

#     if matched_values.size > 0:
#         return matched_values[0] if len(matched_values) == 1 else matched_values
#     else:
#         print('No matching HKL found')
#         return None


# LEGACY: Single-peak fitting not wired to GUI
# LEGACY: initialise_parameters_single_peak commented out - not used by GUI
# def initialise_parameters_single_peak(twoThetaGuess, ampGuess = 1, sigGuess = 0.05):
#     '''
#     Defines the parameters for the fit. The params are a dictionary based on unit cell dimensions and HKL values.
#     '''
#     # Defining the various parameters
#     params = Parameters()

#     params.add('BG', value = 0, min= -.5, max=.5)
#     params['BG'].set(vary=False)

#     params.add('cen', value = twoThetaGuess)
#     params.add('amp', value = ampGuess, min= ampGuess*0.0, max= ampGuess*1.5)
#     params.add('sig', value = sigGuess, min= sigGuess*0.0, max= sigGuess*1.5)

#     print('Initialising Parameters')
#     return params


# Define the fitting function
# LEGACY: Single-peak fitting not wired to GUI
# LEGACY: single_peak_fit_Model commented out - not used by GUI
# def single_peak_fit_Model(params,x,y):#, WL):
#     '''
#     Defines a Peak shape for the Fit. Here it is a Gaussian
#     '''
#     y_fit = params['BG'].value

#     cen = params['cen'].value
#     amp = params['amp'].value
#     sig = params['sig'].value

#     y_fit += amp * np.exp(-0.5 * ((x-cen)/sig)**2) 

#     return (y_fit - y)


# LEGACY: fit_peak_single commented out - not used by GUI
# def fit_peak_single(data_path, phases, poni_file, save_path, 
#                     frame, twoThetaMin, twoThetaMax, framePressureGuess,
#                     phase_used = 0, HKL_input = [0,0,0], 
#                     prominence = 0.10, height= 0.08, order = 1, 
#                     ampGuess = 1, sigGuess = 0.05, 
#                     excludePeakList=[], plotFit=False):
#     '''
#     Subtracts the BG, and fits a single peak with a specific HKL
#     ''' 
#     # Load the n-th frame (i.e. diffraction file)
#     twoTheta, valueInt, errorInt = load_frame_fast(frame, data_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)    

#     # Calculates the BG
#     try: 
#         twoTheta, valueBG = backgroundFit(data_path, frame=frame, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax,
#                                              peakSearchAuto = True, plotBG=False, prominence=prominence, 
#                                              height = height, order=order, excludePeakList=excludePeakList)   
#     except:
#         twoTheta = twoTheta
#         valueBG = np.zeros_like(twoTheta)
#         print('No Background subtraction performed. Two theta range too small.')

#     # Subtracts the BG and retruns two np.arrays with BG subtracted data
#     twoTheta, _, _, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)

#     if HKL_input == [0,0,0]:
#         print('No HKL value specified')
#         pass
#     else:
#         HKL = phases['phase_'+str(phase_used)+'_HKL']  
#         dk_reflection_list, twoTheta_reflection_list = ([] for i in range(2))
#         dk_reflection_list, twoTheta_reflection_list = reflection_List(phases[f'phase_{phase_used}_unit_cell'], 
#                                                                        HKL, WL = load_poni(poni_file), 
#                                                                        twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax, 
#                                                                        crystal_system = phases[f'phase_{phase_used}_crystal_system'])

#         print(f'Reflections for unit cell {phase_used} are:')
#         print(twoTheta_reflection_list)
#         twoTheta_HKL = find_matching_HKL(twoTheta_reflection_list, HKL_input)

#         print('\nPeak fitted for the ' + str(HKL_input) + ' reflection.')

#     if plotFit == True:
#         plt.scatter(twoTheta, valueIntBGsub, marker='.', color='gray', s= 3)

#     # Initialises the parameters for the fit, i.e. lattice parameters etc.
#     params = initialise_parameters_single_peak(twoTheta_HKL, ampGuess = 1, sigGuess = 0.05)

#     try:
#         result = minimize(lambda params: single_peak_fit_Model(params, twoTheta, valueIntBGsub),
#                   params, method='least_squares')

#         print('')
#         print(f"Fitting success: {result.success}")
#         #print(f"Fitting message: {result.message}")
#     except:
#         print('Fitting error in frame ' + str(frame))# + '. Better Pressure guess needed.')


#     # Compute fitted Gaussian
#     BG = result.params['BG'].value
#     cen = result.params['cen'].value
#     amp = result.params['amp'].value
#     sig = result.params['sig'] .value  

#     cen_err = result.params['cen'].stderr
#     amp_err = result.params['amp'].stderr
#     sig_err = result.params['sig'].stderr

#     best_fit = amp * np.exp(-0.5 * ((twoTheta - cen) / sig)**2) + BG
#     best_fit_parameters = np.array([cen, amp, sig])
#     best_fit_errors = np.array([cen_err, amp_err, sig_err]) 

#     print('Fit parameters:\t' + str(best_fit_parameters))
#     print('Fit errors:\t' + str(best_fit_errors))

#     # Create subfolder (if it doesn't exist)
#     subfolder_path = save_path+"/fits_singlePeak"  # Adjust as needed
#     os.makedirs(subfolder_path, exist_ok=True)

#     # Saves Data of Fit
#     with open(save_path + 'fits_singlePeak/' + 'frame_' + str(frame) + '.txt', 'w') as f:
#         f.write('Frame' + str(frame) + ' \n' + '2Theta \t Int \t Fit \n')
#         np.savetxt(f, np.column_stack((twoTheta, valueIntBGsub, best_fit)), fmt='%.6f', delimiter='\t')

#     # Saves the figure 
#     if plotFit == True:
#         plt.plot(twoTheta, best_fit, c='red', ls='-', lw=1, label='Best fit')
#         plt.xlim(left=twoThetaMin)  
#         plt.xlim(right=twoThetaMax)
#         plt.legend()
#         plt.savefig(save_path + 'fits/' + 'frame_' + str(frame) + '.png', format='png')  
#         plt.show()
#         plt.clf()

#     # Saves the fit as a dictionary
#     resultsFit = {}
#     resultsFit['fitted_HKL'] = HKL_input
#     resultsFit['fitParameters'] = np.array([cen, amp, sig])
#     resultsFit['fitErrors'] = np.array([cen_err, amp_err, sig_err])
#     resultsFit['data_BGsub'] = np.array([twoTheta, valueInt])
#     resultsFit['data_fit'] = np.array([twoTheta, best_fit])
#     print('\n============================================= \n')
#     return resultsFit


# LEGACY: plot_peak_fit_single commented out - not used by GUI
# def plot_peak_fit_single(data_path, phases, poni_file, save_path, phase_rules = (lambda f: f > 0, [0]),
#                          twoThetaMin = 0, twoThetaMax = 30, framePressureGuess = 0,
#                          collectionTime = 1, ylabel = 'Frame',
#                          frameStart = 40, frameEnd = 90, frameStep = 1,
#                          phase_used = 0, HKL_input = [1,1,1], 
#                          plotFit = False, logPlot = True, CorrNegativeValues = 0.02,
#                          prominence = 0.10, height = 0.08, order = 1, excludePeakList=[],
#                          ampGuess = 1, sigGuess = 0.05, 
#                          method = 'pressureWalk', deltaP = 0.5,
#                          offsetStep = 0.05):
#     '''
#     Uses JCPDS files to guess peak positions and fits them independently. 
#     tb.fit_peak_all should in the future return a table of HKL, xc, amp,   
#     '''
#     # Initialises plot the data and the fit
#     fig = plt.figure(constrained_layout=True, figsize=(8, 8))
#     spec = fig.add_gridspec(3, 4)

#     # Prints the batch view 
#     ax1 = fig.add_subplot(spec[0:1, :2])   
#     XRD_patterns={'frame':[], 'twoTheta':[], 'valueInt':[]}
#     for i, frame in enumerate(np.arange(frameStart,frameEnd+1,frameStep)):
#         twoTheta, valueIntBGsub, errorInt = load_frame_fast(frame, data_path, twoThetaMin, twoThetaMax)
#         timeStamp = [frame*collectionTime]
#         XRD_patterns["frame"].append(timeStamp)

#         # Comment this block out if you don't want a log plot!
#         if logPlot == True:
#             #CorrNegativeValues = 0.02
#             for i in range(len(valueIntBGsub)):  
#                 valueIntBGsub[i] = math.log(valueIntBGsub[i]+CorrNegativeValues)

#         XRD_patterns["twoTheta"].append(twoTheta)
#         XRD_patterns["valueInt"].append(valueIntBGsub)	


#     im = ax1.pcolormesh(XRD_patterns['twoTheta'], 
#                     XRD_patterns['frame'], 
#                     XRD_patterns['valueInt'],
#                     #edgecolors='w',
#                     cmap="binary_r")

#     cbar = fig.colorbar(im, ax=ax1)#, 'l')
#     cbar.set_label('Intensity (counts)')    
#     ax1.set(ylabel='time (s)')
#     #ax1.set(xlabel='Two Theta (deg)')
#     ax1.set_aspect('auto')
#     ax1.set_xlim((twoThetaMin, twoThetaMax))
#     #ax1.set_ylim((0, 680*0.05))

#     # Prints the refined patterns
#     ax2 = fig.add_subplot(spec[1:3, :2])
#     ax2.set(ylabel='Intensity (arb.u.)')
#     ax2.set(xlabel='Two Theta (deg)')
#     ax2.set_xlim((twoThetaMin, twoThetaMax))

#     offset = 0
#     # offsetStep = 1

#     # Creates a frame list to process:
#     frameList = []

#     for i in range(frameStart, frameEnd+1, frameStep):
#         frameList.append(i)  

#     offset = 0
#     #offsetStep = 0.05

#     for i, frame in enumerate(frameList):
#         # Specifies the phases for a frame
#         print(frame)
#         specify_phases(phases, phase_rules, frame)
#         #print('For frame '+ str(frame) +' the phases ' + str(phases['phases_used']) + ' are used')

#         # Pressure Helper; Uses Pessure array to fit data. 
#         if method == 'pressureHelper':
#             for i, phase in enumerate(phases['phases_used']):
#                 # Resets the unit cell to the 0 GPa value
#                 update_unitCell(phases['phase_'+ str(phase) +'_unit_cell_0'], 
#                                 phases, phase)
#                 # Calculates the unit cell for a pressure guess
#                 update_phases(frame, phases, phase, framePressureGuess, 
#                                  deltaP=0, resultsFit=[],
#                                  resultsFitUC = [0,0,0,0,0,0], 
#                                  method = 'pressureHelper')

#         # Fitting Function
#         resultsFit = fit_peak_single(data_path, phases, poni_file, save_path, 
#                                         frame, twoThetaMin, twoThetaMax, framePressureGuess,
#                                         phase_used, HKL_input, 
#                                         prominence, height, order, 
#                                         ampGuess, sigGuess, 
#                                         excludePeakList=[], plotFit=plotFit,
#                                         )

#                         # resultsFit['fitted_HKL'] = np.array([H,K,L])
#                         # resultsFit['fitParameters'] = np.array([cen, amp, sig])
#                         # resultsFit['fitErrors'] = np.array([cen_err, amp_err, sig_err])
#                         # resultsFit['data_BGsub'] = np.array([twoTheta, valueInt])
#                         # resultsFit['data_fit'] = np.array([twoTheta, best_fit])  

#         # Plots the fitted data
#         ax1.scatter(resultsFit['fitParameters'][0], frame*collectionTime, color='red', marker='.', 
#                     s=0.7, label = resultsFit['fitted_HKL'])

#         # Sequential Fitting; uses the last fit as an input for the next iteration
#         if method == 'sequential':
#             for i, phase in enumerate(phases['phases_used']):
#                 update_phases(frame, phases, phase, framePressureGuess, resultsFit,
#                                   resultsFit['phase_'+str(phase)+'_unit_cell_fit'], 
#                                   method = 'sequential', deltaP = deltaP)

#         # Walking Pressures; creates a UC guess based on a small pressure increase   
#         if method == 'pressureWalk':
#             for i, phase in enumerate(phases['phases_used']):
#                 update_phases(frame, phases, phase, framePressureGuess, resultsFit,
#                                   resultsFit['phase_'+str(phase)+'_unit_cell_fit'], 
#                                   method = 'pressureWalk', deltaP = deltaP)

#         offset = offset + offsetStep
#         ax2.scatter(resultsFit['data_BGsub'][0], resultsFit['data_BGsub'][1]+offset, marker = '.', color = 'lightgrey')
#         ax2.plot(resultsFit['data_fit'][0], resultsFit['data_fit'][1]+offset, color = 'red')
#         offset = offset + offsetStep


#     plt.savefig(save_path + 'singleFit.pdf', format='pdf')
#     plt.show() 

#     #print('\n\t It took', round(time.time()-start), 'seconds to run this script.') 

