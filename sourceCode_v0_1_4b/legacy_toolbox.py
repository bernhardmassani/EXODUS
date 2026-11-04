# legacy_toolbox.py for EXODUS v0.1.4
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
# Retired code of earlier EXODUS versions, kept for reference only. Every
# line is commented out and nothing imports this file. The code refers to
# the old module names (BatchFit_toolbox as tb, EoS_toolbox as EoS) and
# would need updating before it could be used again.
#
# ============================================================================


# ============================================================================
#                  EQUATIONS OF STATE (old EoS_toolbox.py)
# ============================================================================
###############
## P vs time ##
###############

# LEGACY: Not called from GUI
# LEGACY: plot_pressure commented out - not used by GUI
# def plot_pressure(t_list, P_list, save_path, collectionTime = 1.5):
#     plt.figure(figsize=(6, 4))  # You can adjust size as needed
#     plt.plot(t_list, P_list, 'o', markersize=4, color='black', label='Pressure Fit')
#     plt.ylabel('Pressure (GPa)')
#     plt.xlabel('Time (ms)')
#     plt.axvline(x=49*collectionTime, color='skyblue', linestyle='--', linewidth=1, label='Ice VI')
#     plt.axvline(x=50*collectionTime, color='blue', linestyle='--', linewidth=1, label='Ice VII')
#     #plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(save_path + 't_vs_P.pdf', format='pdf')
#     plt.show()

# LEGACY: Superseded by find_volume_at_pressure() with expanding bounds
# LEGACY: find_volume_at_pressure_simple commented out - not used by GUI
# def find_volume_at_pressure_simple(P, unit_cell, K0, K0P):
#     """
#     Find volume at a given pressure using the third-order Birch-Murnaghan EOS based on unit_cell and JCPDS input
#     Returns: Array [a, b, c, alpha, beta, gamma, V] specifying the lattice constants and volume at the given pressure
#     """
#     # Extract lattice constants
#     #a0, b0, c0, alpha0, beta0, gamma0 = unit_cell[0]

#     # Calculate the equilibrium volume
#     V0 = unitCellVolume(unit_cell)

#     # Define a function to find the root of (BM3_EOS - P)
#     def volume_difference(V):
#         return BM3_EOS(V, V0, K0, K0P) - P

#     print(f"volume_difference at V0*0.5: {volume_difference(V0 * 0.5)}")
#     print(f"volume_difference at V0*1.5: {volume_difference(V0 * 1.5)}")

#     # Use brentq to find the root (volume)
#     #V = brentq(volume_difference, V0 * 0.5, V0 * 1.5)
#     V = brentq(volume_difference, V0 * 0.01, V0 * 5.0)
#     return V

# LEGACY: Only used by find_UC_at_0GPa (also unused)
# LEGACY: invert_BM3 commented out - not used by GUI
# def invert_BM3(VP, P, K0, K0P):
#     """
#     Invert Birch-Murnaghan EOS to recover V0 from VP and pressure.
#     """
#     # Make sure VP and P are floats, not arrays
#     VP = float(np.squeeze(VP))
#     P = float(P)

#     def diff(V0):
#         return BM3_EOS(VP, V0, K0, K0P) - P

#     try:
#         V0_recovered = brentq(diff, VP * 1.0, VP * 2.5)
#         return V0_recovered
#     except Exception as e:
#         print(f"Inversion failed: {e}")
#         return None


# LEGACY: Not called from GUI or BatchFit_toolbox
# LEGACY: find_UC_at_0GPa commented out - not used by GUI
# def find_UC_at_0GPa(unit_cell_P, P, K0, K0P):
#     """
#     Take a high-pressure unit cell and decompress it to 0 GPa.
#     """
#     VP = unitCellVolume(unit_cell_P)
#     V0 = invert_BM3(VP, P, K0, K0P)

#     if V0 is None:
#         print("Could not invert EOS to get V0.")
#         return None

#     scale = (V0 / VP) ** (1/3)
#     return [
#         unit_cell_P[0] * scale,
#         unit_cell_P[1] * scale,
#         unit_cell_P[2] * scale,
#         unit_cell_P[3],
#         unit_cell_P[4],
#         unit_cell_P[5]
#     ]


# LEGACY: specify_transitionPressure commented out - not used by GUI
# def specify_transitionPressure(phases, phase, P_trans = 0):
#     '''
#     Sets a unit cell from JCPDS to the values at which this phase first appears
#     '''
#     unitCell_start = find_UC_at_P(phases[f'phase_{phase}_unit_cell_0'],
#                                   P = P_trans,
#                                   V0 = phases['phase_'+str(phase)+'_compression_constants'][0],
#                                   K0 = phases['phase_'+str(phase)+'_compression_constants'][1],
#                                   K0P = phases['phase_'+str(phase)+'_compression_constants'][2]
#                                   )
#     phases[f'phase_{phase}_unit_cell'] = unitCell_start
#     return unitCell_start

# LEGACY: Unused EOS (also had math.ln bug, now fixed to math.log)
# LEGACY: simple_EOS commented out - not used by GUI
# def simple_EOS(V, V0, K0):
#     """
#     Calculate pressure (P) using the simplest EOS.
#     """
#     P = K0 * math.log(V0/V)
#     return P

# LEGACY: Unused EOS (also had ^ bug, now fixed to **)
# LEGACY: Murnaghan_EOS commented out - not used by GUI
# def Murnaghan_EOS(V, V0, K0, K0P):
#     """
#     Calculate pressure (P) using the Murnaghan EOS.
#     """
#     P = K0/K0P * ((V0/V)**K0P - 1)
#     return P


# LEGACY: Unused - second-order BM EOS
# LEGACY: BM2_EOS commented out - not used by GUI
# def BM2_EOS(V, V0, K0):
#     """
#     Calculate pressure (P) using the second-order Birch-Murnaghan EOS.
#     """
#     P = (3/2) * K0 * ((V0 / V)**(7/3) - (V0 / V)**(5/3))
#     return P

# def BM3_EOS(V, V0, K0, K0P):
#     """
#     Calculate pressure (P) using the third-order Birch-Murnaghan EOS.
#     """
#     P = (3/2) * K0 * ((V0 / V)**(7/3) - (V0 / V)**(5/3)) * (1 + (3/4) * (K0P - 4) * ((V0 / V)**(2/3) - 1))
#     return P

# def calculate_KT(P, K0, K0P):
#     """
#     Calculate isothermal bulk modulus (KT) for a given P using a linear model
#     """
#     KT = K0 + K0P * P
#     return KT
# LEGACY: Unused - noted as possibly incorrect
# LEGACY: calculate_KT_BM2 commented out - not used by GUI
# def calculate_KT_BM2(V, V0, K0, K0P):
#     """
#     Calculate isothermal bulk modulus (KT) for a given volume (V) using a second-order BM EOS.
#     Does not work - mayby physics is wrong?
#     """
#     # Calculate KT using the corrected definition
#     KT = -V * (3 / 2) * K0 * ((7 / 3) * (V0 / V)**(4 / 3) - (5 / 3) * (V0 / V)**(2 / 3)) * (V0 / V**2)
#     return KT


# LEGACY: Unused - noted as possibly incorrect
# LEGACY: calculate_KT_BM3 commented out - not used by GUI
# def calculate_KT_BM3(V, V0, K0, K0P):
#     """
#     Calculate isothermal bulk modulus (KT) for a given volume (V) using a third-order BM EOS.
#     Does not work - mayby physics is wrong?
#     """
#     # Calculate KT using the corrected definition
#     dPdV = (3 / 2) * K0 * ((7 / 3) * (V0 / V)**(4 / 3) - (5 / 3) * (V0 / V)**(2 / 3)) * (1 + (3 / 4) * (K0P - 4) * ((V0 / V)**(2 / 3) - 1) - (3 / 4) * (K0P - 4) * (2 / 3) * (V0 / V)**(-1 / 3))
#     KT = -V * dPdV
#     return KT


################################
###     Thermal Pressure     ###
################################

# LEGACY: Unused thermal pressure function
# LEGACY: calculate_thermal_pressure commented out - not used by GUI
# def calculate_thermal_pressure(P, deltaT, alphaT, V0, K0, K0P):
#     percent = 0.3
#     P_thermal = percent * deltaT * alphaT * calculate_KT(P, K0, K0P)
#     return P_thermal

# LEGACY: Unused thermal pressure function
# LEGACY: calculate_thermal_pressure_constantKT commented out - not used by GUI
# def calculate_thermal_pressure_constantKT(P, deltaT, alphaT, V0, K0, K0P):
#     '''
#     Assumes a constant alpha*KT according to Dewaele - probably incorrect.
#     '''
#     P_thermal = deltaT * alphaT * K0
#     return P_thermal

# LEGACY: Unused - also calls tb.load_JCPDS with old API
# LEGACY: calculate_isotherms commented out - not used by GUI
# def calculate_isotherms(V0, V_End, jcpds):
#     '''
#     Takes two volumina and calculates the isotherm between them
#     '''
#     crystal_system, a, b, c, alpha, beta, gamma, V0 , K0, K0P, alphaT, DK0DT, DK0PDT, HKL, WL = tb.load_JCPDS(jcpds)

#     volume_lst = np.arange(V_End,V0,.01) # [60,58,54,53,52,45,42]
#     deltaT1 = 2000
#     for i, volume in enumerate(volume_lst):
#         P = BM3_EOS(volume, V0, K0, K0P)
#         V =volume
#         P_melt = P + calculate_thermal_pressure_constantKT(P, deltaT1, alphaT, V0, K0, K0P )
#         plt.plot(P, V, marker='.', color='b')
#         plt.plot(P_melt, V, marker='.', color='r')

#     #plt.xlim((30,40))
#     #plt.ylim((43.5,47))
#     plt.xlabel('Pressure (GPa)')
#     plt.ylabel(r'Volume ($\AA ^3$)')
#     plt.show()



# LEGACY: Stub - returns unit_cell unchanged
# LEGACY: calculate_unitCell_BM3 commented out - not used by GUI
# def calculate_unitCell_BM3(P, unit_cell, V0, K0, K0P):
#     """
#     Calculates the unit cell parameters as a np.array [a,b,c,alpha,beta,gamma] based on a pressure
#     """
#     # Calculate KT using the corrected definition

#     return unit_cell


# ============================================================================
#            PHASE / PRESSURE HANDLING (old BatchFit_toolbox.py)
# ============================================================================




# LEGACY: update_phases_used commented out - not used by GUI
# def update_phases_used(phases_used, phases):
#     '''
#     Replaces the phases used in the phaes dictionary
#     '''
#     phases['phases_used'] = np.array(phases_used)


# def specify_phases(phases, phase_rules, frame):
#     '''
#     Used to specify which phases are loaded for a given frame.
#     '''
#     for condition, phase_list in phase_rules:
#         if condition(frame):
#             phases['phases_used'] = phase_list
#             break
#     else:
#         raise ValueError(f"Unexpected frame number: {frame}")
#
#
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
#
#     return phases

# def update_unitCell(newUnitCell, phases, phaseNumber):
#     '''
#     Takes a new unit cell and replaces the values in the phaes dictionary
#     '''
#     newUnitCell = np.array(newUnitCell)     # makes sure it is a np.array
#
#     key = 'phase_' + str(phaseNumber) + '_unit_cell'
#     if key in phases:
#         phases[key] = newUnitCell
#     else:
#         print('Error. No matching unit cell to update.')
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

# def pressure_interpolation(framePressureGuess, plotPressureGuess = False):
#     '''
#     Takes Pressure points as an array [[frame, Pressure],[],...] and intrapolates the pressures in between
#     '''
    # Extract x and y values from framePressure
#     frame = np.array([point[0] for point in framePressureGuess])
#     P = np.array([point[1] for point in framePressureGuess])
#
#
#     framePressureGuess = np.array(framePressureGuess)
    # Create an interpolation function
#     interp_func = interp1d(frame, P, kind='linear', fill_value="extrapolate")
#
    # Generate new x values for interpolation
#     frame_interp = np.linspace(int(framePressureGuess[0, 0]), int(framePressureGuess[-1, 0]), int(framePressureGuess[-1, 0])) #Start, stop, samples
#     frame_interp = np.array(frame_interp)
#
    # Perform interpolation
#     P_interp = interp_func(frame_interp)
#     P_interp = np.array(P_interp)
#
    # Plot the original data and interpolated values
#     if plotPressureGuess == True:
#         plt.plot(frame, P, 'o', label='Original Data')
#         plt.plot(frame_interp, P_interp, '-', label='Interpolated Data')
#         plt.legend()
#         plt.show()
#
#     return frame_interp, P_interp


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

# def update_phases(frame, phases, phase, framePressureGuess,
#                   deltaP, resultsFit,
#                   resultsFitUC = [0,0,0,0,0,0],
#                   method = 'pressureHelper'):
#
#     if method == 'sequential':
#         update_unitCell(resultsFitUC, phases, phase)
#         print('Unit Cell ' + str(phase) + ' guess: \n' + str(resultsFitUC))
#         return resultsFitUC
#
#     elif method == 'pressureHelper':
        # Get fitted unit cell and calculate volume
#         unitCell_fit = phases['phase_'+str(phase)+'_unit_cell']
#
        # Get EoS parameters
#         unitCell_0 = phases[f'phase_{phase}_unit_cell_0']
#         V0 = float(phases[f'phase_{phase}_compression_constants'][0])
#         K0 = float(phases[f'phase_{phase}_compression_constants'][1])
#         K0P = float(phases[f'phase_{phase}_compression_constants'][2])
#
#         P_fit = EoS.BM3_EOS(EoS.unitCellVolume(phases['phase_'+str(phase)+'_unit_cell']),
#                                                V0, K0, K0P)
#
        # Creates an interpolation for the pressures and therefore the unitcell volume
#         frame_interp, P_interp = pressure_interpolation(framePressureGuess, plotPressureGuess=False)
#
        # Scales UC based on last UC - more reliable than doing it from UC0
#         unitCell_guess = EoS.scale_UC_at_P(unitCell_fit, P_interp[frame],
#                                            unitCell_0, K0, K0P)
        # Scales UC based on last UC0
        # unitCell_new = unit_cell_P_guess(phases['phase_'+str(phase)+'_unit_cell'],
        #                                          frame, framePressureGuess, phases, phase)
#
#         print('Unit Cell ' + str(phase) + ' guess: \n' + str(unitCell_guess))
#
#         update_unitCell(unitCell_guess, phases, phase)
#         return unitCell_guess
#
#     elif method == 'pressureWalk':
        # Get fitted unit cell and calculate volume
#         unitCell_fit = resultsFit[f'phase_{phase}_unit_cell_fit']
#
        # Get EoS parameters
#         unitCell_0 = phases[f'phase_{phase}_unit_cell_0']
#         V0 = float(phases[f'phase_{phase}_compression_constants'][0])
#         K0 = float(phases[f'phase_{phase}_compression_constants'][1])
#         K0P = float(phases[f'phase_{phase}_compression_constants'][2])
#
        # Calculate current pressure using BM3
#         P_fit = EoS.BM3_EOS(EoS.unitCellVolume(resultsFit['phase_'+ str(phase) +'_unit_cell_fit']),
#                                                V0, K0, K0P)
        #print(f"  Phase {phase}: P_fit = {P_fit:.2f} GPa")
#
        #UC_fits[phase].append(UC_fit)
#
        # Predict next frame's Pressure
#         P_next = P_fit + deltaP
        #print('Pressure of the fit is' + str(P_fit) + ' and pressure of the next frame is ' + str(P_next))
#
        # Predict next frame's unit cell
#         unitCell_next = EoS.scale_UC_at_P(unitCell_fit, P_next,
#                                           unitCell_0, K0, K0P)
        # unitCell_next = EoS.find_UC_at_P(unitCell_0,
        #                        P_next, V0, K0, K0P)
#
        #print('Nex UC prediction for phase '+ str(phase) +' from pressure walk:')
        #print(unitCell_next)
#
        # Update unit cell for next frame
#         phases[f'phase_{phase}_unit_cell'] = unitCell_next
#
#         print('Unit Cell ' + str(phase) + ' guess: \n' + str(unitCell_next))
#         return unitCell_next

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


# ============================================================================
#                   FILE LOADING (old BatchFit_toolbox.py)
# ============================================================================
# def extract_number(filename):
#     """Extracts the numeric part from filenames like 'run1_93' safely.
#
#     Returns float('inf') for any filename that doesn't end in an integer
#     suffix (e.g. 'metadata.txt', '.xy', or empty); these get sorted to
#     the end and then filtered out by load_frame.
#     """
#     base = os.path.basename(filename)
#     parts = base.split("_")
#
#     try:
#         return int(parts[-1].split(".")[0])  # Extract number before any extension
#     except (ValueError, IndexError, AttributeError):
#         return float('inf')  # If no valid number, send to the end

# LEGACY: Not called from GUI - use len(file_paths) instead
# LEGACY: load_frameNumber commented out - not used by GUI
# def load_frameNumber(data_path):
#     # Get all files in the folder and filter out non-numeric ones
#     files = sorted(glob.glob(os.path.join(data_path, "*")), key=extract_number)
#     return len(files)

# def load_frame(frame, data_path, twoThetaMin = 0, twoThetaMax = 30):
    # Get all files in the folder and filter out non-numeric ones
#     files = sorted(glob.glob(os.path.join(data_path, "*")), key=extract_number)
#     if files:
#         print(f"Found {len(files)} files in the directory.")# + str(data_path))
        #num_files = len(glob.glob(data_path + "/*"))  # Counts all files
        #print(f"Number of files: {num_files}")
#     else:
#         print("No data files found in the directory!")
#
    # Remove files where extract_number() returned 'inf' (non-numeric suffix)
#     files = [f for f in files if extract_number(f) != float('inf')]
#
#     total_files = len(files)
#
#     if total_files == 0:
#         print("No valid numbered files found!")
#         return None
#
#     if 1 <= frame <= total_files:
#         file_path = files[frame - 1]  # Convert to 0-based index
        #print(f"Loading file {frame}/{total_files}: {file_path}")
        # Open and read file content
#         twoTheta, valueInt, errorInt = loadData(file_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)
        #print("File loaded successfully!")
#
#         return twoTheta, valueInt, errorInt
#     else:
#         print(f"Error: Frame {frame} is out of range. Total files: {total_files}")
#         return None

# def load_frame_fast(frame, data_path, twoThetaMin = 0, twoThetaMax = 30):
    # Get all files in the folder and filter out non-numeric ones
#     files = sorted(glob.glob(os.path.join(data_path, "*")), key=extract_number)
#
    # Remove files where extract_number() returned 'inf' (non-numeric suffix)
#     files = [f for f in files if extract_number(f) != float('inf')]
#     total_files = len(files)
#
#     if total_files == 0:
#         print("No valid numbered files found!")
#         return None
#
#     if 1 <= frame <= total_files:
#         file_path = files[frame - 1]  # Convert to 0-based index
        # Open and read file content
#         twoTheta, valueInt, errorInt = loadData(file_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)
#         return twoTheta, valueInt, errorInt
#     else:
#         print(f"Error: Frame {frame} is out of range. Total files: {total_files}")
#         return None

# LEGACY: Formula noted as wrong in docstring - do not use
# LEGACY: twoTheta_to_q commented out - not used by GUI
# def twoTheta_to_q(twoTheta, WL):
#     '''
#     Input twoTheta (deg); output Q; Wrong Formula - fix before use
#     '''
#     return  4 * math.pi / WL *math.sin(twoTheta/2)



# ============================================================================
#        FILE-BASED BACKGROUND WRAPPERS (old BatchFit / EXODUS_core)
# ============================================================================
# def backgroundFit(data_path, twoThetaMin = 0, twoThetaMax = 30,
#                   peakSearchAuto = True, prominence=0.10, height = 0.08,
#                   order=6, excludePeakList=[], peakwidth = 0.35,
#                   frame=None, plotBG=False):
#     '''
#     Finds peaks based on prominence=0.1 and height = 0.3 - these values can be varied
#     Fits a polynomial of i-th order (standardwise 6th order) to the background
#         Returns original data and the respective BG-polynomial. When called a BG-subtraced array must be created
#     '''
#     twoTheta, valueInt, errorInt = loadData(
#         data_path, twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax)
#     return backgroundFit_arrays(
#         twoTheta, valueInt, errorInt,
#         peakSearchAuto=peakSearchAuto, prominence=prominence, height=height,
#         order=order, excludePeakList=excludePeakList, peakwidth=peakwidth)

# def backgroundFit_ALS(data_path, twoThetaMin=0, twoThetaMax=30,
#                       lam=1e5, p=0.01, niter=10):
#     """
#     Asymmetric Least Squares background subtraction
#     lam : float,Smoothness parameter (10^4 – 10^7 typical)
#     p : float, Asymmetry parameter (0.001 – 0.1 typical)
#     niter : int, Iterations (10–20 typical)
#     """
#
    # Load data
#     twoTheta, valueInt, errorInt = loadData(data_path,
#                                             twoThetaMin=twoThetaMin,
#                                             twoThetaMax=twoThetaMax)
#     return backgroundFit_ALS_arrays(twoTheta, valueInt, errorInt,
#                                     lam=lam, p=p, niter=niter)

# def run_background_fit(data_path, twoThetaMin, twoThetaMax,
#                        prominence, height, order, excludePeakList,
#                        peakwidth = 0.35):
#
#     twoTheta, valueInt, errorInt, valueBG, peaks = tb.backgroundFit(
#         data_path,
#         twoThetaMin=twoThetaMin,
#         twoThetaMax=twoThetaMax,
#         peakSearchAuto=True,
#         prominence=prominence,
#         height=height,
#         order=order,
#         excludePeakList=excludePeakList,
#         peakwidth = 0.35)
#
#     twoTheta, valueInt, valueBG, valueIntBGsub = tb.subtract_BG(twoTheta,
#                                                              valueInt, valueBG)
#
#     return twoTheta, valueInt, errorInt, valueBG, valueIntBGsub, peaks

# def run_backgroundFit_ALS(data_path, twoThetaMin, twoThetaMax,
#                       lam, p, niter):
#
#     twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = tb.backgroundFit_ALS(
#                                                           data_path,
#                                                           twoThetaMin,
#                                                           twoThetaMax,
#                                                           lam, p, niter)
#
    # twoTheta, valueInt, valueBG, valueIntBGsub = tb.subtract_BG(twoTheta,
    #                                                          valueInt, valueBG)
#
#     return twoTheta, valueInt, errorInt, valueBG, valueIntBGsub

# def load_WL(poni_file):
#     WL = tb.load_poni(poni_file)
#     return WL

# def load_JCPDS(JCPDS):
#     phases = tb.load_JCPDS(JCPDS)
#     return phases


# ============================================================================
#             LE BAIL / PAWLEY HELPERS (old BatchFit_toolbox.py)
# ============================================================================
# def _LB_phase_stencil(HKL, crystal_system):
#     """Return float64 H, K, L columns plus a fast-path discriminator."""
#     key = (id(HKL), crystal_system)
#     cached = _LB_geom_cache.get(key)
#     if cached is not None:
#         return cached
#     HKL_arr = np.asarray(HKL)
#     H = HKL_arr[:, 0].astype(np.float64, copy=False)
#     K = HKL_arr[:, 1].astype(np.float64, copy=False)
#     L = HKL_arr[:, 2].astype(np.float64, copy=False)
#     if crystal_system == 'CUBIC':
        # d = a / sqrt(H^2+K^2+L^2)
#         denom = H*H + K*K + L*L
#         cached = ('CUBIC', H, K, L, denom)
#     elif crystal_system in ('TETRAGONAL', 'HEXAGONAL'):
#         if crystal_system == 'HEXAGONAL':
#             ab_term = (4.0/3.0) * (H*H + H*K + K*K)
#         else:
#             ab_term = H*H + K*K
#         c_term = L*L
#         cached = (crystal_system, H, K, L, ab_term, c_term)
#     else:
        # Other systems use the slower path through reflection_List.
#         cached = (crystal_system, H, K, L)
#     _LB_geom_cache[key] = cached
#     return cached

# def _LB_clear_geom_cache():
#     """Empty the per-phase HKL stencil cache. Call when phases dict
#     changes (e.g. new JCPDS loaded) so stale stencils aren't reused."""
#     _LB_geom_cache.clear()

# def get_sigma(params, sigma_mode, phase, j, cen, deg_to_rad):
    # Matching sigma usage in the Model function
#     if sigma_mode == "fixed":
#         return params['sig_global'].value
#
#     elif sigma_mode == "per_phase":
#         return params[f'sig_{phase}'].value
#
#     elif sigma_mode == "caglioti":
#         theta = cen / 2 * deg_to_rad
#         sigma0 = params['sig_global'].value
#         return sigma0 / np.cos(theta)
        #return sigma0 *(1 + 25* np.tan(theta)**2)
#
#     elif sigma_mode == "separate":
        # Guard: if a reflection shifted into the window during fitting and j exceeds
        # the number of sigma params created at initialisation, reuse the last one.
#         key = f'sig_{phase}_{j+1}'
#         if key not in params:
            # Find the highest created index for this phase
#             existing = [k for k in params if k.startswith(f'sig_{phase}_')]
#             if existing:
#                 key = sorted(existing, key=lambda k: int(k.rsplit('_',1)[-1]))[-1]
#             else:
#                 raise KeyError(f"No sigma parameters found for phase {phase}")
#         return params[key].value
#
#     else:
#         raise ValueError(f"Unknown sigma_mode: {sigma_mode}")

# def create_ticks_guess(phases, twoThetaMin, twoThetaMax, WL):
#     '''
#     Returns a list of list of 2theta reflections arrays for each phase.
#     '''
#     tickArray = []
#     for phase in phases['phases_used']:
#         _, twoTheta_reflection_list = reflection_List(phases[f'phase_{phase}_unit_cell'],
#                                                       phases['phase_' + str(phase) + '_HKL'],
#                                                       WL,
#                                                       twoThetaMin=twoThetaMin,
#                                                       twoThetaMax=twoThetaMax,
#                                                       crystal_system=phases['phase_' + str(phase) + '_crystal_system']
#                                                       )
        # Take only the first column (2Theta values) and append as a separate list
#         tickArray.append(twoTheta_reflection_list[:, 0].tolist())
#     return tickArray  # list of lists

# def create_guessComponents(twoTheta, phases, params, twoThetaMin, twoThetaMax,
#                            sigma_mode = 'caglioti', printInputComponents = False):
#     '''
#     Creates the components for the overall envelope function and returns the envelope.
#
#     Amplitudes and per-peak sigmas are keyed by the index of the reflection
#     in the FULL HKL array (the same scheme used by initialise_parameters_LB
#     and LB_fit_Model). Previously this looped `for j, _ in enumerate(...)`
#     over the WINDOW-FILTERED reflection list and looked up `amp_<phase>_<j+1>`
#     -- which only happens to coincide with the HKL index when every HKL row
#     falls inside the fit window. As soon as any row was filtered out, the
#     displayed/saved per-peak components used the wrong amplitudes.
#     '''
#     components = {}
#     components['BG'] = params['BG'].value  # always present even if no phases
#     for i, phase in enumerate(phases['phases_used']):
#         HKL = phases['phase_'+str(phase)+'_HKL']
#         dk_reflection_list, twoTheta_reflection_list = reflection_List(
#               [params['a_'+str(phase)].value, params['b_'+str(phase)].value, params['c_'+str(phase)].value,
#                params['alp_'+str(phase)].value, params['bet_'+str(phase)].value, params['gam_'+str(phase)].value],
#               HKL, WL=params['WL'].value, crystal_system=phases['phase_'+str(phase)+'_crystal_system'],
#               twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax
#               )
        # Build HKL-triplet -> full-array-index lookup so we can recover
        # the original HKL index for each window-passing reflection.
#         HKL_full = np.asarray(HKL)
#         hkl_keys = {(int(round(h)), int(round(k)), int(round(l))): idx
#                     for idx, (h, k, l) in enumerate(HKL_full[:, :3])}
#         for j, dk_reflection in enumerate(dk_reflection_list):
#             cen = twoTheta_reflection_list[j][0]
#             row = twoTheta_reflection_list[j]
#             hkl_idx = hkl_keys.get(
#                 (int(round(row[1])), int(round(row[2])), int(round(row[3]))), j)
#             amp = params['amp_'+str(phase) + '_' + str(hkl_idx+1)].value
#             sig = get_sigma(params, sigma_mode, phase, hkl_idx, cen, deg_to_rad)
#             if printInputComponents == True:
#                 print('xc '+ str(cen) + '\tamp ' + str(amp) + '\tsig ' + str(sig))
#             components['comp_'+str(phase)+'_'+str(hkl_idx+1)] = amp * np.exp(-0.5 * ((twoTheta-cen)/sig)**2)
    # Sum outside the loop so bestFit always exists (even for zero phases)
#     bestFit = sum(components.values())
#     return bestFit, components

# def calculate_Rw(y_obs, y_calc, weights=None):
#     """
#     Weighted profile R-factor (Rwp), as defined in the Rietveld refinement
#     literature: sqrt( sum(w*(y_obs-y_calc)^2) / sum(w*y_obs^2) ).
#
#     With `weights=None` (the default) we use Poisson counting-statistics
#     weights w_i = 1/max(|y_obs_i|, 1) so the result is the conventional
#     Rwp. Pass `weights=np.ones_like(y_obs)` to recover the old unweighted
#     R-profile (Rp) behaviour. Pass `weights = 1/sigma**2` if you have a
#     real per-point variance.
#
#     Note: y_obs here is BG-subtracted, so it can legitimately be small or
#     even slightly negative. The clip at 1 stops 1/y blowing up on
#     near-zero counts (which would otherwise dominate Rw entirely).
#     """
#     y_obs = np.asarray(y_obs, dtype=float)
#     y_calc = np.asarray(y_calc, dtype=float)
#     if weights is None:
        # # Gives Rwp - is buggy due to BG subtraction before fit is carried out
        # weights = 1.0 / np.maximum(np.abs(y_obs), 1.0)
#
        # # Uses Rp - while not as good as Rwp, it gives a better goodness of
        # # fit for data which is processed in EXODUS' fashion.
#         weights = np.ones_like(y_obs)
#
#
#     numerator = np.sum(weights * (y_obs - y_calc)**2)
#     denominator = np.sum(weights * (y_obs)**2)
#     if denominator <= 0:
#         return float('nan')
#
#     Rw = np.sqrt(numerator / denominator)
#     return Rw

# def fitStatistics_legacy(result, twoTheta, fullFitReport=False, verbose=False):
#     """
#     DEPRECATED. Kept as a shim for any external code that imported the old
#     name. The original body called result.model.conf_interval(...), which
#     is broken (MinimizerResult has no .model attribute -- only ModelResult
#     does). Forwards to fitStatistics() instead.
#     """
#     import warnings
#     warnings.warn(
#         "fitStatistics_legacy is deprecated; use fitStatistics() instead.",
#         DeprecationWarning, stacklevel=2,
#     )
#     return fitStatistics(result, twoTheta,
#                          fullFitReport=fullFitReport, verbose=verbose)

# def BM3_K0pp_implied(K0, K0P):
#     """K0'' implied by truncating the Birch-Murnaghan EoS at 3rd order
#     (Anderson 1995):  K0'' = -[(3 - K0')(4 - K0') + 35/9] / K0   (GPa^-1)."""
#     if not K0:
#         return float('nan')
#     return -((3.0 - K0P) * (4.0 - K0P) + 35.0 / 9.0) / K0



# LEGACY: Unused - confidence interval approach not implemented
# LEGACY: residual_for_minimizer commented out - not used by GUI
# def residual_for_minimizer(params, x, y, phases, twoThetaMin=0, twoThetaMax=30):
#     # It should return residual array
#     return LB_fit_Model(params, x, y, phases, twoThetaMin, twoThetaMax,
#                         sigma_mode = 'caglioti')




# ============================================================================
#             OLD STAND-ALONE WORKFLOW (old BatchFit_toolbox.py)
# ============================================================================
# def fit_LB(data_path, phases, poni_file, save_path,
#            frame = 10, twoThetaMin = 0, twoThetaMax = 10,
#            framePressureGuess = [[0,0],[3000,0]],
#            prominence = 0.10, height = 0.08, order = 20,
#            ampGuess = 1, sigGuess = 0.05,
#            excludePeakList=[], plotLB = True):
#     '''
#     Subtracts the BG, and fits the data using the Le Bail method
#     '''
#
    # Load the n-th frame (i.e. diffraction file)
#     twoTheta, valueInt, errorInt = load_frame(frame, data_path, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)
#
    # Calculates the BG
#     try:
#         twoTheta, valueBG = backgroundFit(data_path, frame=frame, twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax,
#                                              peakSearchAuto = True, plotBG=False, prominence=prominence,
#                                              height = height, order=order, excludePeakList=excludePeakList)
#     except:
#         twoTheta = twoTheta
#         valueBG = np.zeros_like(twoTheta)
#         print('No Background subtraction performed. ')
#
    # Subtracts the BG and retruns two np.arrays with BG subtracted data
#     twoTheta, _, _, valueIntBGsub = subtract_BG(twoTheta, valueInt, valueBG)
#
    # Plots Data
#     if plotLB == True:
        # Only run once
#         plt.figure()
#         plt.scatter(twoTheta, valueIntBGsub, marker='.', color='gray', label='Raw Data', s=3)
#
    # Experimental: Checks if there are redundant reflections, i.e. reflections that are 0  or near to 0 in this plot
    # HKL_array[0] = np.delete(HKL_array[0], 1, axis=0)
    # for phase, i in enumerate(HKL_array):
    #     for j in len(phase):
    #         print(HKL_array[j,3])
#
#
    # Initialises the parameters for the fit, i.e. lattice parameters etc.
#     params = initialise_parameters_LB(phases, WL = load_poni(poni_file), ampGuess = max(valueIntBGsub),
#                                       twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax,
#                                       sigma_mode = 'caglioti', sigGuess = 0.05)
#
    # Evaluate the individual components of the fitted model separately
#     guessFit, components = create_guessComponents(twoTheta, phases, params,
#                                             twoThetaMin, twoThetaMax,
#                                             sigma_mode = 'caglioti', printInputComponents = False)
#
    # Plots the initial guess
#     if plotLB == True:
#         plt.plot(twoTheta, guessFit, c='blue', ls='dotted', lw=1, label='Initial guess')
#
#
    # Fit data
#     try:
#         result = minimize(lambda params: LB_fit_Model(params, twoTheta, valueIntBGsub, phases,
#                                                       twoThetaMin, twoThetaMax,
#                                                       sigma_mode = 'caglioti'),
#                           params,
#                           method='least_squares',
                          #options={'max_nfev': 1000, 'verbose': 2},
#                           calc_covar=True,
#                           )
#
        # Will be implemented later to get better errors:
        # # Create Minimizer
        # minimizer = Minimizer(residual_for_minimizer, params,
        #                       fcn_args=(twoTheta, valueIntBGsub, phases, twoThetaMin, twoThetaMax)
        #                       )
#
        # # Perform minimization
        # result = minimizer.minimize()
#
        # # Now compute confidence intervals
        # ci = conf_interval(minimizer, result, sigmas=[1, 2], trace=False, maxiter=200)
        # printfuncs.report_ci(ci)
#
#     except:
#         print('Error in frame ' + str(frame))
#
    # Creates the envelope for the best fit and saves the data
#     bestFit, components = create_fitComponents(twoTheta, phases, params, result,
#                                                save_path, frame, valueIntBGsub,
#                                                twoThetaMin, twoThetaMax,
#                                                sigma_mode = 'caglioti', printOutputComponents = False)
#
    # Fitting Statistics
#     printStats = False
#     printStats = True
#     if printStats == True:
#         estimatedError = fitStatistics(result, twoTheta)
#         Rw = calculate_Rw(y_obs = valueIntBGsub, y_calc = bestFit, weights=None)
#         print(f'Rw for this fit is {Rw:.4g}')
#
    # Creates Tickmarks
#     tickArray = create_ticks(result, phases, twoThetaMin, twoThetaMax, WL=load_poni(poni_file))
#
    # Saves the figure
#     if plotLB == True:
#         tickStep = max(valueIntBGsub)/10
#         for j in range(len(tickArray)):
#             for i in range(len(tickArray[j])):  # Only loop up to the number of reflections for that phase
#                 plt.scatter(tickArray[j][i], -tickStep - (j+2) * tickStep,
#                             marker='|', linewidths=2, s=100, color=colours[j])
#
        # Adds best fit to the plot
#         plt.plot(twoTheta, bestFit, c='red', ls='-', lw=1, label='Best fit')
#         plt.plot(twoTheta, valueIntBGsub-bestFit-tickStep, c='blue', ls='-', lw=1, label='Residuals')
#         plt.xlim(left=twoThetaMin)
#         plt.xlim(right=twoThetaMax)
#         plt.legend()
#         plt.ylabel('Intensity (arb.u.)')
#         plt.xlabel('TwoTheta (deg)')
#         plt.savefig(save_path + 'fits/' + 'frame_' + str(frame) + '.pdf', format='pdf')
#         plt.show()
#         plt.clf()
#
    # Saves the fit as a dictionary
#     resultsFit = {}
    #print('Following peak positions were calculated: \n' +str(tickArray))
#     for i, phase in enumerate(phases['phases_used']):
#         resultsFit['phase_'+str(phase)+'_unit_cell_fit'] = unitCell_fit(result, phase)
#         print('Unit Cell ' + str(phase) + ' fit: \n' + str(unitCell_fit(result, phase)))
#         resultsFit['phase_'+str(i)+'_unit_cell_error'] = np.array([estimatedError,
#                                                                    estimatedError,
#                                                                    estimatedError,
#                                                                    estimatedError,
#                                                                    estimatedError,
#                                                                    estimatedError])
#         resultsFit['phase_'+str(phase)+'_V_fit'] = float(EoS.unitCellVolume([result.params['a_'+str(phase)].value, result.params['b_'+str(phase)].value, result.params['c_'+str(phase)].value,
#                           result.params['alp_'+str(phase)].value,result.params['bet_'+str(phase)].value,result.params['gam_'+str(phase)].value]))
#         resultsFit['phase_'+str(phase)+'_P_fit'] = float(EoS.BM3_EOS(resultsFit['phase_'+str(phase)+'_V_fit'],
#                                                         V0 = float(phases['phase_'+str(phase)+'_compression_constants'][0]),
#                                                         K0 = float(phases['phase_'+str(phase)+'_compression_constants'][1]),
#                                                         K0P = float(phases['phase_'+str(phase)+'_compression_constants'][2])))
#         resultsFit['phase_'+str(phase)+'_BG_fit'] = result.params['BG'].value
#         resultsFit['phase_'+str(phase)+'_peakPosition_fit'] = tickArray[i]
#         resultsFit['data_BGsub'] = np.array([twoTheta, valueIntBGsub])
#         resultsFit['data_fit'] = np.array([twoTheta, bestFit])
#
#     print('\n============================================= \n')
#
#     return resultsFit



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




# ============================================================================
#                   OLD SCRIPT DRIVER (old EXODUS_core.py)
# ============================================================================
# '''
# Indexes ONE specific frame with the initial guess based on your choosen parameters
# index_pattern = False
# index_pattern = True
# if index_pattern == True:
#     frame = 2
#
    # Specifies the phases and creates the phases dictionary (dic)
#     phases = tb.load_JCPDS(JCPDS)
#     tb.specify_phases(phases, phase_rules, frame)
#
    # Specifies the phases and updates pressure/unit cell in dic
#     for i, phase in enumerate(phases['phases_used']):
        # Legacy function; needs a unit cell and dP but if they are set to 0, they will be ignored
#         tb.update_phases(frame, phases, phase, framePressureGuess,
#                          deltaP=0, resultsFit=[],
#                          resultsFitUC = [0,0,0,0,0,0],
#                          method = 'pressureHelper')
#
    # Does the indexing
#     tb.index_LB(data_path, phases, poni_file, save_path,
#                frame, twoThetaMin = 9, twoThetaMax = 18, plotIndex=True,
#                framePressureGuess=[[0,0],[3000,0]],
#                prominence = prominence, height = height, order = 20, excludePeakList=[],
#                ampGuess = 1, sigGuess = 0.05)
#
###############################################################################
#
# Fits ONE pattern for a specific frame based on your specified parameters
# simpleTest=False
# simpleTest=True
# if simpleTest == True:
#     frame = 2
#     phases = tb.load_JCPDS(JCPDS)
#
#     tb.specify_phases(phases, phase_rules, frame)
#
#     for i, phase in enumerate(phases['phases_used']):
        # Legacy function; needs a unit cell and dP but if they are set to 0, they will be ignored
#         tb.update_phases(frame, phases, phase, framePressureGuess,
#                          deltaP=0, resultsFit=[],
#                          resultsFitUC = [0,0,0,0,0,0],
#                          method = 'pressureHelper')
#
    # Does the fitting
#     resultsFit = tb.fit_LB(data_path, phases, poni_file, save_path,
#                            frame, twoThetaMin = 9, twoThetaMax = 18, plotLB=True,
#                            framePressureGuess=framePressureGuess,
#                            prominence = prominence, height = height, order = 6, excludePeakList=[],
#                            ampGuess = 1, sigGuess = 0.05)
    # This is how you access the output:
    # resultsFit['phase_'+str(phase)+'_unit_cell_fit']
    # resultsFit['phase_'+str(i)+'_unit_cell_error']
    # resultsFit['phase_'+str(phase)+'_V_fit']
    # resultsFit['phase_'+str(phase)+'_P_fit']
    # resultsFit['phase_'+str(phase)+'_BG_fit']
    # resultsFit['phase_'+str(phase)+'_peakPosition_fit']
    # resultsFit['data_BGsub']
    # resultsFit['data_fit']
#
#     print('Pressure in this frame is ' + str(resultsFit['phase_0_P_fit']) + ' GPa.')
#
###############################################################################
#
# Fits a sequence of patterns (i.e. the actual batch fit part)
# It calls tb.fit_LB for every fit.
# fitAll = False
# fitAll = True
# if fitAll == True:
#
    # Creates the phases dic
#     phases = tb.load_JCPDS(JCPDS)
#
    # Specifies the UC for the first apprearance of a high-pressure phase
#     for phase in phases['phases_Number_array']:
#         P_trans = transition_pressures.get(phase, 0.0)  # default 0.0 if not found
#         EoS.specify_transitionPressure(phases, phase, P_trans)
#
    # Does the sequential fitting
#     tb.plot_LB_fit(data_path, phases, poni_file, save_path,
#                    framePressureGuess, phase_rules,
#                    collectionTime = 1, #ylabel = 'time (ms)',
#                    frameStart = 1, frameEnd = 10, frameStep = 1,
#                    twoThetaMin = 9, twoThetaMax = 18,
#                    plotLB=False, logPlot = True, CorrNegativeValues = 0.02,
#                    prominence = prominence, height = height, order = 6, excludePeakList=[],
#                    ampGuess = 1, sigGuess = 0.05,
#                    method = 'pressureHelper', deltaP = 0.5,
#                    offsetStep = 20)
#
###############################################################################
#
# Fits ONE single peak in ONE pattern based on the chosen HKL value.
# singleTest=False
#singleTest=True
# if singleTest== True:
#     frame = 50
#     phase = 0
#
    # Creates the phases dic, and updates it
#     phases = tb.load_JCPDS(JCPDS)
#     tb.specify_phases(phases, phase_rules, frame)
#     for i, phase in enumerate(phases['phases_used']):
#         tb.update_phases(frame, phases, phase, framePressureGuess,
#                          deltaP=0, resultsFit=[],
#                          resultsFitUC = [0,0,0,0,0,0],
#                          method = 'pressureHelper')
#
    # Does the fitting
#     resultsFit = tb.fit_peak_single(data_path, phases, poni_file, save_path,
#                                     frame, twoThetaMin, twoThetaMax, framePressureGuess,
#                                     phase_used = 1, HKL_input = [1,1,0],
#                                     prominence = 0.10, height= 0.08, order = 1,
#                                     ampGuess = 1, sigGuess = 0.05,
#                                     excludePeakList=[], plotFit=True)
    # This is how you access the output:
    # resultsFit['fitted_HKL'] = np.array([H,K,L])
    # resultsFit['fitParameters'] = np.array([cen, amp, sig])
    # resultsFit['fitErrors'] = np.array([cen_err, amp_err, sig_err])
    # resultsFit['data_BGsub'] = np.array([twoTheta, valueInt])
    # resultsFit['data_fit'] = np.array([twoTheta, best_fit])
#
#
#
###############################################################################
#
# Fits a sequence of patterns with ONE single peak based on the chosen HKL value.
# singleAll=False
#singleAll=True
# if singleAll== True:
#
    # Loads Phases
#     phases = tb.load_JCPDS(JCPDS)
#
    # Specifies the UC for the first apprearance of a high-pressure phase
#     for phase in phases['phases_Number_array']:
#         P_trans = transition_pressures.get(phase, 0.0)  # default 0.0 if not found
#         EoS.specify_transitionPressure(phases, phase, P_trans)
#
    # Does the sequential fitting
#     tb.plot_peak_fit_single(data_path, phases, poni_file, save_path, phase_rules,
#                              twoThetaMin = 8, twoThetaMax = 18, framePressureGuess = framePressureGuess,
#                              collectionTime=collectionTime, ylabel = 'time (ms)',
#                              frameStart = 40, frameEnd = 90, frameStep = 1,
#                              phase_used = 0, HKL_input = [1,1,1],
#                              plotFit=False, logPlot = True, CorrNegativeValues = 0.02,
#                              prominence = 0.10, height = 0.08, order = 1, excludePeakList=[],
#                              ampGuess = 1, sigGuess = 0.05,
#                              method = 'pressureHelper', deltaP = 0.5,
#                              offsetStep = 0.05)
#
# print('\n\t It took', round(time.time()-start), 'seconds to run this script.')
#
# '''


# ============================================================================
#                      MAIN WINDOW (old EXODUS_main.py)
# ============================================================================
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

#     @staticmethod
#     def _at_bound(p):
#         if p is None or not p.vary:
#             return False
#         tol = 1e-6 * max(abs(p.min) if np.isfinite(p.min) else 1.0,
#                          abs(p.max) if np.isfinite(p.max) else 1.0, 1.0)
#         return ((np.isfinite(p.min) and abs(p.value - p.min) < tol) or
#                 (np.isfinite(p.max) and abs(p.value - p.max) < tol))

#     @staticmethod
#     def _get_sigma_and_err(params, sigma_mode, phase, hkl_idx, cen, deg_to_rad):
#         """Return (sigma_value, sigma_stderr) for one peak.
#
#         Mirrors get_sigma() in BatchFit_toolbox but also returns the
#         propagated standard error for the value used at this peak. For
#         caglioti mode the error scales as 1/cos(theta) the same way the
#         value does.
#         """
#         def _stderr(p):
#             return float(p.stderr) if p.stderr is not None else np.nan
#
#         if sigma_mode == 'fixed':
#             p = params['sig_global']
#             return float(p.value), _stderr(p)
#
#         if sigma_mode == 'per_phase':
#             p = params[f'sig_{phase}']
#             return float(p.value), _stderr(p)
#
#         if sigma_mode == 'caglioti':
#             theta = cen / 2.0 * deg_to_rad
#             cos_th = np.cos(theta) if np.cos(theta) != 0 else np.nan
#             p = params['sig_global']
#             val = float(p.value) / cos_th
#             err = _stderr(p) / cos_th if np.isfinite(_stderr(p)) else np.nan
#             return val, err
#
#         if sigma_mode == 'separate':
#             key = f'sig_{phase}_{hkl_idx + 1}'
#             if key not in params:
                # Fall back to highest available index for this phase
#                 existing = [k for k in params
#                             if k.startswith(f'sig_{phase}_')]
#                 if not existing:
#                     return np.nan, np.nan
#                 key = sorted(existing,
#                              key=lambda k: int(k.rsplit('_', 1)[-1]))[-1]
#             p = params[key]
#             return float(p.value), _stderr(p)
#
#         return np.nan, np.nan

#     def _phase_index_for_active_tab(self):
#         """Return the phase index that the currently-active pressure
#         guess tab is bound to, or None."""
#         table = self._active_pressure_guess_table()
#         if table is None:
#             return None
#         pi = table.property('phase_index')
#         if pi is None:
#             return None
#         try:
#             return int(pi)
#         except (TypeError, ValueError):
#             return None

#     def _rebuild_2d_on_range_change(self):
#         """Legacy helper - kept for backwards compatibility but no longer
#         wired to anything. Use recalculate_bg() instead."""
#         if not self.file_paths:
#             return
#         try:
#             self._build_full_2d_stack(showProgressDialog=False)
#             self.update_2d_plot()
#         except Exception as e:
#             print(f"  2D rebuild failed: {e}")

#     def plot_peaks_on_2d(self, frame, usedPhases):
#         for phase in usedPhases:
#
#             peaks = self.sequentialFitResultsPeaks[f'frame_{frame}'][f'phase_{phase}']
#
#             for peak in peaks:
#                 line = pg.InfiniteLine(
#                     pos=peak,
#                     angle=90,
#                     pen=pg.mkPen('r', width=1)
#                 )
#                 self.plotWidget_2D.addItem(line)

#     def plot_2d_raw_data(self, resamplePoints=None):
#         """
#         Prepare 2D array for plotting:
#         - shape = (num_patterns, num_points)
#         - intensity only (background-subtracted)
#         - twoTheta stored separately
#         Optional resampling for faster plotting.
#         """
#         if not hasattr(self, 'file_paths') or len(self.file_paths) == 0:
#             print("No files loaded.")
#             return
#
#         method = "ALS" if self.ui.checkBox_ALSfit.isChecked() else "POLY"
#
#         params = {
#             "twoThetaMin": self.ui.doubleSpinBox_twoThetaMin.value(),
#             "twoThetaMax": self.ui.doubleSpinBox_twoThetaMax.value(),
#             "lam": self.ui.doubleSpinBox_ALS_lam.value(),
#             "p": self.ui.doubleSpinBox_ALS_p.value(),
#             "niter": self.ui.doubleSpinBox_ALS_niter.value(),
#             "prominence": self.ui.doubleSpinBox_BG_Prominence.value(),
#             "height": self.ui.doubleSpinBox_BG_Height.value(),
#             "order": self.ui.spinBox_BG_Order.value(),
#             "peakwidth": self.ui.doubleSpinBox_BG_width.value(),
#         }
#
#         allIntensities = []
#         twoTheta_ref = None
#
#         for idx, filePath in enumerate(self.file_paths):
#             result = self.data_manager.get_background(filePath, method, params)
#             if result is None:
#                 continue
#
#             twoTheta, _, _, valueIntBGsub = result
#
#             if resamplePoints is not None and len(twoTheta) != resamplePoints:
#                 from scipy.signal import resample
#                 valueIntBGsub = resample(valueIntBGsub, resamplePoints)
#                 twoTheta = resample(twoTheta, resamplePoints)
#
#             allIntensities.append(valueIntBGsub)
#
#             if twoTheta_ref is None:
#                 twoTheta_ref = twoTheta
#
#         if not allIntensities:
#             print("No patterns could be processed.")
#             return
#
        # Flip the list so that pattern 0 is last (bottom)
#         allIntensities = allIntensities[::-1]
#
        # Convert to 2D array: (num_patterns, num_points)
#         self.plot2DrawData = np.array(allIntensities)
#         self.twoTheta = twoTheta_ref
#
#         print("plot2DrawData shape:", self.plot2DrawData.shape)

#     def has_errors(self):
#         return any(np.isfinite(e) and e > 0 for _v, e in self.data.values())

#     def _read_eos(self):
        # Returns (K0, K0P, alphaT, dK0dT, dK0PdT)
#         return (self.spn_K0.value(), self.spn_K0P.value(),
#                 self.spn_alphaT.value(), self.spn_dK0dT.value(),
#                 self.spn_dK0PdT.value())

        # elif self.ui.checkBox_sequentialplus.isChecked():
        #     print("Running Sequential Plus")
        #     self.sequentialPlus_fit()
        # self.fitModeGroup.addButton(self.ui.checkBox_sequentialplus)
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
    #         UC_at_P = tb.find_uc_at_p(UC0, P, V0, K0, K0P)

    #         # Update the variable unit cell used for plotting
    #         self.phases[f'phase_{phase_index}_unit_cell'] = np.array(UC_at_P)

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
