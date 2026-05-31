"""
Created on Sat Feb  1 23:32:35 2025
@ Dr Bernhard Massani 
-*- coding: utf-8 -*-
"""
import math
import BatchFit_toolbox as tb
import numpy as np
import matplotlib as plt
from scipy.optimize import brentq
import matplotlib.pyplot as plt

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



###############
## UC Volume ##
###############

def unitCellVolume(unit_cell):
    '''
    Input unit cell array; output volume.

    Returns NaN when the cell geometry is invalid (negative radicand, i.e.
    angles that don't form a closable parallelepiped). The previous
    behaviour was to return 1, which silently masked geometry errors and
    propagated meaningless-but-finite pressures into the BM3 EOS. NaN
    surfaces the problem instead.
    '''
    a, b, c = unit_cell[0], unit_cell[1], unit_cell[2]
    alp = math.radians(unit_cell[3])
    bet = math.radians(unit_cell[4])
    gam = math.radians(unit_cell[5])
    radicand = (1 - math.cos(alp)**2 - math.cos(bet)**2 - math.cos(gam)**2
                + 2 * math.cos(alp) * math.cos(bet) * math.cos(gam))
    if radicand <= 0:
        return float('nan')
    try:
        volume = a * b * c * math.sqrt(radicand)
    except (ValueError, TypeError):
        return float('nan')
    return volume

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


def find_volume_at_pressure(P, unit_cell, K0, K0P, max_iter=50):
    """
    Find volume at a given pressure using the third-order Birch-Murnaghan EOS based on unit_cell and JCPDS input
    Returns: Array [a, b, c, alpha, beta, gamma, V] specifying the lattice constants and volume at the given pressure
    """
    V0 = unitCellVolume(unit_cell)

    def volume_difference(V):
        return BM3_EOS(V, V0, K0, K0P) - P

    # Start with reasonable bounds
    V_low = V0 * 0.5
    V_high = V0 * 1.5

    # Expand bounds until function changes sign
    for i in range(max_iter):
        f_low = volume_difference(V_low)
        f_high = volume_difference(V_high)

        if f_low * f_high < 0:
            # Sign change found → safe to apply brentq
            try:
                V = brentq(volume_difference, V_low, V_high)
                return V
            except Exception as e:
                print(f"Brentq failed: {e}")
                return None

        # No sign change: expand search window
        V_low *= 0.9   # Decrease lower bound
        V_high *= 1.1  # Increase upper bound

    print("ERROR: Could not find sign change after expanding bounds.")
    return None


def find_UC_at_P(unit_cell_0, P, V0, K0, K0P):
    """
    Scale a 0 GPa unit cell to pressure P using the BM3 EOS.

    LIMITATION (isotropic compression assumption)
    ---------------------------------------------
    The scaling is *isotropic*: all three lattice constants are multiplied
    by (V_target/V_ref)^(1/3) and the angles are kept fixed. This is exact
    for CUBIC, and a reasonable approximation for cells with similar axial
    compressibilities. For tetragonal, hexagonal, orthorhombic, monoclinic,
    and triclinic systems the c-axis (and angles, for mono/triclinic)
    typically respond differently to pressure than a/b. Use this function
    only as an *initial guess* for non-cubic phases; the Le Bail refinement
    will then move each lattice constant independently.
    """
    #V0 = unitCellVolume(unit_cell_0)
    VP = find_volume_at_pressure(P, unit_cell_0, K0, K0P)

    if VP is None:
        print("Volume at pressure could not be determined.")
        return None

    scale = (VP / V0) ** (1/3)
    return [
        unit_cell_0[0] * scale,
        unit_cell_0[1] * scale,
        unit_cell_0[2] * scale,
        unit_cell_0[3],
        unit_cell_0[4],
        unit_cell_0[5]
    ]

def scale_UC_at_P(unitCell_fit, P_target, unitCell_0, K0, K0P):
    """
    Scales a unit cell from pressure P_fit to pressure P_target using the BM3 EOS.
    
    Parameters:
    - unit_cell_fit: list of 6 lattice parameters at P_fit
    - P_fit: known pressure of the unit cell
    - P_target: desired target pressure
    - K0, K0P: BM3 bulk modulus parameters

    Returns:
    - unit_cell_target: scaled unit cell at P_target

    LIMITATION (isotropic compression assumption)
    ---------------------------------------------
    Same caveat as find_UC_at_P: a, b, c are all scaled by the same
    cube-root factor and angles are preserved. Exact for CUBIC, an
    approximation otherwise. Intended as an initial guess for the next
    frame in a sequential ('pressureWalk') fit.
    """

    V_fit = unitCellVolume(unitCell_fit)
    
    # Recover V0 using inverse BM3
    V0 = unitCellVolume(unitCell_0)
    
    if V0 is None:
        print("V0 could not be recovered from fitted unit cell.")
        return None

    # Calculate predicted volume at P_target
    V_P = find_volume_at_pressure(P_target, unitCell_0, K0, K0P)
    
    if V_P is None:
        print("Volume at target pressure could not be determined.")
        return None

    scale = (V_P/ V_fit) ** (1/3)

    V_new = [unitCell_fit[0] * scale,
             unitCell_fit[1] * scale,
             unitCell_fit[2] * scale,
             unitCell_fit[3],
             unitCell_fit[4],
             unitCell_fit[5]
             ]
    
    return V_new



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



###########################
###     EOS Fitting     ###    
###########################



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

def BM3_EOS(V, V0, K0, K0P):
    """
    Calc
    """
    V = float(V)
    V0 = float(V0)
    P = (3/2) * K0 * ((V0 / V)**(7/3) - (V0 / V)**(5/3)) * \
        (1 + (3/4) * (K0P - 4) * ((V0 / V)**(2/3) - 1))
    return P


def calculate_KT(P, K0, K0P):
    """
    Calculate isothermal bulk modulus (KT) for a given P using a linear model
    """
    KT = K0 + K0P * P
    return KT

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