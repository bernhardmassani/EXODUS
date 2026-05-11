# NOTE: The triple-quoted block at the bottom of this file is legacy
# example/script code. It is NOT executed. Safe to delete once you have
# confirmed it is no longer needed as reference.

"""
EXODUS
Equation-of-state X-ray Observation and Diffraction Unit-cell Solver

Version 1.0 - August 2025
author: Dr Bernhard Massani
"""

import EoS_toolbox as EoS
import BatchFit_toolbox as tb
#import Constants as const
import time
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from lmfit import Parameters, minimize, fit_report
from lmfit import Minimizer, Parameters, conf_interval
from lmfit.confidence import conf_interval
from lmfit.printfuncs import report_ci
import re # for splitting JCPDS/HKL
from matplotlib.colors import LogNorm

'''
##################
#  Dependencies  #
##################
# This script needs lmfit to work
# To install put the line into the conda terminal: conda install -c conda-forge lmfit 


###########################
#  Feeedback/Suggestions  #
###########################

EXODUS_main.py        → GUI only
Exodus_core.py        → Workflow orchestration
BatchFit_toolbox.py   → Maths/Physics (Diffraction)
EoS_toolbox.py        → Maths/Physics (Equations of State)


# BM: Propagate amp and sigma is done, but also propagate shift parameters?
# BM: add loading of other data files from dioptas. e.g. xy files.
'''

#################################
###       General Input       ###
#################################

def load_WL(poni_file):
    WL = tb.load_poni(poni_file)
    return WL


def load_JCPDS(JCPDS):
    phases = tb.load_JCPDS(JCPDS)
    return phases
    

#################################
###       Background Fit      ###
#################################
def run_background_fit(data_path, twoThetaMin, twoThetaMax,
                       prominence, height, order, excludePeakList,
                       peakwidth = 0.35):

    twoTheta, valueInt, errorInt, valueBG, peaks = tb.backgroundFit(
        data_path,
        twoThetaMin=twoThetaMin,
        twoThetaMax=twoThetaMax,
        peakSearchAuto=True,
        prominence=prominence,
        height=height,
        order=order,
        excludePeakList=excludePeakList,
        peakwidth = 0.35)
    
    twoTheta, valueInt, valueBG, valueIntBGsub = tb.subtract_BG(twoTheta, 
                                                             valueInt, valueBG)  

    return twoTheta, valueInt, errorInt, valueBG, valueIntBGsub, peaks 


def run_backgroundFit_ALS(data_path, twoThetaMin, twoThetaMax,
                      lam, p, niter):

    twoTheta, valueInt, errorInt, valueBG, valueIntBGsub = tb.backgroundFit_ALS(
                                                          data_path, 
                                                          twoThetaMin, 
                                                          twoThetaMax,
                                                          lam, p, niter)
    
    # twoTheta, valueInt, valueBG, valueIntBGsub = tb.subtract_BG(twoTheta, 
    #                                                          valueInt, valueBG)
    
    return twoTheta, valueInt, errorInt, valueBG, valueIntBGsub


#################################
###        Diffraction        ###
#################################
def calculate_tickmarks(phases, WL, twoThetaMin, twoThetaMax, i):
    # reflection_List(unit_cell, HKL, WL, twoThetaMin = 5, twoThetaMax = 30, crystal_system = 'CUBIC')
    dk_reflection_list, twoTheta_reflection_list =  tb.reflection_List(
        [phases[f'phase_{i}_unit_cell'][0], 
         phases[f'phase_{i}_unit_cell'][1], 
         phases[f'phase_{i}_unit_cell'][2],
         phases[f'phase_{i}_unit_cell'][3],
         phases[f'phase_{i}_unit_cell'][4],
         phases[f'phase_{i}_unit_cell'][5]], 
         HKL = phases[f'phase_{i}_HKL'],
         crystal_system = phases[f'phase_{i}_crystal_system'], 
         WL = WL,
         twoThetaMin = twoThetaMin, twoThetaMax = twoThetaMax)
    #print(twoTheta_reflection_list)
    return twoTheta_reflection_list
    

def fit_LB(twoTheta, valueIntBGsub, phases, WL,
           frame=10, twoThetaMin=0, twoThetaMax=10,
           framePressureGuess=[[0,0],[3000,0]],
           ampGuess=1, sigGuess=0.05,
           sigma_mode='caglioti', sigmaBounds=2.0,
           ampPrefactor=1.0, maxShift=0.5,
           excludePeakList=[],
           ftol=1e-5, xtol=1e-5, gtol=1e-5):
    '''
    Subtracts the BG, and fits the data using the Le Bail method.

    `ftol`, `xtol`, `gtol` are passed straight through to scipy's
    least_squares (via lmfit). Defaults of 1e-5 give near-identical
    lattice parameters to the old 1e-8 default at ~2.5x the speed.
    '''   
    
    # Experimental: Checks if there are redundant reflections, i.e. reflections that are 0  or near to 0 in this plot
    # HKL_array[0] = np.delete(HKL_array[0], 1, axis=0)
    # for phase, i in enumerate(HKL_array):
    #     for j in len(phase): 
    #         print(HKL_array[j,3])
            
            
    # Initialises the parameters for the fit, i.e. lattice parameters etc.
    params = tb.initialise_parameters_LB(phases, WL,
                                         ampGuess=max(valueIntBGsub) * ampPrefactor,
                                         twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                                         sigma_mode=sigma_mode, sigGuess=sigGuess,
                                         sigmaBounds=sigmaBounds,
                                         maxShift=maxShift)   

    # Evaluate the individual components of the fitted model separately
    guessFit, components = tb.create_guessComponents(twoTheta, phases, params,
                                            twoThetaMin, twoThetaMax,
                                            sigma_mode=sigma_mode,
                                            printInputComponents=False)
    
    # Fit data
    result = None
    try:
        result = minimize(lambda params: tb.LB_fit_Model(params, twoTheta, valueIntBGsub,
                                                         phases, twoThetaMin, twoThetaMax,
                                                         sigma_mode=sigma_mode),
                          params,
                          method='least_squares',
                          calc_covar=True,
                          ftol=ftol, xtol=xtol, gtol=gtol,
                          )
        import numpy as np

        # # For debugging only! This prints a full error report of the fits.
        # print("=" * 60)
        # print(f"covar is None?  {result.covar is None}")
        # print(f"success:        {result.success}")
        # print(f"message:        {result.message}")
        # print(f"nfev:           {result.nfev}")
        # print(f"chisqr:         {result.chisqr}")
        # print(f"redchi:         {result.redchi}")
        # print(f"n_var_params:   {len(result.var_names)}")
        # print(f"var_names:      {result.var_names[:20]}...")  # first 20
        
        # # Check whether any params hit bounds
        # print("\nParams at bounds:")
        # for name in result.var_names:
        #     p = result.params[name]
        #     if p.min > -np.inf and p.max < np.inf:
        #         tol = 1e-5 * max(abs(p.min), abs(p.max), 1)
        #         if abs(p.value - p.min) < tol or abs(p.value - p.max) < tol:
        #             mn = "MIN" if abs(p.value-p.min) < tol else "MAX"
        #             print(f"  {name}: AT {mn}  value={p.value}")
        
        # # Print the lattice params
        # print("\nLattice params:")
        # for name in result.var_names:
        #     if any(name.startswith(p) for p in ('a_','b_','c_','alp_','bet_','gam_')):
        #         p = result.params[name]
        #         print(f"  {name}: value={p.value:.6f}, stderr={p.stderr}, "
        #               f"min={p.min}, max={p.max}, vary={p.vary}")
        # print("=" * 60)

    except Exception as e:
        print(f'Error in frame {frame}: {e}')
        return {}

    if result is None:
        return {}
                              
    # Creates the envelope for the best fit and saves the data
    bestFit, components = tb.create_fitComponents(twoTheta=twoTheta, phases=phases,
                                                  params=params, result=result,
                                                  frame=frame, valueIntBGsub=valueIntBGsub,
                                                  twoThetaMin=twoThetaMin, twoThetaMax=twoThetaMax,
                                                  sigma_mode=sigma_mode,
                                                  printOutputComponents=False, save_path='')
    
    # Fitting Statistics
    estimatedError = tb.fitStatistics(result, twoTheta)
    Rw = tb.calculate_Rw(y_obs=valueIntBGsub, y_calc=bestFit, weights=None)
    print(f'Rw for this fit is {Rw:.4g}')
        
    # Creates Tickmarks
    tickArray = tb.create_ticks(result, phases, twoThetaMin, twoThetaMax, WL)
                                  
    # Saves the fit as a dictionary
    resultsFit = {}
    # Cache the global Rw and the fit's chi^2 stats so the GUI can show
    # them on the processed plot when the user navigates back.
    resultsFit['Rw'] = float(Rw)
    resultsFit['chisqr'] = float(result.chisqr)
    resultsFit['redchi'] = float(result.redchi)
    resultsFit['nfev'] = int(result.nfev)

    for i, phase in enumerate(phases['phases_used']):
        resultsFit['phase_'+str(phase)+'_unit_cell_fit'] = tb.unitCell_fit(result, phase)
        print('Unit Cell ' + str(phase) + ' fit: \n' + str(tb.unitCell_fit(result, phase)))

        # Per-parameter standard errors. Falls back to the global
        # sqrt(reduced chi-square) if covariance is not available
        # (e.g. fit on bounds), so callers always get a sensible number.
        resultsFit['phase_'+str(phase)+'_unit_cell_error'] = tb.unitCell_errors(
            result, phase, fallback=estimatedError)

        resultsFit['phase_'+str(phase)+'_V_fit'] = float(EoS.unitCellVolume([result.params['a_'+str(phase)].value, result.params['b_'+str(phase)].value, result.params['c_'+str(phase)].value,
                          result.params['alp_'+str(phase)].value,result.params['bet_'+str(phase)].value,result.params['gam_'+str(phase)].value]))
        resultsFit['phase_'+str(phase)+'_P_fit'] = float(EoS.BM3_EOS(resultsFit['phase_'+str(phase)+'_V_fit'], 
                                                        V0 = float(phases['phase_'+str(phase)+'_compression_constants'][0]),
                                                        K0 = float(phases['phase_'+str(phase)+'_compression_constants'][1]), 
                                                        K0P = float(phases['phase_'+str(phase)+'_compression_constants'][2])))
        resultsFit['phase_'+str(phase)+'_BG_fit'] = result.params['BG'].value   # this is a constant value!!
        resultsFit['phase_'+str(phase)+'_peakPosition_fit'] = tickArray[i]
        resultsFit['data_BGsub'] = np.array([twoTheta, valueIntBGsub])
        resultsFit['data_fit'] = np.array([twoTheta, bestFit])
        
    print('\n============================================= \n')

    return resultsFit




























'''
# Indexes ONE specific frame with the initial guess based on your choosen parameters
index_pattern = False
index_pattern = True
if index_pattern == True:
    frame = 2
   
    # Specifies the phases and creates the phases dictionary (dic)
    phases = tb.load_JCPDS(JCPDS)
    tb.specify_phases(phases, phase_rules, frame)
    
    # Specifies the phases and updates pressure/unit cell in dic
    for i, phase in enumerate(phases['phases_used']):
        # Legacy function; needs a unit cell and dP but if they are set to 0, they will be ignored
        tb.update_phases(frame, phases, phase, framePressureGuess, 
                         deltaP=0, resultsFit=[],
                         resultsFitUC = [0,0,0,0,0,0],       
                         method = 'pressureHelper')
    
    # Does the indexing
    tb.index_LB(data_path, phases, poni_file, save_path, 
               frame, twoThetaMin = 9, twoThetaMax = 18, plotIndex=True, 
               framePressureGuess=[[0,0],[3000,0]],
               prominence = prominence, height = height, order = 20, excludePeakList=[],
               ampGuess = 1, sigGuess = 0.05)

###############################################################################

# Fits ONE pattern for a specific frame based on your specified parameters
simpleTest=False 
simpleTest=True 
if simpleTest == True: 
    frame = 2
    phases = tb.load_JCPDS(JCPDS)
    
    tb.specify_phases(phases, phase_rules, frame)
    
    for i, phase in enumerate(phases['phases_used']):
        # Legacy function; needs a unit cell and dP but if they are set to 0, they will be ignored
        tb.update_phases(frame, phases, phase, framePressureGuess, 
                         deltaP=0, resultsFit=[],
                         resultsFitUC = [0,0,0,0,0,0], 
                         method = 'pressureHelper')
    
    # Does the fitting                 
    resultsFit = tb.fit_LB(data_path, phases, poni_file, save_path, 
                           frame, twoThetaMin = 9, twoThetaMax = 18, plotLB=True, 
                           framePressureGuess=framePressureGuess,
                           prominence = prominence, height = height, order = 6, excludePeakList=[],
                           ampGuess = 1, sigGuess = 0.05)
    # This is how you access the output:  
    # resultsFit['phase_'+str(phase)+'_unit_cell_fit']
    # resultsFit['phase_'+str(i)+'_unit_cell_error']
    # resultsFit['phase_'+str(phase)+'_V_fit']
    # resultsFit['phase_'+str(phase)+'_P_fit']
    # resultsFit['phase_'+str(phase)+'_BG_fit']
    # resultsFit['phase_'+str(phase)+'_peakPosition_fit']
    # resultsFit['data_BGsub']
    # resultsFit['data_fit']
    
    print('Pressure in this frame is ' + str(resultsFit['phase_0_P_fit']) + ' GPa.')

###############################################################################

# Fits a sequence of patterns (i.e. the actual batch fit part)
# It calls tb.fit_LB for every fit.
fitAll = False
fitAll = True
if fitAll == True:
    
    # Creates the phases dic
    phases = tb.load_JCPDS(JCPDS)
    
    # Specifies the UC for the first apprearance of a high-pressure phase
    for phase in phases['phases_Number_array']:
        P_trans = transition_pressures.get(phase, 0.0)  # default 0.0 if not found
        EoS.specify_transitionPressure(phases, phase, P_trans)
    
    # Does the sequential fitting
    tb.plot_LB_fit(data_path, phases, poni_file, save_path,
                   framePressureGuess, phase_rules,
                   collectionTime = 1, #ylabel = 'time (ms)',
                   frameStart = 1, frameEnd = 10, frameStep = 1, 
                   twoThetaMin = 9, twoThetaMax = 18, 
                   plotLB=False, logPlot = True, CorrNegativeValues = 0.02,
                   prominence = prominence, height = height, order = 6, excludePeakList=[],
                   ampGuess = 1, sigGuess = 0.05,
                   method = 'pressureHelper', deltaP = 0.5,
                   offsetStep = 20)
    
###############################################################################

# Fits ONE single peak in ONE pattern based on the chosen HKL value.
singleTest=False 
#singleTest=True
if singleTest== True:
    frame = 50
    phase = 0
    
    # Creates the phases dic, and updates it
    phases = tb.load_JCPDS(JCPDS)
    tb.specify_phases(phases, phase_rules, frame)
    for i, phase in enumerate(phases['phases_used']):
        tb.update_phases(frame, phases, phase, framePressureGuess, 
                         deltaP=0, resultsFit=[],
                         resultsFitUC = [0,0,0,0,0,0], 
                         method = 'pressureHelper')
    
    # Does the fitting
    resultsFit = tb.fit_peak_single(data_path, phases, poni_file, save_path, 
                                    frame, twoThetaMin, twoThetaMax, framePressureGuess,
                                    phase_used = 1, HKL_input = [1,1,0], 
                                    prominence = 0.10, height= 0.08, order = 1, 
                                    ampGuess = 1, sigGuess = 0.05, 
                                    excludePeakList=[], plotFit=True)
    # This is how you access the output:
    # resultsFit['fitted_HKL'] = np.array([H,K,L])
    # resultsFit['fitParameters'] = np.array([cen, amp, sig])
    # resultsFit['fitErrors'] = np.array([cen_err, amp_err, sig_err])
    # resultsFit['data_BGsub'] = np.array([twoTheta, valueInt])
    # resultsFit['data_fit'] = np.array([twoTheta, best_fit])  
                        
                        
                        
###############################################################################

# Fits a sequence of patterns with ONE single peak based on the chosen HKL value.    
singleAll=False 
#singleAll=True
if singleAll== True:
    
    # Loads Phases
    phases = tb.load_JCPDS(JCPDS)
    
    # Specifies the UC for the first apprearance of a high-pressure phase
    for phase in phases['phases_Number_array']:
        P_trans = transition_pressures.get(phase, 0.0)  # default 0.0 if not found
        EoS.specify_transitionPressure(phases, phase, P_trans)
    
    # Does the sequential fitting
    tb.plot_peak_fit_single(data_path, phases, poni_file, save_path, phase_rules,
                             twoThetaMin = 8, twoThetaMax = 18, framePressureGuess = framePressureGuess,
                             collectionTime=collectionTime, ylabel = 'time (ms)',
                             frameStart = 40, frameEnd = 90, frameStep = 1,
                             phase_used = 0, HKL_input = [1,1,1], 
                             plotFit=False, logPlot = True, CorrNegativeValues = 0.02,
                             prominence = 0.10, height = 0.08, order = 1, excludePeakList=[],
                             ampGuess = 1, sigGuess = 0.05, 
                             method = 'pressureHelper', deltaP = 0.5,
                             offsetStep = 0.05)
    
print('\n\t It took', round(time.time()-start), 'seconds to run this script.') 

'''