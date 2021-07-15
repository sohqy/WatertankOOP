# -*- coding: utf-8 -*-
"""
PC6 OPTMISATION MODEL 
Qiao Yan Soh, Created 16 Mar 2021
=================================
    Implemented: 
        - Surface Overflow ONLY
        - Separation tank secondary flow (as a large orifice), Conditional. 
        - Overflow as conditional 
        - SH outlet allowed to discharge 0 for feasibility without tank overflow variable
"""        

import pyomo.environ as pyo
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from pyomo.opt import TerminationCondition

import rffxns as rf

#%%
# ===== FORMULATION EQUATIONS ===== 
def HarObjFx(m):
    """ Maximising yield from system. """
    return m.Level['Harvesting', m.t[-1]]


# ----- Tank Mass balance
def SepBalanceFx(m, t):
    """ Discretized mass balance for the Separation Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else: 
        return m.Overflow[t+1]/m.TankArea['Separation'] + m.Level['Separation', t+1] == m.Level['Separation', t] \
                + (m.RainIn[t] + m.Overflow[t] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t]) / m.TankArea['Separation']
        
def HarBalanceFx(m, t):
    """ Discretized mass balance for the Harvesting Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Level['Harvesting', t+1] == m.Level['Harvesting', t] \
            + m.Discharge['SH', t] / m.TankArea['Harvesting']
    
def DetBalanceFx(m, t):
    """ Discretized mass balance for the Detention Tank. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Level['Detention', t+1] == m.Level['Detention', t] \
            + (m.Discharge['SD', t] + m.Discharge['SD2', t] - m.Discharge['DO', t]) / m.TankArea['Detention']
            
# ----- Overflow
def OverflowFx(m, t):
    """ Definition of system overflow. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1] >= m.TankArea['Separation'] * m.Level['Separation', t] + m.RainIn[t] \
            + m.Overflow[t] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t] \
                - m.TankArea['Separation'] * m.TankHeight['Separation']

def OverflowFxUB2(m, t):
    """ Upper bound effective for setting value when condition is True. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t+1] <= m.TankArea['Separation'] * m.Level['Separation', t] + m.RainIn[t] \
            + m.Overflow[t] - m.Discharge['SH', t] - m.Discharge['SD', t] - m.Discharge['SD2', t] \
                - m.TankArea['Separation'] * m.TankHeight['Separation'] \
                    - (m.DischargeLimit['SH'] + m.DischargeLimit['SD'] + m.DischargeLimit['SD2'] +  m.TankArea['Separation'] * m.TankHeight['Separation']) * m.OverflowBinary[t] \
                        + (m.DischargeLimit['SH'] + m.DischargeLimit['SD'] + m.DischargeLimit['SD2'] +  m.TankArea['Separation'] * m.TankHeight['Separation']) 
                
def OverflowLm(m, t):
    """ Desired upper bound for overflow. """
    return m.Overflow[t] <= m.Epsilon * m.OverflowBinary[t]

# ----- Flows 
def SHLowerFx(m, t):
    """ Lower bound effective when condition is True. Deactivated for increased flexibility. """
    return m.Discharge['SH', t] >= 2.66 * m.OrificeArea['SH'] * (m.Level['Separation', t] - m.OrificeHeight['SH']) # * m.DeltaT

def SHUpperFx1(m, t):
    """ Sets upper bound of 0 for SH flow when conditions are not met """
    return m.Discharge['SH', t] <= 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.Binary['SH', t] \
        * (m.TankHeight['Separation'] - m.OrificeHeight['SH'])

def SHUpperFx2(m, t):
    """ Sets upper bound when condition is True. """
    return m.Discharge['SH', t] <= 2.66 * m.OrificeArea['SH'] * m.DeltaT * (m.Level['Separation', t] - m.OrificeHeight['SH']) \
        - 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.Binary['SH', t] * m.OrificeHeight['SH'] \
            + 2.66 * m.OrificeArea['SH'] * m.DeltaT * m.OrificeHeight['SH']
    
            
def SD2LowerFx(m, t):
    """ Lower bound for SD2, effective when condition is True."""
    return m.Discharge['SD2', t] >= 2.66 * m.OrificeArea['SD2'] * (m.Level['Separation', t] - m.OrificeHeight['SD2']) # * m.DeltaT

def SD2UpperFx1(m, t):
    """ Sets upper bound of 0 for SD2 flow when conditions are not met """
    return m.Discharge['SD2', t] <= 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.Binary['SD2', t] \
        * (m.TankHeight['Separation'] - m.OrificeHeight['SD2'])
        
def SD2UpperFx2(m, t): 
    """ Sets upper bound when condition is True. """
    return m.Discharge['SD2', t] <= 2.66 * m.OrificeArea['SD2'] * m.DeltaT * (m.Level['Separation', t] - m.OrificeHeight['SD2']) \
        - 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.Binary['SD2', t] * m.OrificeHeight['SD2'] \
            + 2.66 * m.OrificeArea['SD2'] * m.DeltaT * m.OrificeHeight['SD2']
            
            
def DOFlowFx(m, t):
    """ Inflexible, unconditional flow out of the system. """
    return m.Discharge['DO', t] == 2.66 * m.OrificeArea['DO'] * m.Level['Detention', t] * m.DeltaT


# ----- Vol Cons
def VolConsFx(m, t):
    """ Overall volume conservation of the system. """
    if t == m.t[-1]:
        return pyo.Constraint.Skip
    else:
        return m.Overflow[t] + m.TankArea['Separation'] * m.Level['Separation', t] \
        + m.TankArea['Harvesting'] * m.Level['Harvesting', t] + m.TankArea['Detention'] * m.Level['Detention', t] \
            == m.TankArea['Separation'] * m.Level['Separation', 0] + m.TankArea['Harvesting'] * m.Level['Harvesting', 0] \
                + m.TankArea['Detention'] * m.Level['Detention', 0] \
                    + sum(m.RainIn[k] for k in range(0, t)) - sum(m.Discharge['DO', k] for k in range(0, t))


# ----- End Level
def EndFx(m):
    """ Leaves separation tank levels to a desired level at the end of the optimisation loop """
    return m.Level['Separation', m.t[-1]] <= 0.5


# ----- Bound Assignment and Initialisation 
def LevelBounds(m, T, t): 
    """ Sets upper bounds for Level variables """
    return m.Level[T, t] <= m.TankHeight[T]

def DischargeBounds(m, O, t):
    """ Sets upper bounds for Discharge variables """
    return m.Discharge[O, t] <= m.DischargeLimit[O]


def LevelInitFx(m, T):
    """ Initializes Level variables """
    return m.Level[T, 0] == m.InitialLevel[T]

def DischargeInitFx(m, O):
    """ Initializes Discharge variables """
    return m.Discharge[O, 0] == 0.0 

def OverflowInitFx(m):
    """ Initializes Overflow variable """
    return m.Overflow[0] == 0.0


#%%

def CreateAbstractModel(DeltaT = 300, Epsilon = 100):
    """
    Creates an abstract model based on functions defined in this module file. 
    Returns a PYOMO abstract model object. 
    
    INPUTS: 
        DeltaT : INT, optional. The default is 300.
            Time step size in seconds for the optimisation model. To be matched to rainfall input signal time step. 
        Epsilon : TYPE, optional. The default is 100.
            Upper bound on overflow variable. 
    """
    m = pyo.AbstractModel()
    
    m.t = pyo.Set(ordered = True, doc = 'Simulation timesteps')
    m.Tanks = pyo.Set()
    m.Outlets = pyo.Set()
    m.CondOutlets = pyo.Set(within = m.Outlets)

    m.TankHeight = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.TankArea = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.DischargeLimit = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)
    m.OrificeArea = pyo.Param(m.Outlets, within = pyo.NonNegativeReals)  #np.pi * (0.2 ** 2)    
    m.OrificeHeight = pyo.Param(m.CondOutlets, within = pyo.NonNegativeReals)
    m.InitialLevel = pyo.Param(m.Tanks, within = pyo.NonNegativeReals)
    m.RainIn = pyo.Param(m.t, within = pyo.NonNegativeReals)
    
    m.DeltaT = DeltaT
    m.Epsilon = Epsilon                                   
    
    m.Level = pyo.Var(m.Tanks, m.t, within = pyo.NonNegativeReals)
    m.Overflow = pyo.Var(m.t, within = pyo.NonNegativeReals)
    m.Discharge = pyo.Var(m.Outlets, m.t, within = pyo.NonNegativeReals)
    
    m.Binary = pyo.Var(m.CondOutlets, m.t, within = pyo.Binary)
    m.OverflowBinary = pyo.Var(m.t, within = pyo.Binary)
    
    # ----- 
    m.Obj = pyo.Objective(rule = HarObjFx, sense = pyo.maximize)
    m.SepTank = pyo.Constraint(m.t, rule = SepBalanceFx)
    m.HarTank = pyo.Constraint(m.t, rule = HarBalanceFx)
    m.DetTank = pyo.Constraint(m.t, rule = DetBalanceFx)
    
    m.OF = pyo.Constraint(m.t, rule = OverflowFx)
    m.OFUB = pyo.Constraint(m.t, rule = OverflowFxUB2)
    m.OFLim = pyo.Constraint(m.t, rule = OverflowLm)
    
    #m.SH1 = pyo.Constraint(m.t, rule = SHLowerFx)
    m.SH2 = pyo.Constraint(m.t, rule = SHUpperFx1)
    m.SH3 = pyo.Constraint(m.t, rule = SHUpperFx2)
    
    m.SD21 = pyo.Constraint(m.t, rule = SD2LowerFx)
    m.SD22 = pyo.Constraint(m.t, rule = SD2UpperFx1)
    m.SD23 = pyo.Constraint(m.t, rule = SD2UpperFx2)
    
    m.DO = pyo.Constraint(m.t, rule = DOFlowFx)
    
    m.VolumeConservation = pyo.Constraint(m.t, rule = VolConsFx)
    
    m.EndState = pyo.Constraint(rule = EndFx)
    
    # ----- 
    m.LevelLim = pyo.Constraint(m.Tanks, m.t, rule = LevelBounds)
    m.DischargeLim = pyo.Constraint(m.Outlets, m.t, rule = DischargeBounds)

    m.InitLevel = pyo.Constraint(m.Tanks, rule = LevelInitFx)
    m.InitDischarge = pyo.Constraint(m.Outlets, rule = DischargeInitFx)
    m.InitOverflow = pyo.Constraint(rule = OverflowInitFx)
    
    return m

#%% 
def CompileData(RainArray, SDSize = 0.05):
    """
    Compiles information into dataframe for initializing PYOMO concrete model instances.
    Returns a dictionary of system information. 
    
    INPUTS: 
        RainArray : Pandas Series
            Rainfall signal to be optimised for. 
        SDSize : FLOAT, optional. 
            Size of the SD orifice to be simulated. 
    """
    RFInput = pd.concat([RainArray, pd.Series([0])]).reset_index(drop=True) # Add initial value to rainfall data
    OrificeAreas = {'SH': np.pi * (0.15 ** 2), 'SD': np.pi * (SDSize  ** 2) , 'DO': np.pi*(0.175**2), \
                    'SD2': np.pi * (0.4 ** 2),}
        
    dct = {None: {
        # ----- SETS
        't': {None: list(range(len(RFInput)))}, # This can be changed 
        'Tanks': {None: ['Separation', 'Harvesting', 'Detention']},
        'Outlets' : {None: ['SH', 'SD', 'SD2', 'DO']},
        'CondOutlets' : {None: ['SH', 'SD2']}, 
        # ----- PARAMETERS`
        'TankHeight' :{'Separation': 1.227, 'Harvesting': 2.40, 'Detention': 2},
        'TankArea' : {'Separation': 5.02, 'Harvesting': 42.4, 'Detention': 264.52 },
        # TO DO: Add Secondary flows and change primary flow rates. 
        'DischargeLimit': {'SH': 2.66 * OrificeAreas['SH'] * (1.227 - 0.52) * 300, 'SD': 2.66 * OrificeAreas['SD'] * 1.227 * 300, \
                        'DO': 156, 'SD2': 200 - 2.66 * OrificeAreas['SD'] * 1.227 * 300}, # 156.3 is max allowable by PUB. 
        'OrificeArea': OrificeAreas,
        'OrificeHeight': {'SH': 0.52, 'SD2': 0.66, },
        'InitialLevel': {'Separation': 0.0, 'Detention': 0.0, 'Harvesting': 0.0},
        # ----- RAINFALL INPUT
        'RainIn': dict(enumerate(RFInput)),
           }}
    
    return dct    
        
def CompileDataParams(RainArray, TankHeights = None, TankAreas = None, OrificeAreas = None, InitialLevels = None):
    """
    Compiles information into dataframe for initializing PYOMO concrete model instances.
    Returns a dictionary of system information. 
    
    INPUTS: 
        RainArray : Pandas Series
            Rainfall signal to be optimised for. 
        SDSize : FLOAT, optional. 
            Size of the SD orifice to be simulated. 
    """
    RFInput = pd.concat([RainArray, pd.Series([0])]).reset_index(drop=True) # Add initial value to rainfall data
    
    # Set default if none is provided. 
    if TankHeights is None:
        TankHeights = {'Separation': 1.227, 'Harvesting': 2.40, 'Detention': 2}
    if TankAreas is None:
        TankAreas = {'Separation': 5.02, 'Harvesting': 42.4, 'Detention': 264.52 }
    if OrificeAreas is None:
        OrificeAreas = {'SH': np.pi * (0.15 ** 2), 'SD': np.pi * (0.05 ** 2) , 'DO': np.pi*(0.175**2), \
                    'SD2': np.pi * (0.4 ** 2),}
    if InitialLevels is None:
        InitialLevels = {'Separation': 0.0, 'Detention': 0.0, 'Harvesting': 0.0}
        
    dct = {None: {
        # ----- SETS
        't': {None: list(range(len(RFInput)))}, # This can be changed 
        'Tanks': {None: ['Separation', 'Harvesting', 'Detention']},
        'Outlets' : {None: ['SH', 'SD', 'SD2', 'DO']},
        'CondOutlets' : {None: ['SH', 'SD2']}, 
        # ----- PARAMETERS
        'TankHeight' : TankHeights,
        'TankArea' : TankAreas,
        # TO DO: Add Secondary flows and change primary flow rates. 
        'DischargeLimit': {'SH': 2.66 * OrificeAreas['SH'] * (TankHeights['Separation'] - 0.52) * 300, 'SD': 2.66 * OrificeAreas['SD'] * TankHeights['Separation'] * 300, \
                        'DO': 156, 'SD2': 2.66 * OrificeAreas['SD2'] * (TankHeights['Separation'] - 0.66) * 300}, # 156.3 is max allowable by PUB. 
        'OrificeArea': OrificeAreas,
        'OrificeHeight': {'SH': 0.52, 'SD2': 0.66, },
        'InitialLevel': InitialLevels,
        # ----- RAINFALL INPUT
        'RainIn': dict(enumerate(RFInput)),
           }}
    
    return dct            
        
#%% 
def SolveInstance(inst, solver = 'cplex', PrintSolverOutput = True, Summarise = True, Gap = None):
    """
    
    """
    opt = pyo.SolverFactory(solver)
    
    if Gap is None: 
        Gap = 1e-4
    opt.options['mipgap'] = Gap
    
    #opt.options['feasopt'] == True
    opt.options['timelimit'] = 60
    
    opt_results = opt.solve(inst, tee = PrintSolverOutput, )
    
    
    if Summarise == True: 
        # ----- Reshape and summarize results
        Opt_Level = pd.DataFrame.from_dict(inst.Level.extract_values(), orient = 'index', columns = ['Level'])
        Opt_Level.index = pd.MultiIndex.from_tuples(Opt_Level.index, names = ['Tank', 'Timestep'])
        Opt_Level = Opt_Level.reset_index().pivot(index = 'Timestep', columns = 'Tank', values = 'Level')
        
        Opt_Discharge = pd.DataFrame.from_dict(inst.Discharge.extract_values(), orient = 'index', columns = ['Discharge'])
        Opt_Discharge.index = pd.MultiIndex.from_tuples(Opt_Discharge.index, names = ['Outlet', 'Timestep'])
        Opt_Discharge = Opt_Discharge.reset_index().pivot(index = 'Timestep', columns = 'Outlet', values = 'Discharge')
        
        Rainin = pd.DataFrame.from_dict(inst.RainIn.extract_values(), orient = 'index', columns = ['RainIn'])
        
        Opt_OF = pd.DataFrame.from_dict(inst.Overflow.extract_values(), orient = 'index', columns = ['Overflow'])
       
        results = pd.concat([Rainin, Opt_Level, Opt_Discharge, Opt_OF], axis = 1)
        
        return {'Results':results, 'SolverStatus': opt_results, 'Instance': inst} 
    else:
        return opt_results
    

#%% 
def Run(hist_df, date = '2011-01-30', Eps = 100):
    """
    
    """
    rf_t = rf.extractprofile(date, hist_df).Rainfall * 15.7
    data = CompileData(rf_t)
    m = CreateAbstractModel(Epsilon = Eps)
    instance = m.create_instance(data)
    Res = SolveInstance(instance)
    
    return Res

def RunOptimisation(Data, Eps = 75):
    m = CreateAbstractModel(Epsilon = Eps)
    instance = m.create_instance(Data)
    Res = SolveInstance(instance, PrintSolverOutput = False)

    return Res

def ManualEpsMin(rf_t, TankHeights = None, TankAreas = None, OrificeAreas = None, InitialLevels = None):

    Eps = 0
    InfesFlag = 1
    data = CompileDataParams(rf_t, TankHeights = TankHeights, TankAreas = TankAreas, \
                                       OrificeAreas = OrificeAreas, InitialLevels = InitialLevels)
    while InfesFlag == 1: 
        OptResult = RunOptimisation(data, Eps = Eps)
        
        if (OptResult['SolverStatus'].solver.termination_condition == TerminationCondition.optimal) \
            | (OptResult['SolverStatus'].solver.termination_condition == TerminationCondition.maxTimeLimit):
            InfesFlag = 0 
        else: 
            Eps += 2

    return OptResult

def CalcKPIs(SolvedDict):
    Dataframe = SolvedDict['Results']
    Instance = SolvedDict['Instance']
    
    n_timesteps = len(Instance.t) - 1
    
    Overflow = max(Dataframe.Overflow)
    HarvestedVol = Dataframe.Harvesting[288] * Instance.TankHeight.extract_values()['Harvesting']
    OverflowRisk = sum(Dataframe.Separation )/(n_timesteps * Instance.TankHeight.extract_values()['Separation'])
    Unused =  Instance.TankArea.extract_values()['Separation'] * (Instance.TankHeight.extract_values()['Separation'] - max(Dataframe.Separation))\
        + Instance.TankArea.extract_values()['Detention'] * (Instance.TankHeight.extract_values()['Detention'] - max(Dataframe.Detention))
    DischargeMax = max(Dataframe.DO)
    
    return {'Overflow': Overflow, 'Harvested Volume': HarvestedVol, \
               'Unused Volume': Unused, 'Overflow Risk': OverflowRisk, 'Discharge Max': DischargeMax}

def PlotResults(SolvedDict):
    Dataframe = SolvedDict['Results']
    Instance = SolvedDict['Instance']
    
    colors = sns.color_palette()
    
    plt.figure()
    # ----- Plot rain input
    plt.subplot(311)
    plt.plot(Dataframe.RainIn, label = 'Rainfall')
    plt.ylabel('Rainfall Input  \n [m$^3$]')
    
    # ----- Plot Tank Dynamics
    cnt = 0
    plt.subplot(312)
    plt.plot(Dataframe.Harvesting, label = 'Harvested Volume', color = colors[cnt])
    plt.axhline(Instance.TankHeight.extract_values()['Harvesting'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1 
    plt.plot(Dataframe.Separation, label = 'Separation Tank Levels', color = colors[cnt])
    plt.axhline(Instance.TankHeight.extract_values()['Separation'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.legend()
    plt.ylabel('Tank Water Levels \n [m$^3$]')
    
    # ----- Plot Rates 
    plt.subplot(313)
    cnt = 0
    plt.plot(Dataframe.SH, label = 'Harvest Rate', color = colors[cnt])
    plt.axhline(Instance.DischargeLimit.extract_values()['SH'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    cnt += 1
    plt.plot(Dataframe.SD, label = 'Public Outflow Rate', color = colors[cnt])
    plt.axhline(Instance.DischargeLimit.extract_values()['SD'], xmin = 0, xmax = 1, color = colors[cnt], linestyle = ':', alpha = 0.3)
    plt.legend()
    plt.ylabel('Discharge Volumes \n per timestep [m$^3$/s]')
    plt.xlabel('Simulation Timestep ID')
    


def PrintModelFormulation():
    print("MODEL FORMULATION: \n\
------------------\n \
    max( Hh_K )      <- Final harvested volume \n\
        \n\
Subject to \n\
    OF_(k+1)/As + Hs_(k+1) = Hs_k + (R_k + OF_k - SH_k - SD_k - SD2_k)/As   for k != K      <- Separation Tank Mass Balance \n\
    Hh_(k+1) = Hh_k + SH_k/Ah                       for k != K      <- Harvesting Tank Mass Balance   \n\
    Hd_(k+1) = Hd_k + (SD_k + SD2_k - DO_k)/Ad      for k != K      <- Detention Tank Mass Balance \n\
    \n\
    OF_(k+1) >= As * Hs_k + R_k + OF_k - SH_k - SD_k - SD2_k - As * Hs_max      <- Lower bound, controls overflow = True condition. \n\
    OF_(k+1) <= As * Hs_k + R_k + OF_k - SH_k - SD_k - SD2_k - As * Hs_max \    <- Upper bound, effective when overflow = True \n\
                    - (SH_max + SD_max + SD2_max + As * H_max * phi_k) \n\
                        + (SH_max + SD_max + SD2_max + As * H_max * phi_k) \n\
    OF_k <= e * phi_k       \n\
    \n\
    TODO: Condflows")