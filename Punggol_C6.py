# -*- coding: utf-8 -*-
"""

Created on Sat Jun 19 00:32:14 2021

@author: sohqi
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import watertankoop as wt
import rffxns as rf


def GeneratePC6Objects(ControlMethod, TankHeights = None, TankAreas = None, \
                       InitialVolumes = None, OrificeHeights = None, OrificeAreas = None):
    
    # ===== Define default parameter values
    if TankHeights is None: 
        TankHeights = {'Separation': 1.227, 'Detention': 2.0,  'Treatment': 2.0}
    if TankAreas is None: 
        TankAreas = {'Separation': 5.02125, 'Detention': 264.519, 'Treatment': 5.0}
    if OrificeAreas is None:
        OrificeAreas = {'SD': np.pi*(0.05**2), 'SH': np.pi*(0.175**2), 'SD2': 0.9, 'DO': np.pi*(0.175**2) , 'DO2': np.pi*(0.125**2)}
    if OrificeHeights is None: 
        OrificeHeights = {'SD': 0.0, 'SH': TankHeights['Separation'] * (0.52/1.227), 'SD2': TankHeights['Separation'] * (0.66/1.227), \
                          'DO': 0.0, 'DO2':  TankHeights['Detention'] * (1.8/2)}
    # ----- Initial Volumes 
    if InitialVolumes is None: 
        InitialVolumes = {'Separation': 0.0, 'Detention': 0.0, 'Harvesting': 0.0, 'Treatment': 0.0}
    
    
    # ===== Define Tank Objects 
    HarVolFxn = lambda v: np.piecewise(v, [v <= 48.22, v > 48.22], [v/41.109, (v-5.689*1.173)/35.42])
    
    Sep = wt.watertank('Separation', TankAreas['Separation'], TankHeights['Separation'], InitialVolumes['Separation'] * TankHeights['Separation'])
    Det = wt.watertank('Detention', TankAreas['Detention'], TankHeights['Detention'], InitialVolumes['Detention'] * TankHeights['Detention'])
    Trt = wt.watertank('Treatment', TankAreas['Treatment'], TankHeights['Treatment'], InitialVolumes['Treatment'] * TankHeights['Treatment'])
    Har = wt.vartank('Harvesting', HarVolFxn, 91.68, InitialVolumes['Harvesting'] * 91.68)
    
    # ===== Define Orifice and Weir Objects in system
    if ControlMethod == 'Optimise':
        SD = wt.optorifice('SD', OrificeHeights['SD'], OrificeAreas['SD'], [], 'Q', Sep, Det)
    else:        
        SD = wt.controlled('SD', OrificeHeights['SD'], OrificeAreas['SD'], 0.0, Sep, Det)
        
    SH = wt.orifice('SH', OrificeHeights['SH'], OrificeAreas['SH'], Sep, Har)
    SD2 = wt.weir('SD2', OrificeHeights['SD2'], OrificeAreas['SD2'], Sep, Det)
    
    HD = wt.controlled('H->D', 0, np.pi*(0.1**2), 0, Har, Det) # This is a timed gate. NOT IMPLEMENTED. 
    HT = wt.pump('HT', 3.5/3600, 0, Har, Trt)
    
    DO = wt.orifice('DO', OrificeHeights['DO'], OrificeAreas['DO'], Det, 'Public')
    DO2 = wt.orifice('out overflow', OrificeHeights['DO2'], OrificeAreas['DO2'], Det, 'Public')
        
    FreshwaterUse = []
    
    SimulationObjectsDict = {'Sep': Sep, 'Det': Det, 'Har': Har, 'Trt': Trt, \
                             'SD': SD, 'SH': SH, 'SD2': SD2, 'HD': HD, 'HT': HT,\
                                 'DO': DO, 'DO2': DO2, 'Freshwater': FreshwaterUse}
        
    PrintParams(SimulationObjectsDict)
        
    return SimulationObjectsDict


def SimulateObjects(SimObjs, RainSignal, DemandSignal= None, ExtSignal = None, \
                    ControlMethod = None, pos_args = None,):
    
    if ExtSignal is None and ControlMethod is None:
        raise Exception('No control method is supplied. Please provide an external control signal or wt.controlled class method.')
    elif ExtSignal is not None and ControlMethod is not None: 
        raise Exception('Both an external signal and control method has been provided, please only use one input.')
        
    PrintSimulationParams(RainSignal, DemandSignal, ExtSignal, ControlMethod, pos_args)
    n_timesteps = len(RainSignal)
    
    if ExtSignal is not None:
        idx = len(SimObjs['SD'].infile)
        SimObjs['SD'].infile.extend(ExtSignal)
        SimObjs['SD'].q.extend(ExtSignal)
    
    # ===== Demand and Freshwater Use Set up 
    if DemandSignal is None: 
        DemandSignal = np.zeros(len(RainSignal))
        
    
    for i in range(n_timesteps):
        SepInflows = [RainSignal[i], SimObjs['Sep'].overflow[-1], SimObjs['Har'].overflow[-1], SimObjs['Det'].overflow[-1]]
        
        if ExtSignal is None:
            SepOutflows = [SimObjs['SD'].computeflow(ControlMethod(SimObjs['SD'], *pos_args)), SimObjs['SH'].computeflow(), SimObjs['SD2'].computeflow()]
        else:
            SepOutflows = [SimObjs['SD'].optimflow(idx + i), SimObjs['SH'].computeflow(), SimObjs['SD2'].computeflow()]
    
        DetInflows = [SepOutflows[0], SepOutflows[2], 0] # This is where you insert HD flows.
        DetOutflows = [SimObjs['DO'].computeflow(), SimObjs['DO2'].computeflow()]
        
        HarInflows = [SepOutflows[1], SimObjs['Trt'].overflow[-1]]
        HarOutflows = [SimObjs['HT'].relay(1.7/2.46 * SimObjs['Har'].tkhght, 0.5/2.46 * SimObjs['Har'].tkhght), DetInflows[2]]
        
        if DemandSignal is None: 
            F = 0
        else:
            F = DemandSignal[i] - SimObjs['Trt'].level[-1] 
            if F < 0:   # Set value floor. 
                F = 0
        SimObjs['Freshwater'].append(F)
        
        SimObjs['Sep'].massbalance(SepInflows, SepOutflows)
        SimObjs['Det'].massbalance(DetInflows, DetOutflows)
        SimObjs['Har'].massbalance(HarInflows, HarOutflows)
        SimObjs['Trt'].massbalance([HarOutflows[0], SimObjs['Freshwater'][i]], DemandSignal[i]) 


#%%

def PlotObj(SimObjs, title = ''):
    """
    Plots watertankoop simulation objects. 
    """
    plt.figure()
    plt.subplot(211)
    wt.plothv([SimObjs['Sep'], SimObjs['Det'], SimObjs['Har'], SimObjs['Trt']])
    plt.subplot(212)
    wt.plotq([SimObjs['SD'], SimObjs['SH'], SimObjs['SD2'], SimObjs['DO'], SimObjs['DO2']])
    plt.suptitle(title)
    
def CalcKPIs(SimObjs, ):
    n_timesteps = len(SimObjs['Sep'].level)
    
    SepOverflow = max(SimObjs['Sep'].overflow)
    HarvestedVol = SimObjs['Har'].volume[-1] + SimObjs['Trt'].volume[-1]
    FreshwaterUse = sum(SimObjs['Freshwater'])
    OverflowRisk = np.sum(SimObjs['Sep'].level)/(n_timesteps * SimObjs['Sep'].tkhght)
    UnusedVol = SimObjs['Sep'].area * (SimObjs['Sep'].tkhght -  max(SimObjs['Sep'].level)) + \
        SimObjs['Det'].area * (SimObjs['Det'].tkhght -  max(SimObjs['Det'].level))
    DischargeMax = max(SimObjs['DO'].Q() + SimObjs['DO2'].Q())
    
    KPIDict = {'Overflow': SepOverflow, 'Harvested Volume': HarvestedVol, 'Freshwater Use': FreshwaterUse,\
               'Unused Volume': UnusedVol, 'Overflow Risk': OverflowRisk, 'Discharge Max': DischargeMax}
    
    return KPIDict
    

def PrintParams(SimObjs):
    print('\n \tInitialized WatertankOOP objects with parameters: ')
    print('\t ------------------------------------------------------')
    for k, obj in SimObjs.items():
        if type(obj) != list:
            Params = obj.Info()
            
            if type(obj) == wt.watertank:
                Dct = {'Name': Params['name'], 'Height (m)': Params['tkhght'], 'Area (m2)': Params['area'], 'Class': obj.ClassName()}
            elif type(obj) == wt.vartank:
                Dct = {'Name': Params['name'], 'Capacity (m3)': Params['maxvol'], 'Class': obj.ClassName()}
            elif type(obj) == wt.weir:
                Dct = {'Name': Params['name'], 'Class': obj.ClassName(),'Height (m)': Params['height'], 'Length (m)': Params['length']}
            elif type(obj) == wt.pump:
                Dct = {'Name': Params['name'], 'Class': obj.ClassName(),}
            elif type(obj) == wt.orifice or wt.controlled:
                Dct = {'Name': Params['name'], 'Class': obj.ClassName(),'Height (m)': Params['height'], 'Area (m2)': Params['amax']}

            print('\t\t' + str(Dct['Name']) + ' :\t ' + str(Dct['Class']))
            del Dct['Name']
            del Dct['Class']
            for k,v in Dct.items():
                print('\t\t\t {} {:.4f}'.format(k, v))


def PrintSimulationParams(RainSignal, DemandSignal, ExtSignal, ControlMethod, pos_args):
    # TODO: Add pos args printing. 
    if ExtSignal is not None: 
        print('\n \tRunning an Optimised Controller with input signals:')
    elif ControlMethod is not None: 
        print('\n \tRunning a ' +  ControlMethod.__name__ + ' controller with input signals:')
    print('\t ------------------------------------------------------')
    Signals = {'Rain': RainSignal, 'Demand': DemandSignal, 'Optimal Control': ExtSignal}
    Dct = dict()
    for k,v in Signals.items(): 
        if v is not None:
            SignalLength = len(v)
            Dct[k] = SignalLength
            
    for param, length in Dct.items():
        print('\t\t{} signal length \t {}'.format(param, length))
    
    print('')
        
        
        
        
        
        
        
        
        