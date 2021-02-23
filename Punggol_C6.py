# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:47:07 2020

@author: sohqi
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import watertankoop as wt
import rffxns as rf

def sim(n_timesteps, Qin, methodtorun = None, pos_args = None, extsignal = None):
    """
    A wrapper for the simulation for easier use. 
    INPUTS:
        n_timesteps: Number of simulation timesteps
        Qin: Rainfall input signal
        methodtorun: Controlled class method in the form of wt.controlled.method
        pos_args: Method arguments in a list. 
        extsignal: Alternative to running a controlled class method by supplying an external control signal. 
    """
    # ----- Check if there is a method supplied for the simulation.    
    if methodtorun is None and extsignal is None:
        raise Exception('No method has been supplied. Provide a method to run or an external signal')
    
    # ----- Define TANK components in system
    sep = wt.watertank('Separation', 2.575*1.950, 1.227, 0.0)
    det = wt.watertank('Detention', 264.519, 2.0, 0.0)    # For the tank that is directly attached. 
    trt = wt.watertank('Treated', 5, 2, 0.0)              # This has 10m3 capacity. Not interested in exact fill levels. 
    
    # ----- Define VARTANK components in system
    harvolfx = lambda v: np.piecewise(v, [v <= 48.22, v > 48.22], [v/41.109, (v-5.689*1.173)/35.42])
    har = wt.vartank('Harvesting', harvolfx, 91.68, 0.0)
    
    # ----- Define openings in system
    if extsignal is not None:
        if len(extsignal) - 1 != n_timesteps:
            simsignal = rf.uniform_disaggregation(extsignal, np.round(n_timesteps/len(extsignal))) # If this is from GAMS, need to remove initial value.       
        else:
            simsignal = extsignal.copy()    
        q1 = wt.optorifice('q1',0, np.pi*(0.05**2), simsignal,'Q', sep, det)
        
    else:
        q1 = wt.controlled('q1', 0, np.pi*(0.05**2), 0, sep, det)
    
    q2 = wt.orifice('q2', 0.52, np.pi*(0.175**2), sep, har)
    q3 = wt.weir('q3', 0.9, 0.66, sep, har,) # This formulation doesnt consider edge effects
    
    qhd = wt.controlled('RWH Drainage', 0, np.pi*(0.1**2), 0, har, det) # This is a timed gate
    qtrt = wt.pump('treatment', 3.5/3600, 0, har, trt)
    
    qout1 = wt.orifice('out', 0, np.pi*(0.175**2), det, 'Public')
    qout2 = wt.orifice('out overflow', 1.8, np.pi*(0.125**2), det, 'Public')
            
    # ----- Simulation loop 
    for i in range(n_timesteps):    # Want to be able to adjust this as little as possible between options. 
        sepinflows = [Qin[i], sep.overflow[-1], har.overflow[-1], det.overflow[-1]]
        if extsignal is None:
            sepoutflows = [q1.computeflow(methodtorun(q1, *pos_args)), q2.computeflow(), q3.computeflow()]
        else:
            sepoutflows = [q1.optimflow(i), q2.computeflow(), q3.computeflow()]
    
        detinflows = [sepoutflows[0], sepoutflows[2], 0]
        detoutflows = [qout1.computeflow(), qout2.computeflow()]
        
        harinflows = [sepoutflows[1], trt.overflow[-1]]
        haroutflows = [qtrt.relay(1.7, 0.5), detinflows[2]]
        
        sep.massbalance(sepinflows, sepoutflows)
        det.massbalance(detinflows, detoutflows)
        har.massbalance(harinflows, haroutflows)
        trt.massbalance(haroutflows[0], 0)

    return {'sep':sep, 'det':det, 'trt':trt, 'har':har, 'q1':q1, 'q2':q2,\
            'q3':q3, 'qhd':qhd, 'qtrt':qtrt, 'qout1':qout1, 'qout2':qout2}

def get_systemparams(simobj): # Put into watertankoop?
    # TO DO: Add overall system params!! E.g. control, which outlet controlled, 
    # Check object type for more generic function. 
    tanks = ['sep', 'det', 'trt', 'har']
    outlets = ['q1', 'q2', 'q3', 'qhd', 'qtrt', 'qout1', 'qout2']
    
    tnkidx = []
    tnkmaxvol = []
    tnkheight = []
    
    outidx = []
    outpos = []
    outarea = []
    
    for k in simobj:
        if k in tanks:
           tnkidx.append(k)
           tnkmaxvol.append(simobj[k].maxvol)
           tnkheight.append(simobj[k].tkhght)
            
        if k in outlets: 
            outidx.append(k)
            outpos.append(simobj[k].height)
            outarea.append(simobj[k].amax) # Add TYPE of outlet, Control type 
            
    tnk_df = pd.DataFrame({'Name': tnkidx, 'Max Vol': tnkmaxvol, 'Tank Height': tnkheight})
    out_df = pd.DataFrame({'Name': outidx, 'Max Area': outarea, 'Position': outpos})
    
    return {'Tank Info': tnk_df, 'Outlet Info': out_df}
    

def print_systemparams(simobj): # Put this into watertankoop? 
    """
    Prints a pretty(ish) console output for the system parameters extracted using the get_systemparams function. 
    """
    info = get_systemparams(simobj)
    tnks = info['Tank Info'].set_index('Name', inplace = True)
    outs = info['Outlet Info'].set_index('Name', inplace = True)
    
    print('========================= \n Simulated Parameters \n=========================')
    print(' 1. TANKS \n' + " " * 10 + str(tnks).replace('\n', '\n          '))
    print(' 2. OUTLETS \n' + ' ' * 10 + str(outs).replace('\n','\n          '))

    
# ----- KPIs 
def calckpi(simobjs, n_timesteps):
    sep, det, qout1, qout2, har, trt = (simobjs.get(key) for key in ['sep', 'det', 'qout1', 'qout2', 'har', 'trt'])
    overflowrisk = np.sum(sep.level)/(n_timesteps * sep.tkhght) 
    qoutmax = max(qout1.Q()+qout2.Q())
    unused = sep.area * (sep.tkhght - max(sep.level)) + det.area * (det.tkhght - max(det.level))
    hvsted = har.volume[-1] + trt.volume[-1]
    #sepovfl = np.sum(sep.overflow)
    sepovfl = max(sep.overflow)
    
    #output = [overflowrisk, qoutmax, unused, hvsted, sepovfl]
    #print('Overflow risk ', overflowrisk, '\nDischarge ', qoutmax, '\nUnused capacity ', unused, '\nHarvested ', hvsted, '\nsepoverflow', sepovfl)
    
    return overflowrisk, qoutmax, unused, hvsted, sepovfl

def plotobj(simobjs, title):
    sep, det, qout1, qout2, har, trt, q1, q2, q3 = \
        (simobjs.get(key) for key in ['sep', 'det', 'qout1', 'qout2', 'har', 'trt', 'q1', 'q2', 'q3'])
    plt.figure()
    plt.subplot(211)
    wt.plothv([sep, det, har, trt])
    plt.subplot(212)
    wt.plotq([q1, q2, q3, qout1, qout2])
    plt.suptitle(title)

def plotgamsresults(df):
    plt.figure()
    ax1 = plt.subplot(211)
    df.plot('timestep', ['S', 'D', 'H'], ax = ax1)
    ax1.set_xlabel('')
    ax1.set_ylabel('Tank Water Levels \n[m]')
    ax2 = plt.subplot(212)
    df.plot('timestep', ['1','2','3', ], ax = ax2)
    ax2.set_xlabel('Timestep ID')
    ax2.set_ylabel('Flow rates \n[m$^3$/s]')

# def extractprofile(date, A, dys = 1):
#     """
#     Extracts the historical rainfall profile for a stipulated period
#     date :  Date to be extracted. String in the form of 'YYYY-MM-DD'
#     A :     Dataframe containing data. 
#     dys :   Number of days to be extracted.
#     """
#     d = pd.to_datetime(date)
#     df = A[(A.index > d) & (A.index <= (d + pd.Timedelta(days = dys)))].copy()
#     df.reset_index(inplace = True)
    
#     rain = df.Rainfall
#     prof = wt.divinput(rain, 300) 
    
#     return  prof    
    
def sim_params(n_timesteps, Qin, sf = 1, init_s = 0.0, init_d = 0.0, init_h = 0.0, o_pos = None, o_size = None, methodtorun = None, pos_args = None, extsignal = None, tnkhghts = None, tnkareas = None):
    """
    Wrapper for a more general parameter input simulation. 
    INPUTS: 
        n_timesteps: INTEGER. Number of timesteps in simulation.
        Qin: PANDAS SERIES OBJECT. Rainfall signal. 
        sf: FLOAT, optional, default value 0.0. Scale factor for adjusting the overall system capacity. 
        init_s: FLOAT, optional, default value 0.0. Initial fill percentage for the separation tank.
        init_d: FLOAT, optional, default value 0.0. Initial fill percentage for the detention tank.
        init_h: FLOAT, optional, default value 0.0. Initial fill percentage for the harvesting tank.
        o_pos: DICTIONARY, optional, defaults to PC6 values. Dictionary containing positions of openings in the system. 
        o_size: DICTIONARY, optional, defaults to PC6 values. Dictionary containing sizes of openings in the system. 
        methodtorun: METHOD. Control method for a single controlled orifice. 
        pos_args: LIST. Arguments required for METHOD called.
        ext_signal: SERIES, optional. External control signal. 
        tnkhghts: DICTIONARY, optional, defaults to PC6 values. 
        tnkareas: DICTIONARY, optional, defaults to PC6 values. 
    """
    if methodtorun is None and extsignal is None:
        raise Exception('No method has been supplied')
     
    # Define default values to case study PC6 system
    if tnkhghts == None: 
        tkhghts = {'sep': 1.227 * sf, 'det': 2.0 * sf, 'trt': 2.0}
    if tnkareas == None:
        tnkareas =  {'sep': 5.02125 * sf, 'det': 264.519 * sf, 'trt': 5.0}
    if o_pos == None:
        o_pos = {'Q1': 0.0, 'Q2': tkhghts['sep'] * (0.52/1.227), 'Q3': tkhghts['sep'] * (0.66/1.227), 'Qout1': 0.0, 'Qout2':  tkhghts['det'] * (1.8/2)}
    if o_size == None: 
        o_size = {'Q1': np.pi*(0.05**2), 'Q2': np.pi*(0.175**2), 'Q3': 0.9, 'Qout1': np.pi*(0.175**2) , 'Qout2': np.pi*(0.125**2)}
    
    # Define Tank parameters 
    sep = wt.watertank('Separation', tnkareas['sep'], tkhghts['sep'], init_s * tkhghts['sep'])
    det = wt.watertank('Detention', tnkareas['det'], tkhghts['det'], init_d * tkhghts['det'])
    trt = wt.watertank('Treated', tnkareas['trt'], tkhghts['trt'], 0.0)

    harvolfx = lambda v: np.piecewise(v, [v <= 48.22, v > 48.22], [v/41.109, (v-5.689*1.173)/35.42])
    har = wt.vartank('Harvesting', harvolfx, 91.68 * (sf**2), init_h * 91.68 * (sf**2) )
    
    # ----- Define openings in system
    if extsignal is not None:
        if len(extsignal) - 1 != n_timesteps:
            simsignal = rf.uniform_disaggregation(extsignal, np.round(n_timesteps/len(extsignal)))                
        else:
            simsignal = extsignal.copy()    
        q1 = wt.optorifice('q1', o_pos['Q1'], o_size['Q1'], simsignal,'Q', sep, det)
        
    else:
        q1 = wt.controlled('q1', o_pos['Q1'], o_size['Q1'], 0, sep, det)

    q2 = wt.orifice('q2', o_pos['Q2'], o_size['Q2'], sep, har)
    q3 = wt.weir('q3', o_pos['Q3'], o_size['Q3'], sep, har,) # This formulation doesnt consider edge effects
    
    qhd = wt.controlled('RWH Drainage', 0, np.pi*(0.1**2), 0, har, det) # This is a timed gate. NOT IMPLEMENTED. 
    qtrt = wt.pump('treatment', 3.5/3600, 0, har, trt)
    
    qout1 = wt.orifice('out', o_pos['Qout1'], o_size['Qout1'], det, 'Public')
    qout2 = wt.orifice('out overflow', o_pos['Qout2'], o_size['Qout2'], det, 'Public')
    
    # ----- Simulation loop 
    for i in range(n_timesteps):    # Want to be able to adjust this as little as possible between options. 
        sepinflows = [Qin[i], sep.overflow[-1], har.overflow[-1], det.overflow[-1]]
        if extsignal is None:
            sepoutflows = [q1.computeflow(methodtorun(q1, *pos_args)), q2.computeflow(), q3.computeflow()]
        else:
            sepoutflows = [q1.optimflow(i), q2.computeflow(), q3.computeflow()]
    
        detinflows = [sepoutflows[0], sepoutflows[2], 0]
        detoutflows = [qout1.computeflow(), qout2.computeflow()]
        
        harinflows = [sepoutflows[1], trt.overflow[-1]]
        haroutflows = [qtrt.relay(1.7/2.46 * har.tkhght, 0.5/2.46 * har.tkhght), detinflows[2]]
        
        sep.massbalance(sepinflows, sepoutflows)
        det.massbalance(detinflows, detoutflows)
        har.massbalance(harinflows, haroutflows)
        trt.massbalance(haroutflows[0], 0)
        
    return {'sep':sep, 'det':det, 'trt':trt, 'har':har, 'q1':q1, 'q2':q2,\
            'q3':q3, 'qhd':qhd, 'qtrt':qtrt, 'qout1':qout1, 'qout2':qout2}
        
def SimParamsQ2(n_timesteps, Qin, sf = 1, init_s = 0.0, init_d = 0.0, init_h = 0.0, o_pos = None, o_size = None, methodtorun = None, pos_args = None, extsignal = None, tnkhghts = None, tnkareas = None):
    """
    Wrapper for a more general parameter input simulation. 
    INPUTS: 
        
    """
    if methodtorun is None and extsignal is None:
        raise Exception('No method has been supplied')
     
    # Define default values to case study PC6 system
    if tnkhghts == None: 
        tkhghts = {'sep': 1.227 * sf, 'det': 2.0 * sf, 'trt': 2.0}
    if tnkareas == None:
        tnkareas =  {'sep': 5.02125 * sf, 'det': 264.519 * sf, 'trt': 5.0}
    if o_pos == None:
        o_pos = {'Q1': 0.0, 'Q2': tkhghts['sep'] * (0.52/1.227), 'Q3': tkhghts['sep'] * (0.66/1.227), 'Qout1': 0.0, 'Qout2':  tkhghts['det'] * (1.8/2)}
    if o_size == None: 
        o_size = {'Q1': np.pi*(0.05**2), 'Q2': np.pi*(0.175**2), 'Q3': 0.9, 'Qout1': np.pi*(0.175**2) , 'Qout2': np.pi*(0.125**2)}
    
    # Define Tank parameters 
    sep = wt.watertank('Separation', tnkareas['sep'], tkhghts['sep'], init_s * tkhghts['sep'])
    det = wt.watertank('Detention', tnkareas['det'], tkhghts['det'], init_d * tkhghts['det'])
    trt = wt.watertank('Treated', tnkareas['trt'], tkhghts['trt'], 0.0)

    harvolfx = lambda v: np.piecewise(v, [v <= 48.22, v > 48.22], [v/41.109, (v-5.689*1.173)/35.42])
    har = wt.vartank('Harvesting', harvolfx, 91.68 * (sf**2), init_h * 91.68 * (sf**2) )
    
    # ----- Define openings in system
    if extsignal is not None:
        if len(extsignal) - 1 != n_timesteps:
            simsignal = rf.uniform_disaggregation(extsignal, np.round(n_timesteps/len(extsignal)))                
        else:
            simsignal = extsignal.copy()    
        q2 = wt.optorifice('q2', o_pos['Q2'], o_size['Q2'], simsignal,'Q', sep, har)
        
    else:
        q2 = wt.controlled('q2', o_pos['Q2'], o_size['Q2'], 0, sep, har)

    q1 = wt.orifice('q1', o_pos['Q1'], o_size['Q1'], sep, det)
    q3 = wt.weir('q3', o_pos['Q3'], o_size['Q3'], sep, det) # This formulation doesnt consider edge effects
    
    qhd = wt.controlled('RWH Drainage', 0, np.pi*(0.1**2), 0, har, det) # This is a timed gate. NOT IMPLEMENTED. 
    qtrt = wt.pump('treatment', 3.5/3600, 0, har, trt)
    
    qout1 = wt.orifice('out', o_pos['Qout1'], o_size['Qout1'], det, 'Public')
    qout2 = wt.orifice('out overflow', o_pos['Qout2'], o_size['Qout2'], det, 'Public')
    
    # ----- Simulation loop 
    for i in range(n_timesteps):    # Want to be able to adjust this as little as possible between options. 
        sepinflows = [Qin[i], sep.overflow[-1], har.overflow[-1], det.overflow[-1]]
        if extsignal is None:
            sepoutflows = [q1.computeflow(), q2.computeflow(methodtorun(q2, *pos_args)), q3.computeflow()]
        else:
            sepoutflows = [q1.computeflow(), q2.optimflow(i), q3.computeflow()]
    
        detinflows = [sepoutflows[0], sepoutflows[2], 0]
        detoutflows = [qout1.computeflow(), qout2.computeflow()]
        
        harinflows = [sepoutflows[1], trt.overflow[-1]]
        haroutflows = [qtrt.relay(1.7/2.46 * har.tkhght, 0.5/2.46 * har.tkhght), detinflows[2]]
        
        sep.massbalance(sepinflows, sepoutflows)
        det.massbalance(detinflows, detoutflows)
        har.massbalance(harinflows, haroutflows)
        trt.massbalance(haroutflows[0], 0)
        
    return {'sep':sep, 'det':det, 'trt':trt, 'har':har, 'q1':q1, 'q2':q2,\
            'q3':q3, 'qhd':qhd, 'qtrt':qtrt, 'qout1':qout1, 'qout2':qout2}

def GenerateOptimAsmtReport(results, name = None):
    Har_PercentageImproved = len(results[results['Harvesting Improvement'] > 0])/len(results)
    Har_PercentageDeteriorate = len(results[results['Harvesting Improvement'] < 0])/len(results)
    
    OF_PercentageImproved = len(results[results['OF Improvement'] > 0])/len(results)
    OF_PercentageDeteriorate = len(results[results['OF Improvement'] < 0])/len(results)
    
    # Magnitude of improvements 
    Har_NonZeroMean = np.mean(results[results['Harvesting Improvement'] > 0]['Harvesting Improvement'])
    Har_Mean = np.mean(results['Harvesting Improvement'])
    Har_NonZeroMeanD = np.mean(results[results['Harvesting Improvement'] < 0]['Harvesting Improvement'])
    
    OF_Mean = np.mean(results['OF Improvement'])
    OF_NonZeroMean = np.mean(results[results['OF Improvement'] > 0]['OF Improvement'])
    OF_NonZeroMeanD = np.mean(results[results['OF Improvement'] < 0]['OF Improvement'])
    
    # Tradeoffs 
    NZ = results[(results[['Harvesting Improvement', 'OF Improvement']].T != 0).any()]
    TOres = results[results != 0].dropna()          # Dataframe of results with values in both categories. 
    OF_only = NZ[NZ['Harvesting Improvement'] == 0]  # Overflow improvement only
    Har_only = NZ[NZ['OF Improvement'] == 0]          # Harvesting improvement only
    
    # WRITE OUTPUT FILE 
    if name is None:
        file = open('SimulationsSummary.txt', 'a') 
    else:
        file = open(str(name), 'a')
        
    file.write('====================\n SIMULATIONS REPORT \n====================\n')
    
    if 'Capacity Scale' in results.columns: 
        file.write('Capacity Scale = ' + str(results['Capacity Scale'].unique()) + '\n')
    
    file.write('Number of Simulations = ' + str(len(results)) +'\n\n')
    file.write('1. HARVESTED YIELD \n------------------\n' + 'Average Difference: ' + str("{:.2f}".format(Har_Mean)) + ' m^3 \n\n'\
               +'\tImprovements = ' + str(len(results[results['Harvesting Improvement'] > 0])) + ' (' + str("{:.2f}".format(Har_PercentageImproved * 100)) + '%)\n'\
                   + '\t\tNon-Zero Average Improvement: ' + str("{:.2f}".format(Har_NonZeroMean)) + ' m^3 \n'\
                  
               + '\tDeteriorations = ' + str(len(results[results['Harvesting Improvement'] < 0])) + ' (' + str("{:.2f}".format(Har_PercentageDeteriorate* 100)) + '%)\n' \
                   + '\t\tNon-Zero Average Deterioration: ' + str("{:.2f}".format(Har_NonZeroMeanD)) + ' m^3 \n'\
                   +'\n')
    
    file.write('2. OVERFLOW REDUCTION \n---------------------\n' + 'Average Difference: ' + str("{:.2f}".format(OF_Mean)) + ' m^3 \n\n'\
               +'\tImprovements = ' + str(len(results[results['OF Improvement'] > 0])) + ' (' + str("{:.2f}".format(OF_PercentageImproved* 100)) + '%)\n'\
                   + '\t\tNon-Zero Average Improvement: ' + str("{:.2f}".format(OF_NonZeroMean)) + ' m^3 \n'\
                  
               + '\tDeteriorations = ' + str(len(results[results['OF Improvement'] < 0])) + ' (' + str("{:.2f}".format(OF_PercentageDeteriorate* 100)) + '%)\n' \
                   + '\t\tNon-Zero Average Deterioration: ' + str("{:.2f}".format(OF_NonZeroMeanD)) + ' m^3 \n'\
                   +'\n')
    
    file.write('3. COMBINATIONS \n------------- \nBoth improved = ' + str(len(TOres[(TOres[['Harvesting Improvement', 'OF Improvement']] > 0).all(axis =1)])) + ' Scenarios\n' \
               + 'Both deteriorated = ' + str(len(TOres[(TOres[['Harvesting Improvement', 'OF Improvement']] < 0).all(axis =1)])) + ' Scenarios\n' \
               + '\nSingular Improvements: \n' \
                   + '\tOverflow Improvement Only = ' + str(len(OF_only[OF_only['OF Improvement'] > 0])) + ' Scenarios \n'\
                   + '\tOverflow Deteriorations only = ' + str(len(OF_only[OF_only['OF Improvement'] < 0])) + ' Scenarios \n'\
                   + '\tHarvesting Improvements only = ' + str(len(Har_only[Har_only['Harvesting Improvement'] > 0])) + ' Scenarios \n'\
                   + '\tHarvesting Deteriorations only = ' + str(len(Har_only[Har_only['Harvesting Improvement'] < 0])) + ' Scenarios \n'\
               + '\nTrade-Offs: \n' \
                   + '\tHarvesting Prioritised: ' + str(len(TOres[(TOres['Harvesting Improvement'] > 0) & (TOres['OF Improvement'] < 0)])) + ' Scenarios \n' \
                   + '\tOverflow Prioritised: ' + str(len(TOres[(TOres['Harvesting Improvement'] < 0) & (TOres['OF Improvement'] > 0)])) + ' Scenarios \n')
    file.close()
    
def plot_PercentageCurve(R):
    Hu = []
    Ou = []
    Hd = []
    Od = []
    
    for c in R['Capacity Scale'].unique():
        results = R[R['Capacity Scale'] == c]
        Hu.append(len(results[results['Harvesting Improvement'] > 0])/len(results))
        Hd.append(len(results[results['Harvesting Improvement'] < 0])/len(results))
        
        Ou.append(len(results[results['OF Improvement'] > 0])/len(results))
        Od.append(len(results[results['OF Improvement'] < 0])/len(results))
    
    plt.figure()
    plt.plot(R['Capacity Scale'].unique(), Hu, marker = 'o', label = 'Improved Harvesting')
    plt.plot(R['Capacity Scale'].unique(), Ou, marker = 'o', label = 'Improved Overflow Reduction')
    plt.plot(R['Capacity Scale'].unique(), Hd, marker = 'x', label = 'Deteriorated Harvesting')
    plt.plot(R['Capacity Scale'].unique(), Od, marker = 'x', label = 'Overflow Increased')
    plt.legend()
    plt.xlabel('Capacity Scale')
    plt.ylabel('Percentage of Scenarios')    
    
def plot_AverageResponse(Ropt, Rpas, grp = 'Capacity Scale', indicator = 'Separation OF Volume'):
    plt.figure()
    plt.plot(Ropt.groupby(grp)[indicator].mean(), marker = 'o', label = 'Optimised Response')
    plt.plot(Ropt.groupby(grp)[indicator].max(), marker = 'o', label = 'Optimised Response Max', alpha = 0.5)
    plt.plot(Rpas.groupby(grp)[indicator].mean(), marker = 'x', label = 'Passive Response')
    plt.plot(Rpas.groupby(grp)[indicator].max(), marker = 'x', label = 'Passive Response Max', alpha = 0.5)
    plt.xlabel('Capacity Scale')
    plt.ylabel(indicator)
    plt.legend()
    