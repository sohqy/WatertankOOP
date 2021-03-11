# -*- coding: utf-8 -*-
"""
Function file for handling rainfall input 
"""
import pandas as pd
import numpy as np
import itertools as itr

def read_historicaldata():
    """
    Reads full historical data set for S81, which has rainfall in 5 minute time resolution
    between 2010 and 2018. 
    
    OUTPUTS:
        DATAFRAME with a Datetime index and single 'Rainfall' column. 
    """
    hist_rf = pd.read_csv('full2010-2018data.csv')
    hist_rf.Datetime = pd.to_datetime(hist_rf.Datetime)  # This is the slow part of reading historical data. 
    hist_rf.set_index('Datetime', inplace = True)    
    
    # hist_rf['Date'] = hist_rf.index.date
    # hist_rf['Time'] = hist_rf.index.time
    # hist_rf['Year'] = hist_rf.index.year
    # hist_rf['Month'] = hist_rf.index.month 
    # hist_rf['Day'] = hist_rf.index.day
    # hist_rf['Week'] = hist_rf.index.week
    
    return hist_rf

def extractprofile(date, A, dys = 1):
    """
    Extracts the historical rainfall profile for a stipulated period
    
    INPUTS:
        date :  Date to be extracted. String in the form of 'YYYY-MM-DD'
        A :     Dataframe containing data. 
        dys :   Number of days to be extracted.
        
    OUTPUTS:
        DATAFRAME with columns 'Datetime' and 'Rainfall'.
    """
    d = pd.to_datetime(date)
    df = A[(A.index > d) & (A.index <= (d + pd.Timedelta(days = dys)))].copy()
    df.reset_index(inplace = True)
    
    return df 

def uniform_disaggregation(A, f):
    """
    Disaggregates data into smaller timesteps, assuming uniform distribution within the original timesteps. 
    
    INPUTS: 
        A : LIST-LIKE. Series containing data to be disaggregated.
        f : Scale factor to be used in the disaggregation. 
    
    OUTPUTS:
        LIST of disaggregated data - Datetime information is not kept.
    """
    f = int(f)          # Ensure the scale factor is an integer
    rfs = []            # Storage list
    
    for a in A:         
        scaled = a/f    # Scale each element in input data
        rfs.append(np.ones(f) * scaled)     # Copy this f times to scale
        
    return np.concatenate(rfs)

def padinput(A, f):
    """
        
    """
    f = int(f)
    rfs = [[A[0]]]
    
    for a in A[1:]:
        rfs.append(np.ones(f) * a)
    
    return np.concatenate(rfs)

# ===== ADAPTED FROM RFANALYSIS. NEED TO RECONCILIATE BOTH FILES? 
def prof_maxcont(A):
    """
    A is a list or array containing data for a single rainfall profile. 
    """
    count = 0   
    cmax = 0
    
    for j in A[:-1]:    # For each element in the day, except the last element 
        if j > 0:           # If it is raining
            count += 1          # Add counter
        else:               # If it is not raining
            if count > cmax :   # Check if the recorded length is the longest so far
                cmax = count    # If its longer than previous records, set highest observed count
                count = 0       # Reset count
                
    if A[-1] > 0 and count + 1 > cmax:  # If the last element in the day shows rain, and the latest
        cmax = count + 1                # episode is the longest observed, set highest observed count
    
    return cmax

def episodecounter(A, gap = 10):
    """
    Counts the number of rain episodes in a given rainfall data series based on a defined time gap
    between two rain episodes. 
    INPUTS: 
        x : rainfall time series 
        gap : number of dry timesteps between each episode
    """
       
    # ----- Initialise flags and counters 
    epcounter = 0       # Tracks the number of episodes found
    rainflag = 0        # Flags if the previous data element showed rainfall
    gapcounter = 0      # Counts the number of dry periods to distinguish between two rain episodes
    
    # ----- Loop through data except the final data element 
    for rf in A[:-1]: 
        if rf > 0:              # If it is raining
            if rainflag == 0:   # Is this the first rain element in an episode
                rainflag = 1    # Set the flag
    
                if gapcounter >= gap:   # If there has been a sufficiently long gap between the last found rain element
                    epcounter += 1      # Count this as a new episode
                elif epcounter == 0:    # If this is the first episode encountered within less timesteps that the gap
                    epcounter = 1       # Count episode
                    
            gapcounter = 0          # Reset gap counter
            
        else:                   # If it is not raining
            rainflag = 0        # Clear rain flag
            gapcounter += 1     # Add to gap counter
    
    # ----- Last data element handling
    if A[-1] > 0 :                  
        if rainflag == 0 and gapcounter >= gap:     # If there is rain, check if this belongs to a new episode
                epcounter += 1
    
    return epcounter


def calcrfinfo(A, markers = ['Max', 'Mean', 'Total', 'StD', 'Episodes', 'Wet Periods', 'PCI', 'Max Cont', 'RO Mean', 'RO StD']): # tough luck looks like you got to re-write this shit 
    """
    A is a series. markers for names of markers 
    """
    # generate a dictionary. 
    # Extract using markers
    A = np.array(A) 
    dct = {'Total': np.sum(A), 'Mean': np.mean(A), 'Max': np.max(A), 'StD': np.std(A), 'Episodes':episodecounter(A), \
           'Wet Periods': np.sum(A > 0), 'Max Cont': prof_maxcont(A), 'RO Mean': np.mean(A[np.nonzero(A)]), \
               'RO StD': np.std(A[np.nonzero(A)]), 'PCI': np.sum(A**2)/np.sum(A)**2}
    
    return [dct[k] for k in markers] # Return list of RF info. then concat RF dataframe with rf cols. 
 
# def dividedata(A, i = 1, f ='s'):
#     """
#     Generates a dataframe containing data disaggregated uniformly into smaller timesteps. 
#     A : Dataframe containing data for a single day, in 5 minute timesteps.
#     i : frequency step, integer
#     f : frequency for data to be divided into. string aliases: 'min' or 's' 
#     """
#     rain = A.Rainfall       # A is a df containing extracted data 
#     rfs = []
#     freq = str(i) + f
    
#     if 's' in f:            # Define y, number of smaller timesteps 
#         y = 300 * i         # 300 seconds in 5 minutes
#     else:
#         y = 5               # Else accepts minutes, in which there are 5 in each timestep. 
        
#     for r in rain:
#         rfps = r/y          # Find value of inflow, assuming uniformity within each timestep. 
#         a = np.ones(y)      # Create a y-lengthed list of ones 
#         a = rfps * a        
#         rfs.append(a)

#     rfs = np.concatenate(rfs)
    
#     sdate = A.Date.min()
#     edate = A.Date.max()
#     dt = pd.date_range(start = sdate, end = edate + pd.Timedelta(minutes = 5), freq = freq, closed = 'left')
#     df = pd.DataFrame(data = {'Datetime': dt, 'Rainfall': rfs})
        
#     return df

# ====== STOCHASTIC STEP 1 
def StochProbability(A, bins = None, n=1, output_type = 'Dict'): # WANT THIS TO OUTPUT A DICTIONARY 
    """
    Calculates conditional probability. Probability of given categorised rainfall volume following a specific 
    INPUTS: 
        A: List like. 
            Data containing series or list. 
        bins: LIST, optional
            List containing bin edge definitions.
        n: INTEGER, optional. 
            Number of timesteps before current to take into account for calculating probability. 
    OUTPUTS: 
        
    """
    # ----- Ensure all elements are bounded in bins
    A = A[A >= 0]
    
    # ----- Transform series into categories
    if bins is None:
        bins = DefineStochBins(A)
        
    categorised_series = np.digitize(A, bins, right = True)     # 0 < x <= edge so that 0 can have a bin on its own

    dct = {0: 'Z', 1:'L', 2:'M', 3:'H'}
    categorised_series = [dct[k] for k in categorised_series]   # Map integer category labels into string labels 
    categories = list(dct.values())
    
    # ----- Initialise Counters
    prod = [p for p in itr.product(categories, repeat = n+1)]  # Returns tuples
    counters = np.zeros(len(prod))  # List which elements 
    
    last = len(A) - 1
    
    for index, element in enumerate(categorised_series):
        if index > n-1: 
            tn = tuple(categorised_series[index - n : index + 1])
            
            # Check element category
            idx = prod.index(tn)
            counters[idx] += 1
            
        elif index == last:
            break
    
    pastnames = ['t-' + str(i) for i in reversed(range(n))]
    colnames = pastnames.copy()
    colnames.append('t+1')
    
    df = pd.DataFrame(prod, columns = colnames)
    df['Count'] = counters
    sums = df.groupby(pastnames).sum().to_dict()['Count']
    
    if len(pastnames) > 1:
        df['Past'] = list(zip(*[df[col] for col in df[pastnames]]))
    
    else:
        df['Past'] = df['t-0'].copy()
        
    div = [sums[k] for k in df.Past]
    df['Probability'] = np.divide(counters, div, out = np.zeros_like(div), where = np.array(div) != 0.0)

    if output_type == 'DataFrame':
        results = df.drop(columns = ['Count', 'Past'])
        
    elif output_type == 'Matrix':
        results = df.pivot(index = 'Past', columns = 't+1', values = 'Probability')
    
    else: 
        results = dict(zip(prod, df.Probability))
        
    return results
        
    
def ReduceDataResolution(df, f = '6H',):
    df['Datetime'] = pd.to_datetime(df.Datetime)
    df = df.groupby(pd.Grouper(key = 'Datetime', freq = f)).sum()
    df = df[['Rainfall']]
    return df

def DefineStochBins(A, method = 'Quartiles'):
    """ 
    A is a list. 
    """
    array = A[A>0]
    
    if method == 'Quartiles': # Add other methods. 
        p = [25, 50,]
        
    bins = [0, np.percentile(array, p[0]), np.percentile(array, p[1]),]
    
    return bins
    
def RainScenarioVolumes(A, bins = None): # Output as dict. 
    # Check if A is list. 
    if bins is None:
        bins = DefineStochBins(A)
    bins.append(max(A))
    avg = [(bins[i+1] - bins[i])/2 for i in range(len(bins)-1)]
    avg.insert(0, 0)
    labels = ['Z', 'L', 'M', 'H']
    dct = dict(zip(labels, avg))
    return dct
