# -*- coding: utf-8 -*-
"""
DEMAND MODELS v2 
================

Created on Fri Jun  4 16:33:35 2021

@author: sohqi
"""

import numpy as np
import pandas as pd


def GenerateDT(StartDT, EndDT, Frequency): 
    Datetime = pd.Series(pd.date_range(StartDT, EndDT, freq = Frequency)) # For scheduled demands. 
    return Datetime 

def GenerateDemand(StartDT, EndDT, Frequency, I = True, B = True, ):
    """
    Returns a dataframe with a demand signal, with datetime information. 
    
    INPUTS:
        StartDT :   str. Datetime string for start of demand signal 
        EndDT :     str. Datetime string for end of demand signal 
        Frequency : str. Timestep size for datetime series. 
        I :         boolean, Default = True. Adds irrigation demand when true. 
        B :         Boolean, Default = True. Adds block washing demand when true. 
    """
    Datetime = GenerateDT(StartDT, EndDT, Frequency)
    
    Total = []
    
    # ----- Irrigation 
    if I is True: 
        Irrigation = GenerateIrrigation(Datetime)
        Total.append(Irrigation)

    # ----- Block Washing 
    if B is True: 
        BW = GenerateBlockWashing(Datetime, 12, 18) # How to change this default? 
        Total.append(BW)
        
    # To add other Demands
    
    # ----- Total 
    TotalDemand = pd.DataFrame({'Datetime': Datetime, 'Demand': sum(Total)})
    
    return TotalDemand


# ----- Functions for scheduled uses
def GenerateBlockWashing(DTArray, WashStartHr = None, WashEndHr = None, \
                         ActiveHours = None, FirstWashDay = None, VolPerDay = 8.0):
    """
    Returns a series with demand associated with estate block washing. 
    Block washing is a scheduled activity, estimated to use 6-8m3/block/month, over 2 days. 
    Implements 2 days of block washing only. 

    INPUTS: 
        DTArray : Datetime series object. 
        WashStartHr : int. 24-hour clock representation of when washing begins. 
        WashEndHr : int. 24-hour clock representation of when washing ends. 
        ActiveHours : list, optional, default = None. Uses WashStartHr and WashEndHr if None. 
        FirstWashDay : Boolean, optional, default = None. Uses first 'Day' entry in DTArray if None. 
        VolPerDay : float, optional, default = 8.0. Volume used per day for block washing. 
    """
    if all(v is None for v in [WashStartHr, WashEndHr, ActiveHours]):
        raise Exception('No hours supplied for block washing.')
        
    Hours = DTArray.dt.hour     # Maps datetime supplied with its corresponding hour
    Day = DTArray.dt.day
    
    # ----- Identify active days
    if FirstWashDay is None: 
        FirstWashDay = Day[0]
        
    ActiveDays = Day.mask((Day == FirstWashDay) | (Day == FirstWashDay+1)).isnull()
    
    # ------ Identify Active Hours 
    if ActiveHours is None: 
        ActiveHours = np.arange(WashStartHr, WashEndHr)
    
    ActiveHours = Hours.isin(ActiveHours)
    ActiveTimesteps = ActiveHours & ActiveDays
    
    # ActiveTimesteps = Hours.mask(((Day == FirstWashDay) | (Day == FirstWashDay+1))\
    #                            & ((WashStartHr <= Hours) & (Hours < WashEndHr))\
    #                                ).isnull()       # Boolean Series of timesteps that are active during washing. 
    
    # ----- Identify actual active timesteps. 
    TotalActiveTimesteps = sum(ActiveTimesteps)
    
    WashRate = 8/TotalActiveTimesteps 
    WashingSignal =  ActiveTimesteps * WashRate
    
    return WashingSignal
    
def GenerateIrrigation(DTArray, IrrStartTime = '19:00:00', EstateArea = 15700, GreenCover = 0.5, \
                       VolPerDay = 0.006, EmittersPerM2 = 2):
    """
    Assumes irrigation occuring everyday. 
    # Different types of plants req different 4-8mm/day (4-8L)
    """
    IrrStartTime = pd.to_datetime(IrrStartTime)
    DeltaT = pd.Timedelta(DTArray[1] - DTArray[0]).total_seconds()
    
    # ----- Total Water Required
    GreenArea = EstateArea * GreenCover
    TotalWaterVol = GreenArea * VolPerDay
    
    # ----- Rates
    RatePerEmitter = 0.002 / 3600       # In m3/s
    N_Emitters = EmittersPerM2 * GreenArea
    TotalSystemRate = N_Emitters * RatePerEmitter   # In m3/s 
    
    # -----
    TotalTimeRequired = TotalWaterVol / TotalSystemRate # in seconds
    IrrEndTime = IrrStartTime + pd.Timedelta(seconds = TotalTimeRequired)
    N_Timesteps = TotalTimeRequired/DeltaT
    
    # ----- Calculate Volumes in each timestep 
    VolPerTimestep = TotalSystemRate * DeltaT 
    PartialVolume = (N_Timesteps - int(N_Timesteps)) * VolPerTimestep
    
    # ----- Apply masks to correct times
    Time = DTArray.dt.time
    
    Full = Time.mask((Time >= IrrStartTime.time()) & \
                     (Time <= (IrrEndTime - pd.Timedelta(seconds = DeltaT)).time())\
                         ).isnull() * VolPerTimestep

    Partial = Time.mask((Time >= (IrrEndTime - pd.Timedelta(seconds = DeltaT)).time()) &\
                        (Time <= IrrEndTime.time())).isnull() * PartialVolume
    
    # ----- Sum total signal 
    IrrigationSignal = Full + Partial
   
    return IrrigationSignal

def GenerateChuteWashing():
    # TODO. 
    ChuteSignal = np.random.normal(size = 288)
    return ChuteSignal 

def GenerateCarWashing():
    CarWashSignal = 1
    return CarWashSignal

def GenerateToiletFlushing():
    # TODO. Maybe use Sainsburys data? 
    ToiletSignal = 1
    return ToiletSignal
    



