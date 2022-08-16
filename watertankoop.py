# -*- coding: utf-8 -*-
"""
Object declaration and functions file for watertank and opening objects.

Changes from watertankoop.py: Changes made 2 August 2021
    - watertank objects contain a dictionary of connected outlets
    - outlets on initialisation appends a reference to itself to its source tank
    - watertank massbalance method checks that outflows are below existing tank volumes. (Check #2)
    - outlet computeflow method checks that discharge is below source tank volumes.(Check #1)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

# ========== TANKS ==========
class watertank:
    """
    Class for a tank object.
    INPUTS
        name : tank identifier
        area : area of tank, assumed to be constant [m^2]
        height : tank height [m]
        initlevel : initial water level in tank [m]
    """

    # ---------- Instance attributes ----------
    def __init__(self, name, area, height, initlevel):
        self.name = name            # Tank identifier
        self.area = area            # Tank parameters
        self.tkhght = height

        self.level = [initlevel]    # Set initial tank levels [m]
        self.overflow = [0]         # Set initial overflow volume [m^3]
        self.outlets = {}           # Reference for outlets.
        self.inlets = {}            # Reference for inlets.
        if area != None:
            self.volume = [initlevel * self.area]
            self.maxvol = self.tkhght * self.area

    def __str__(self):              # For print(<object name>) output
        return "This is the " + self.name + " tank, a watertank instance"

    # ---------- Class methods -----------
    def L(self):
        '''Access stored water levels'''
        return np.array(self.level)

    def OF(self):
        '''Access stored overflow volumes'''
        return np.array(self.overflow)

    def V(self):
        '''Access stored volume array'''
        return np.array(self.volume)

    def Info(self):
        return self.__dict__

    def ClassName(self):
        return self.__class__

    def massbalance(self, inflows, outflows):   # Tank dynamics discretised
        """
        Calculates new water level based on the total inflow and outflow volumes
        INPUTS
            inflow : inflow volume [m^3]
            flows : scalar or array of outflow volumes [m^3]
        OUTPUTS
            h : new waterlevel
        """
        # ----- Check if total outflow is less than water available from previous timestep
        outflowtotal = np.sum(outflows)
        availvol = self.level[-1] * self.area

        if outflowtotal > availvol:         # If outflow is more than water available
            # Calculate new discharge volumes
            limoutflowdict = {k: outlet.q[-1]/outflowtotal * availvol for k, outlet in self.outlets.items()}
            for k, v in limoutflowdict.items():     # Overwrite new values into referenced outlet objects
                self.outlets[k].q[-1] = v

            outflows = list(limoutflowdict.values())    # Use new outflows for mass balance

        # ----- Mass balance
        deltav = np.sum(inflows) - np.sum(outflows)     # Calculate change in volume, [m^3]
        h = self.level[-1] + deltav/self.area   # Determine new tank water level, [m]

        # ----- Saturate to 0 if h is somehow a negative value
        if h < 0:           # Ensure water level found is a positive value
            h = 0           # Force 0 if value is negative

        # ------ Saturate to tankheight in case of overflow.
        if h > self.tkhght:                             # Check for overflow
            overflow = (h - self.tkhght) * self.area    # Calculate volume if there is overflow
            h = self.tkhght                             # Set tank water level to maximum possible value
        else:
            overflow = 0    # No overflow volume if the water level is below tank height

        self.level.append(h)
        self.overflow.append(overflow)
        self.volume.append(h*self.area)

class vartank(watertank):
    """
    Child class of WATERTANK that allows for variable tank area. Representation primarily in volume
    rather than the tank height.
    INPUTS
        name : tank identifier, string
        hvfxn : lambda function for the tank height in terms of volume
        maxvol : maximum volume of tank [m^3]
        initvol : initial tank volume [m^3]
    """
    def __init__(self, name, hvfxn, maxvol, initvol, area = None):
        self.maxvol = maxvol
        self.hvfxn = hvfxn
        watertank.__init__(self, name, area, hvfxn(self.maxvol).item(), hvfxn(initvol).item())
        self.volume = [initvol]

    def massbalance(self, inflows, outflows):
        """
        hvfxn should be an anonymous function the height as a function of volume v
        """
        # ----- Check if total outflow is less than water available from previous timestep
        outflowtotal = np.sum(outflows)
        availvol = self.volume[-1]

        if outflowtotal > availvol:         # If outflow is more than water available
            # Calculate new discharge volumes
            limoutflowdict = {k: outlet.q[-1]/outflowtotal * availvol for k, outlet in self.outlets.items()}
            for k, v in limoutflowdict.items():     # Overwrite new values into referenced outlet objects
                self.outlets[k].q[-1] = v

            outflows = list(limoutflowdict.values())    # Use new outflows for mass balance

        deltav = np.sum(inflows) - np.sum(outflows)     # Calculate change in volume
        v = self.volume[-1] + deltav                    # Calculate new volume

        if v < 0:       # Ensure volume is positive
            v = 0       # If negative, saturate both height and volume to 0
            h = 0
        else:
            h = self.hvfxn(v).item()       # Calculate height of water in tank based on volume in tank

        if v > self.maxvol:         # Ensure volume calculated is within max
            overflow = (v - self.maxvol)
            v = self.maxvol         # Set to tank maximum
            h = self.tkhght

        else:
            overflow = 0


        self.level.append(h)
        self.overflow.append(overflow)
        self.volume.append(v)
# -----
class systemext:
    """
    Arbitrary class for storing outflows from the system.
    """
    def __init__(self, name, sourcetank):
        self.name = name
        self.inlets = {}
        self.volume = []

    def __str__(self):
        return "This is the " + self.name + " system output."

    def V(self):
        '''Access stored volume array'''
        return np.array(self.volume)

    def massbalance(self, inflows):
        self.volume.append(inflows)

# ========== OPENINGS ==========
class opening:
    """
    Parent class for flows out of tanks.
    INPUTS
        name : opening identifier
        sourcetank : flow source tank
        destank : flow destination tank
    """
    Cd = 0.6    # Discharge coefficient
    g = 9.81    # Gravitational constant

    def __init__(self, name, sourcetank, destank):
        self.name = name
        self.source = sourcetank
        self.destination = destank

        self.q = []         # storage array for flow volumes
        self.source.outlets[name] = self    # Add outlet object reference to tank object
        if isinstance(destank, str) is False:
            self.destination.inlets[name] = self    # Add outlet object reference to tank object.

        #self.source.neighbour

    def Q(self):
        '''Accesses the flow volume storage array'''
        return np.array(self.q)

    def Info(self):
        return self.__dict__

    def ClassName(self):
        return self.__class__

    def computeflow(self):     # Ensures that this method is written
        '''Main flow calculation function, to be written by individual opening types'''
        raise NotImplementedError()  # This prompts for derived classes to overwrite this function

# ----- CHILD CLASSES OF OPENINGS -----
class pump(opening):
    """
    Class for a pump object.
    INPUTS
        name : object identifier
        maxrate : maximum pump rate (m3/s)
        minrate : minimum pump rate (m3/s)
        sourcetank : source tank object
        destank : destination tank (object, or string)
    """
    def __init__(self, name, maxrate, minrate, sourcetank, destank):
        self.rmax = maxrate
        self.rmin = minrate
        opening.__init__(self, name, sourcetank, destank)

    def __str__(self):
        return('This pump ' + self.name + ' joins the ' +
                self.source + ' tank and the '+ self.destination + ' tank')

    # ----- Control methods
    def relay(self, onpt, offpt, tank = None):
        """
        Implements a two-level on-off controller.
        INPUTS
            tank : source tank object
            onpt : value at which to open the orifice
            offpt : value at which to close the orifice
        """
        if tank is None:            # Sets default value to object tank source
            tank = self.source

        if tank.level[-1] > onpt:       # Value at which to open the orifice
            r = self.rmax
        elif tank.level[-1] < offpt:    # Value at which to close the orifice
            r = self.rmin
        else:       # Use previous value if the water level is in between the on/off points.
            r = self.q[-1]
        self.q.append(r)
        return r


class orifice(opening):
    """
    A small circular passive orifice.
    PARAMETERS
        name : orifice identifier
        height : vertical position of orifice in tank
        area : orifice opening area
        sourcetank : flow source tank object
        destank : flow destination tank object or string name
    """
    def __init__(self, name, height, amax, sourcetank, destank):
        self.height = height
        self.amax = amax
        self.area = []      # Storage array for orifice opening area
        opening.__init__(self, name, sourcetank, destank)

    def __str__(self):
        return('This orifice ' + self.name + ' is a passive orifice joining the ' +
                self.source.name + ' tank and the '+ self.destination.name + ' tank')

    def A(self):
        '''Accesses the changes in area '''
        return np.array(self.area)

    def computeflow(self, tank = None):
        """
        Calculates flow out of the orifice for the given tank, per second. [m^3/s]
        INPUTS
            tank : source tank object
        OUTPUTS
            __discharge : outflow volume per second
        """
        if tank is None:
            tank = self.source
        self.__effH = tank.level[-1] - self.height      # Calculate head above orifice
        if self.__effH < 0:         # Check if water level is above orifice height
            discharge = 0           # No discharge if level not above orifice height
        else:                       # Calculate discharge according to given orifice equation
            discharge = self.Cd * self.amax * np.sqrt(2*self.g*self.__effH)

        if discharge > tank.level[-1] * tank.area:      # Check that this is below what is available
            discharge = tank.level[-1] * tank.area

        self.q.append(discharge)
        self.area.append(self.amax)
        return discharge    # in m^3/s

class weir(opening):
    """
    A rectangular weir.
    PARAMETERS
        name : weir identifier
        height : vertical position of orifice in tank
        length : length of weir. Should be the same as tank width it is located on.
        sourcetank : flow source tank object
        destank : flow destination tank object or name
    """
    def __init__(self, name, height, length, sourcetank, destank):
        self.height = height
        self.length = length
        opening.__init__(self, name, sourcetank, destank)

    def __str__(self):
        return('This weir ' + self.name + ' is a passive weir joining the '
               + self.source.name + ' tank and the ' + self.destination.name + ' tank.')

    def computeflow(self, tank = None):
        """
        Calculates flow out of the weir for the given tank, per second. [m^3/s]
        INPUTS
            tank : source tank object
        OUTPUTS
            __discharge : outflow volume per second
        """
        if tank is None:
            tank = self.source
        self.__effH = tank.level[-1] - self.height      # Calculate head above weir
        if self.__effH < 0:     # Check if water level is above weir height
            discharge = 0       # No discharge if level not above weir height
        else:                   # Calculate discharge according to given weir equation
            discharge = 2/3 * self.Cd * self.length * np.sqrt(2 * self.g) * (self.__effH)**(3/2)

        if discharge > tank.level[-1] * tank.area:      # Check that this is below what is available
            discharge = tank.level[-1] * tank.area

        self.q.append(discharge)
        return discharge

class controlled(orifice):
    """
    Controlled orifice, inherits from ORIFICE class.
    PARAMETERS
        name : orifice identifier
        height : vertical position of orifice in tank
        amax : maximum orifice opening area
        amin : minimum orifice opening area
        sourcetank : flow source tank object
        destank : flow destination tank object or string
    """
    def __init__(self, name, height, amax, amin, sourcetank, destank):
        orifice.__init__(self, name, height, amax, sourcetank, destank)
        self.amin = amin
        self.counter = 0    # For use with a timer function

    def __str__(self):
        return ('This orifice ' + self.name + ', a controlled orifice joining the ' +
                self.source + ' tank and the '+ self.destination + ' tank')

    def computeflow(self, a, tank = None):
        """
        Calculates flow out of the controlled orifice for the given tank, per second. [m^3/s]
        INPUTS
            tank : source tank object
            a : area of orifice. Can be calculated using one of the controller methods available
        """
        if tank is None:
            tank = self.source

        # ----- Store the provided orifice area value
        if a > self.amax:       # Check value is within upper bound
            a = self.amax
        elif a < self.amin:     # Check value is within lower bound
            a = self.amin
        self.area.append(a)     # Store value

        # ------ Calculate volume out of orifice in this second
        self.__effH = tank.level[-1] - self.height
        if self.__effH < 0:
            discharge = 0
        else:
            discharge = self.Cd * a * np.sqrt(2*self.g*self.__effH)

        if discharge > tank.level[-1] * tank.area:      # Check that this is below what is available
            discharge = tank.level[-1] * tank.area

        self.q.append(discharge)    # Store flow rate into storage array

        return discharge

    # ----- Controller methods -----
    # All controller methods should return the calculated orifice size a
    def relay(self, onpt, offpt, tank = None):
        """
        Implements a two-level on-off controller.
        INPUTS
            tank : source tank object
            onpt : value at which to open the orifice
            offpt : value at which to close the orifice
        """
        if tank is None:
            tank = self.source

        if tank.level[-1] > onpt:       # Value at which to open the orifice
            a = self.amax
        elif tank.level[-1] < offpt:    # Value at which to close the orifice
            a = self.amin
        else:       # Use previous value if the water level is in between the on/off points.
            a = self.area[-1]
        return a


    def pcontrol(self, refsig, k, tank = None):
        """
        Implements a Proportional controller.
        INPUTS
            tank : source tank object
            refsig : reference signal value
            k : controller gain
        """
        if tank is None:
            tank = self.source

        e = refsig - tank.level[-1]     # Calculate error value
        a = - k * e                     # Calculate corresponding orifice opening area
        return a


    def timer(self, flag, ontime, tank = None):
        """
        Implements a timer that closes the orifice for a given amount of time.
        INPUTS
            tank : tank source object
            flag : signal to monitor for starting the timer
            ontime : amount of time for the
        """
        if tank is None:
            tank = self.source

        if flag > 0:                # Flag for timer to start
            a = self.amin           # Close the orifice
            if self.counter == 0:   # Set counter to given time if this is the first instance
                self.counter = ontime
            else:                   # Else reduce the counter by 1 timestep
                self.counter = self.counter - 1
        else:
            if self.counter == 0:   # Check if there is an active counter
                a = self.amax       # Open orifice if the counter is inactive
            else:
                a = self.amin       # Else close the orifice and reduce counter by 1
                self.counter = self.counter - 1
        return a

    def passive(self):
        """
        Tests an uncontrolled orifice. To remove the need to change object
        classes when comparing different controllers.
        """
        a = self.amax       # orifice is always open to its maximum
        return a

    def exttimer(self, objref, ontime = 7 * 86400, tank = None):
        """
        Uses an external signal to trigger a timer function.
        """
        flag = 0
        if tank is None:
            tank = self.source

        if len(objref.Q()) < ontime: # if simulation time currently has been less than limit
            a = 0

        elif flag == 0:
            if np.sum(objref.Q()[-ontime:]) == 0:  #qtrt has not been used in the past x days
                a = self.amax
                flag = 1
            else:
                a = 0
        else:
            if self.source.level[-1] != 0:
                a = self.amax
            else:
                a = 0
                flag = 0

        return a

    def firstflush(self, idx, tank = None):
        """
        Implements a controller that looks to stop rainwater harvesting for the
        first 10 minutes of a rainfall event only. Passive operations thereafter.
        """
        if tank is None:
            tank = self.source

        if len(tank.L()) <= idx + 600:
            a = 0
        else:
            a = self.amax

        return a


# ----- External optimised signal orifice
class optorifice(orifice):
    """
    Orifice controlled using an external optimisation signal.
    PARAMETERS
        name : orifice identifier
        height : vertical position of orifice in tank
        amax : maximum orifice opening area
        intsignal : external discharge or area signal
        mode : string indicator for intsignal in terms of orifice discharge 'Q',
                or orifice area 'A'
        sourcetank : flow source tank object
        destank : flow destination tank object or string
    """
    def __init__(self, name, height, amax, intsignal, mode, sourcetank, destank):
        self.infile = intsignal[1:]              # Keep a copy of the input file
        self.mode = mode
        orifice.__init__(self, name, height, amax, sourcetank, destank)

        self.qmax = self.Cd * self.amax * np.sqrt(2*self.g*(self.source.tkhght - self.height))

        if self.mode == 'A':            # Input signal is an orifice area signal
            self.q = []
            self.area = intsignal
        elif self.mode == 'Q':          # Input signal is the orifice discharge signal
            self.area = np.array([])
            self.q = intsignal

    def optimflow(self, i):
        """
        This is the computeflow equivalent function for an optimised orifice.
        INPUTS:
            i : timestep index
        """
        if self.mode == 'Q':

            if self.q[i] > self.qmax: # Caps the outflow to the maximum allowable.
                self.q[i] = self.qmax

            if self.q[i] > self.source.volume[-1]: # Limits outflow to what is available in the tank
                self.q[i] = self.source.volume[-1]

            return self.q[i]

        elif self.mode == 'A':  # This is the computeflow equivalent function
            a = self.area[i]
            if a > self.amax:   # Check that the orifice area is within bounds
                a = self.amax
            self.__effH = self.source.level[-1] - self.height

            if self.__effH < 0:
                discharge = 0
            else:
                discharge = self.Cd * a * np.sqrt(2*self.g*self.__effH)

                if discharge > self.source.level[-1] * self.source.area:      # Check that this is below what is available
                    discharge = self.source.level[-1] * self.source.area
                self.q.append(discharge)
            return discharge

    def computeflow(self, i):
        """ Alias function. """
        return optimflow(self, i)

    def computearea(self, tank = None):
        """
        This should be called when the simulation is over. This means under
        this system, the controller does not take into account amax.
        """
        if tank is None:
            tank = self.source
        self.__effH = tank.L() - self.height    # Calculate height above orifice
        self.area = np.where(self.__effH <= 0, self.q / (self.Cd * np.sqrt(2*self.g*self.__effH)))

        return self.area

    def A(self):
        """ Returns the area adjustments. """
        return np.array(self.area)

# # ===== FUNCTIONS
def GenerateConfigurationFigure(TankObjects, Openings):
    """
    Generates a graph object for visualising the relationships between the initiated objects.
    INPUTS
        TankObjects : LIST. All tank objects to be plotted
        Openings : LIST. All openings to be plotted
    """
    G = nx.DiGraph()

    for tank in TankObjects:
        G.add_node(tank.name)
    for o in Openings:
        if isinstance(o.destination, str) is True:
            G.add_node(o.destination)
            G.add_edge(o.source.name, o.destination)
        else:
            G.add_edge(o.source.name, o.destination.name)

    plt.figure()
    plt.title('System Configuration Generated')
    pos=graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, arrows=True, \
            alpha = 0.7, node_size = 1000, width = 2, node_shape = 's', node_color="skyblue", )
            #bbox=dict(facecolor="None", edgecolor='Grey', boxstyle='Square,pad=0.3', alpha = 0.5)
    return G

# def RunSimulation(TankObjects, RainSignal):
#     simulationtime = len(RainSignal)
#
#     for i in range(simulationtime):
#         for t in TankObjects:
#             # Calculate Inflows
#
#             # Calculate Outflows


# ========== ADDITIONAL FUNCTIONS ==========
def outflowlim(outflows, tank):
    """
    This checks that the outflows are lower than the volume available in the tank.
    """
    tout = np.sum(outflows)
    if tout > tank.level[-1] * tank.area:
        nq = np.zeros(len(outflows))
        for f in range(len(outflows)):     # FOR loop for adaptibility in number of outflows
            nq[f] = outflows[f]/tout * tank.level[-1]
            outflows = nq
    return outflows

def reqvol(area, wrc = 1, rct = 0.55):
    """
    Calculates the required detention tank volume for a given catchment area,
    INPUTS:
        area : area of catchment [ha]
        wrc : weighted runoff coefficient of catchment
        rct : target runoff coefficient.
    """
    C = 8913

    # Time of concentration, as defined in the PUB design guide
    if area < 2.0:
        tc = 5
    elif area < 6.0:
        tc = 10
    else:
        tc = 15

    i_10 = C/(tc + 36)                  # Empirical formula for 10-year return storm in Singapore
    qpeak = wrc * i_10 * area / 360     # Peak discharge from catchment
    qallowed = rct * i_10 * area / 360  # Allowable discharge from catchment

    k1 = C * wrc * area / 6
    k2 = C * rct * area / 12
    k3 = tc + 36

    txm = -k3 + np.sqrt(36 * k1 * k3 / k2)
    txl = (wrc - rct) * k3 / rct

    tx = min(txl, txm)
    vol = k1*(tc + tx)/(k3 + tx) - k2*(2*tc + tx)/k3

    return vol, qpeak, qallowed


# ----- PLOTTING FUNCTIONS
def plotq(objlist):
    """
    Plots the discharge rates of given
    """
    colors = sns.color_palette()
    cnt = 0
    for obj in objlist:
        # Check class.
        plt.plot(obj.Q(), label = obj.name + ' discharge rate', color = colors[cnt])
        cnt += 1
    plt.ylabel('Flow rates \n[m$^3$/s]')
    plt.legend()

def plota(objlist, mode = 'abs'):
    """
    Plots area of orifices. Useful for when there is more than one single controlled orifice.
    Available modes:
        'abs' - plots absolute values
        'percentage' - plots opening percentage
    """
    colors = sns.color_palette()
    cnt = 0
    for obj in objlist:
        if mode == 'abs':
            plt.plot(obj.A(), label = obj.name + ' orifice opening area', color = colors[cnt])
            plt.axhline(obj.amax, xmin = 0, xmax = 1, color = colors[cnt],  linestyle = '--', alpha = 0.6)
        elif mode == 'percentage':
            plt.plot(obj.A()/obj.amax, label = obj.name + ' orifice opening area', linestyle = '--', color = colors[cnt], alpha = 0.6 )
        cnt += 1
    plt.ylabel('Orifice opening areas \n[m$^2$]')
    plt.legend()

def plothv(objlist, mode = 'height'):
    """
    Plots height or volume of water in tanks with its maximum capacity.
    """
    # this needs to plot height, and max height. Give option of plotting V instead as well?
    colors = sns.color_palette()
    cnt = 0
    if mode == 'height':
        for obj in objlist:
            plt.plot(obj.L(), label = obj.name, color = colors[cnt])
            plt.axhline(obj.tkhght, xmin = 0, xmax = 1, color = colors[cnt], linestyle = '--', alpha = 0.5)
            cnt += 1
        plt.ylabel('Tank Water Levels \n[m]')

    elif mode == 'volume':
        for obj in objlist:
            plt.plot(obj.V(), label = obj.name, color = colors[cnt])
            plt.axhline(obj.maxvol, xmin = 0, xmax = 1, color = colors[cnt], linestyle = '--', alpha = 0.5)
            cnt += 1
        plt.ylabel('Tank Volumes \n [m$^3$]')

    elif mode == 'overflow':
        for obj in objlist:
            plt.plot(obj.OF(), label = obj.name)
    else:
        raise TypeError()

    plt.legend()
