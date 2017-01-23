# -*- coding: utf-8 -*-
"""
Simple, stupid model for linked dams. Assumes constant flow rate.
Handles tributaries and spillways, integrates by forward stepping.

Author: Tim Middlemas
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib as mpl
import itertools
import rainfallData as rf

class dam():
    def __init__(self, 
                 V, 
                 Vmax, 
                 CV, 
                 CI, 
                 overflow,
                 R, 
                 min_cap = 0.0,
                 outflow = 0.0, 
                 overflow_rate = 0.0,
                 underflow = 0.0,
                 underflow_rate = 0.0):
        self.V = V #current volume of water in dam
        self.Vmax = Vmax #maximum carrying capacity of dam
        self.CV = CV #Constant for Volume Proportion term
        self.CI = CI #Constant for inflow term
        self.R = R #for tributaries only, amount of water coming out
        self.min_cap = min_cap # minimum capacity (vol) of the dam that should be filled at all times
        self.outflow = outflow #keeps track of the amount of wate flowing out
        self.overflow = overflow # amount by which vol exceeds capacity
        self.overflow_rate = overflow_rate # rate by which vol is exceeding capacity
        self.underflow = underflow # amount by which vol is under min_cap
        self.underflow_rate = underflow_rate # rate by which vol is under min_cap
        
def control(dam, vol, t, dt):
    #Determine outflow, assuming no over or underflow
    if type(dam.R) == float:
        rate = dam.R
        #print('constant rate')
    else:
        rate = dam.R(t)
    flowVol = dam.V*dam.CV*dt + dam.CI*vol + rate*dt

    #add in upstream flow
    dam.V += vol

    #take the appropriate water out of dam
    dam.V -= flowVol
    #handle when volume of the dam tries to go negative (emptyflow)
    if dam.V < 0.0:
        #print('No water in dam',dam.V)
        out = flowVol + dam.V        
        dam.V = 0.0
        dam.outflow = out
        if out < 0.0:
            #print('Upstream',flowVol)
            dam.outflow = 0.0
            return 0.0
        else:
            return out
        dam.underflow += dam.min_cap # update underflow since inadequate water
        dam.underflow_rate = dam.min_cap 
        return out
    #handle overflow
    elif dam.V > dam.Vmax:
        #print('Overflow:',dam.V)
        out = dam.V - dam.Vmax
        dam.V = dam.Vmax
        dam.overflow += out # update overflow since too much water
        dam.overflow_rate = out
        dam.outflow = out + flowVol
                #Don't let dam flow upstream
        if out+flowVol < 0.0:
            #print('Upstream:',dam.V)
            dam.outflow = out
            return out
        else:
            return out + flowVol
    else:
    #normal operation
        if flowVol < 0.0:
            flowVol = 0.0
        if dam.V < dam.min_cap:
            dam.underflow += dam.min_cap - dam.V # update underflow since inadequate water
            dam.underflow_rate = dam.min_cap - dam.V

        dam.outflow = flowVol
        return flowVol

def step(damTree, t, dt):
    #Base Case, resolve flow:
    if len(damTree) == 1:
        return control(damTree[0], 0.0, t, dt)
    #Else Case, recursively resolve flow
    else:
        #Add the flows from the child nodes
        vol = sum(map(lambda x: step(x, t, dt), damTree[1:]))
        return control(damTree[0],vol, t, dt)
                
def extract_vol(damList):
    return map(lambda x: x.V, damList)
    
def extract_overflow(damList):
    return map(lambda x: x.overflow, damList)

def extract_underflow(damList):
    return map(lambda x: x.underflow, damList)

def extract_overflow_rate(damList):
    return map(lambda x: x.overflow_rate, damList)
    
def extract_underflow_rate(damList):
    return map(lambda x: x.underflow_rate, damList)
    
def extract_outflow(damList,dt):
    # returns outflow divided by dt to get flow in km^3/year
    return map(lambda x: x.outflow/dt, damList)
        
def make_plots(times,vols,names):
    mpl.pyplot.plot(times,vols)
    nDams = len(vols[0])
    #mpl.pyplot.legend(map(lambda x: str(x), range(nDams)))
    mpl.pyplot.legend(names)
def run_simulation(damTree,dt,nSteps,damList,damNames):
    t = 0.0
    vols = []
    times = []
    outflows = []
    overflows = []
    overflow_rates = []
    underflows = []
    underflow_rates = []
    allouts = []

    for i in range(nSteps):
        vols += [extract_vol(damList)]
        times += [t]
        outflows += [step(damTree,t,dt)/dt]
        overflows += [extract_overflow(damList)]
        overflow_rates += [extract_overflow_rate(damList)]
        underflows += [extract_underflow(damList)]
        underflow_rates += [extract_underflow_rate(damList)]
        allouts += [extract_outflow(damList,dt)]
        t += dt
        
    vols += [extract_vol(damList)]
    times += [t]
    overflows += [extract_overflow(damList)]
    overflow_rates += [extract_overflow_rate(damList)]
    underflows += [extract_underflow(damList)]
    underflow_rates += [extract_underflow_rate(damList)]
    allouts += [extract_outflow(damList,dt)]
        

    # Make plot of vol of each dam over Time
    mpl.pyplot.figure(0)
    #mpl.pyplot.title('Volume of Each Dam vs Time')
    mpl.pyplot.ylabel('Volume (km^3)',fontsize=18)
    mpl.pyplot.xlabel('Time (years)',fontsize=18)
    make_plots(times,vols,damNames)
    
    # Make plot of kariba outflow over time
    mpl.pyplot.figure(1)
    mpl.pyplot.title('Outflow of Kariba Dam vs Time')
    mpl.pyplot.ylabel('Outflow (km^3/year)')
    mpl.pyplot.xlabel('Time (years)')
    mpl.pyplot.scatter(times[:-1],outflows)
    
    # Make plot of each dam overflow over time
    mpl.pyplot.figure(2)
    mpl.pyplot.title('Overflow of Each Dam vs Time')
    mpl.pyplot.ylabel('Volume (km^3)')
    mpl.pyplot.xlabel('Time (years)')
    make_plots(times,overflows,damNames)
    
    # Make plot of end outflow per step of system over time
    mpl.pyplot.figure(3)
    mpl.pyplot.title('Outflow of System at Each Timestep vs Time')
    mpl.pyplot.ylabel('Outflow (km^3/year)')
    mpl.pyplot.xlabel('Time (years)')
    make_plots(times,allouts,damNames)
    
    # Make plot of total underflow over time
    mpl.pyplot.figure(4)
    mpl.pyplot.title('Underflow of Each Dam vs Time')
    mpl.pyplot.ylabel('Volume (km^3)')
    mpl.pyplot.xlabel('Time (years)')
    make_plots(times,underflows,damNames)
    
    return

def evaluate_overflow(damTree,dt,nSteps,damList):
    for i in range(nSteps):
        step(damTree, t,dt)
    return damTree[0].overflow
    
#Specify data here
    
tVol = 100000000.0
tCap = 1000000000000.0

C1 = 0.1 # reservoirs per year to let out
dam_max_vol = 25.0

min_cap_percent = 0.50
dam_min_cap = dam_max_vol*min_cap_percent

def C2(R,C1):
    return 1.0 - (20.0*C1)/R

#Define dams with initial parameters
# We take flow rates by using mean yearly rate in m^3/s from map one
# Rates are now in km^3/yr

condition = 'normal' # 'normal' or 'drought' 
flooding = True # True or False

averageGrid = np.linspace(10,11) # takes a few years to reach steady state...also avoid flood when there is one.
steadyStateFlows = np.array([
                    rf.getFlow('kariba',condition, flood = flooding)(averageGrid),
                    rf.getFlow('victoria',condition, flood = flooding)(averageGrid),
                    rf.getFlow('8',condition, flood = flooding)(averageGrid),
                    rf.getFlow('9',condition, flood = flooding)(averageGrid),
                    rf.getFlow('10',condition, flood = flooding)(averageGrid),
                    rf.getFlow('11',condition, flood = flooding)(averageGrid),
                    rf.getFlow('12',condition, flood = flooding)(averageGrid),
                    rf.getFlow('13',condition, flood = flooding)(averageGrid)])
meanFlow = np.mean(steadyStateFlows,axis=1)

"""Code for making figures of flow estimates

plt.plot(averageGrid,rf.getFlow('kariba',condition, flood = flooding)(averageGrid),label='Kariba')
plt.plot(averageGrid,rf.getFlow('victoria',condition, flood = flooding)(averageGrid),label='Victoria')
plt.plot(averageGrid,rf.getFlow('8',condition, flood = flooding)(averageGrid),label='Subbasin 8')
plt.plot(averageGrid,rf.getFlow('9',condition, flood = flooding)(averageGrid),label='Subbasin 9')
plt.plot(averageGrid,rf.getFlow('10',condition, flood = flooding)(averageGrid),label='Subbasin 10')
plt.plot(averageGrid,rf.getFlow('11',condition, flood = flooding)(averageGrid),label='Subbasin 11')
plt.plot(averageGrid,rf.getFlow('12',condition, flood = flooding)(averageGrid),label='Subbasin 12')
plt.plot(averageGrid,rf.getFlow('13',condition, flood = flooding)(averageGrid),label='Subbasin 13')
#plt.legend(fontsize=18)
plt.ylabel('Flow Rate (' + condition + ') km^3 / yr',fontsize=18)
plt.xlabel('One year',fontsize=18)

"""


kariba = dam(20.0, dam_max_vol, C1, C2(np.sum(meanFlow),C1), 0.0, 0.0,dam_min_cap) 
tKariba = dam(tVol,tCap, 0.0,0.0,0.0, rf.getFlow('kariba',condition, flood = flooding)) 
 
victoria = dam(20.0, dam_max_vol, C1, C2(np.sum(meanFlow[1:]),C1), 0.0,0.0,dam_min_cap)
tVictoria = dam(tVol,tCap,0.0,0.0,0.0, rf.getFlow('victoria',condition, flood = flooding)) 
 
d8 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[2],C1), 0.0, 0.0,dam_min_cap)
tD8 = dam(tVol,tCap,0.0,0.0,0.0, rf.getFlow('8',condition, flood = flooding)) 
 
d9 = dam(20.0,  dam_max_vol, C1, C2(np.sum(meanFlow[3:]),C1), 0.0, 0.0,dam_min_cap)
tD9 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('9',condition, flood = flooding)) 
 
d10 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[4],C1), 0.0, 0.0,dam_min_cap)
tD10 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('10',condition, flood = flooding)) 
 
d11 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[5],C1), 0.0, 0.0,dam_min_cap)
tD11 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('11',condition, flood = flooding)) 
 
d12 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[6],C1), 0.0, 0.0,dam_min_cap)
tD12 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('12',condition, flood = flooding))
 
d13 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[7],C1), 0.0, 0.0,dam_min_cap)
tD13 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('13',condition, flood = flooding))
 
#Define dam topology and provide dam list
dTree = [kariba,[victoria,[d8,[tD8]],[d9,[d10,[tD10]],[d11,[tD11]],[d12,[tD12]],[d13,[tD13]],[tD9]],[tVictoria]],[tKariba]]
dList = [kariba,victoria,d8,d9,d10,d11,d12,d13]



inflow = dam(10000000.0,1000000000000.0,0.0,0.0,0.0,1.0)
a = dam(0.0,10.0,0.05,0.1,0.0,0.0,0.05)
b = dam(0.0,10.0,0.05,0.1,0.0,0.0,0.05)
testTree = [b,[a,[inflow]]]
testList = [a,b]


def plot_energy():
    oArray = np.zeros((10,10))
    for i in range(10):
        print i
        for j in range(10):
            inflow = dam(10000000.0,1000000000000.0,0.0,0.0,0.0,1.0)
            a = dam(0.0,10.0,0.01*i,0.1*j,0.0,0.0)
            b = dam(0.0,10.0,0.01*i,0.1*j,0.0,0.0)
            testTree = [b,[a,[inflow]]]
            testList = [a,b]
            oArray[i][j] = evaluate_overflow(testTree,0.1,10000,testList)
    mpl.pyplot.imshow(oArray,cmap = mpl.cm.coolwarm,interpolation = 'nearest')
    print oArray[0][0]
    print oArray[1][0]
    print oArray[0][1]

def get_overflow(L):
    return map(lambda x: x.overflow, L)    
    
def get_underflow(L):
    return map(lambda x: x.underflow, L)
    
def get_outflow(L):
    return map(lambda x: x.outflow, L)

def get_data_for_energy(function,dt,nSteps,T,L):
    t = 0.0
    data = []
    for i in range(nSteps):
        step(T,t,dt)
        t+=dt
        data += [function(L)]
    return data
    
def energy_overunder(data):
    return max(data[-1])

def energy_out(atypicalData, normalData):
    atypical = np.array(atypicalData[:][0])
    normal = np.array(normalData[:][0])
    return np.sum(np.square(atypical - normal))
    
<<<<<<< HEAD
def initialize_dams(C1,condition,flood=flooding):
    print C1
    kariba = dam(20.0, dam_max_vol, C1, C2(np.sum(meanFlow), C1), 0.0, 0.0,dam_min_cap) 
    tKariba = dam(tVol,tCap, 0.0,0.0,0.0, rf.getFlow('kariba',condition, flood = flood)) 
    
    victoria = dam(20.0, dam_max_vol, C1, C2(np.sum(meanFlow[1:]), C1), 0.0, 0.0,dam_min_cap)
    tVictoria = dam(tVol,tCap,0.0,0.0,0.0, rf.getFlow('victoria',condition,flood=flooding)) 
    
    d8 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[2], C1), 0.0, 0.0,dam_min_cap)
    tD8 = dam(tVol,tCap,0.0,0.0,0.0, rf.getFlow('8',condition, flood = flood)) 
    
    d9 = dam(20.0,  dam_max_vol, C1, C2(np.sum(meanFlow[3:]), C1), 0.0, 0.0,dam_min_cap)
    tD9 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('9',condition, flood = flood)) 
    
    d10 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[4], C1), 0.0, 0.0,dam_min_cap)
    tD10 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('10',condition, flood = flood)) 
    
    d11 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[5], C1), 0.0, 0.0,dam_min_cap)
    tD11 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('11',condition, flood = flood)) 
    
    d12 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[6], C1), 0.0, 0.0,dam_min_cap)
    tD12 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('12',condition, flood = flood))
    
    d13 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[7], C1), 0.0, 0.0,dam_min_cap)
    tD13 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('13',condition, flood = flood))
=======
flooding = True

def initialize_dams(C1,condition):
    print C1
    kariba = dam(20.0, dam_max_vol, C1, C2(np.sum(meanFlow), C1), 0.0, 0.0,dam_min_cap) 
    tKariba = dam(tVol,tCap, 0.0,0.0,0.0, rf.getFlow('kariba',condition, flood = flooding)) 
    
    victoria = dam(20.0, dam_max_vol, C1, C2(np.sum(meanFlow[1:]), C1), 0.0, 0.0,dam_min_cap)
    tVictoria = dam(tVol,tCap,0.0,0.0,0.0, rf.getFlow('victoria',condition, flood = flooding)) 
    
    d8 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[2], C1), 0.0, 0.0,dam_min_cap)
    tD8 = dam(tVol,tCap,0.0,0.0,0.0, rf.getFlow('8',condition, flood = flooding)) 
    
    d9 = dam(20.0,  dam_max_vol, C1, C2(np.sum(meanFlow[3:]), C1), 0.0, 0.0,dam_min_cap)
    tD9 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('9',condition, flood = flooding)) 
    
    d10 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[4], C1), 0.0, 0.0,dam_min_cap)
    tD10 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('10',condition, flood = flooding)) 
    
    d11 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[5], C1), 0.0, 0.0,dam_min_cap)
    tD11 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('11',condition, flood = flooding)) 
    
    d12 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[6], C1), 0.0, 0.0,dam_min_cap)
    tD12 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('12',condition, flood = flooding))
    
    d13 = dam(20.0,  dam_max_vol, C1, C2(meanFlow[7], C1), 0.0, 0.0,dam_min_cap)
    tD13 = dam(tVol,tCap,0.0,0.0,0.0,rf.getFlow('13',condition, flood = flooding))
>>>>>>> 95664729d92f4849563c343a9f91c860afd4083c
    
    #Define dam topology and provide dam list
    dTree = [kariba,
                [victoria,
                    [d8,
                        [tD8]],
                    [d9,
                        [d10,
                            [tD10]],
                        [d11,[tD11]],
                        [d12,[tD12]],
                        [d13,[tD13]],
                        [tD9]],
                    [tVictoria]],
                [tKariba]
             ]
    dList = [kariba,victoria,d8,d9,d10,d11,d12,d13]
    dNames= ['kariba','victoria','8','9','10','11','12','13']
    return [dTree,dList,dNames]
    
    
def compute_energy_surface(C1start, C1step, nC1, dt, nSteps):
    energyArray = np.zeros(nC1)
    couplingArray = np.zeros(nC1)
    for i in range(nC1):
        #Make dams w/ correct coupling constant        
        [T,L,N] = initialize_dams(C1start + i*C1step,'normal')
        #Run a simulation on the dam, and extract necessary data for computing energy
        normalData = get_data_for_energy(get_outflow,dt,nSteps,T,L)
        #Make dams w/ correct coupling constant        
        [T,L,N] = initialize_dams(C1start + i*C1step,'drought')
        #Run a simulation on the dam, and extract necessary data for computing energy
        atypicalData = get_data_for_energy(get_outflow,dt,nSteps,T,L)
        #Reduce that data using an energy function
        energyArray[i] = energy_out(atypicalData, normalData)
        couplingArray[i] = C1start + i*C1step
    #plot the energy
    mpl.pyplot.figure(0)
    mpl.pyplot.title('Energy vs. Coupling')
    mpl.pyplot.plot(couplingArray,energyArray)
    return energyArray
    

if __name__ == '__main__':
    # auto-runs the larger test sim
    run_simulation(dTree,10.0/365.0,365,dList, dNames)
    # auto runs the smaller (2 dam) test sim
    #run_simulation(testTree,10/365.0,500,testList)
    # auto runs the energy surface sim
    #testArray = compute_energy_surface(0.0,1.0,10,10.0/365.0,365)
    pass
    
