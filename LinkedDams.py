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

class dam():
    def __init__(self, V, Vmax, CV, CI, overflow, R, outflow = 0.0):
        self.V = V #current volume of water in dam
        self.Vmax = Vmax #maximum carrying capacity of dam
        self.CV = CV #Constant for Volume Proportion term
        self.CI = CI #Constant for inflow term
        self.overflow = overflow #place holder to compute total overflow
        self.R = R #for tributaries only, amount of water coming out
        self.outflow = outflow #keeps track of the amount of wate flowing out
        
def control(dam, vol, dt):
    #Determine outflow, assuming no over or underflow
    flowVol = dam.V*dam.CV*dt + dam.CI*vol + dam.R*dt
    #add in upstream flow
    dam.V += vol
    #Don't let dam flow upstream
    if flowVol < 0.0:
        dam.outflow = 0.0
        return 0.0
    #take the appropriate water out of dam
    dam.V -= flowVol
    #handle underflow
    if dam.V < 0.0:
        out = flowVol + dam.V        
        dam.V = 0.0
        dam.outflow = out
        return out
    #handle overflow
    elif dam.V > dam.Vmax:
        out = dam.V - dam.Vmax
        dam.V = dam.Vmax
        dam.overflow += out
        dam.outflow = out + flowVol
        return out + flowVol
    else:
        dam.outflow = flowVol
        return flowVol

def step(damTree, dt):
    #Base Case, resolve flow:
    if len(damTree) == 1:
        return control(damTree[0], 0.0, dt)
    #Else Case, recursively resolve flow
    else:
        #Add the flows from the child nodes
        vol = sum(map(lambda x: step(x, dt), damTree[1:]))
        return control(damTree[0],vol, dt)
                
def extract_vol(damList):
    return map(lambda x: x.V, damList)
    
def extract_overflow(damList):
    return map(lambda x: x.overflow, damList)
    
def extract_outflow(damList,dt):
    return map(lambda x: x.outflow, damList)
        
def make_plots(times,vols):
    mpl.pyplot.plot(times,vols)
    nDams = len(vols[0])
    mpl.pyplot.legend(map(lambda x: str(x), range(nDams)))
    
def run_simulation(damTree,dt,nSteps,damList):
    t = 0.0
    vols = []
    times = []
    outflows = []
    overflows = []
    allouts = []
    for i in range(nSteps):
        vols += [extract_vol(damList)]
        times += [t]
        outflows += [step(damTree,dt)/dt]
        overflows += [extract_overflow(damList)]
        allouts += [extract_outflow(damList,dt)]
        t += dt
    vols += [extract_vol(damList)]
    times += [t]
    overflows += [extract_overflow(damList)]
    allouts += [extract_outflow(damList,dt)]
    mpl.pyplot.figure(0)
    make_plots(times,vols)
    mpl.pyplot.figure(1)
    mpl.pyplot.scatter(times[:-1],outflows)
    mpl.pyplot.figure(2)
    make_plots(times,overflows)
    mpl.pyplot.figure(3)
    make_plots(times,allouts)
    return

def evaluate_overflow(damTree,dt,nSteps,damList):
    for i in range(nSteps):
        step(damTree, dt)
    return damTree[0].overflow
    
#Specify data here
    
tVol = 100000000.0
tCap = 1000000000000.0
C1 = 10.0
def C2(R):
    return 1.0 - (20.0*C1)/R

#Define dams with initial parameters
# We take flow rates by using mean yearly rate in m^3/s from map one
# Rates are now in km^3/yr
kariba = dam(19.2, 25.0, C1, C2(40.0+4.4+38.0+2.2+3.6+24.0+8.7+1.04), 0.0, 0.0) 
tKariba = dam(tVol,tCap, 0.0,0.0,0.0, 40.0) 

victoria = dam(19.2, 25.0, C1, C2(4.4+38.0+2.2+3.6+24.0+8.7+1.04), 0.0, 0.0)
tVictoria = dam(tVol,tCap,0.0,0.0,0.0, 4.4) 

d8 = dam(19.0, 25.0, C1, C2(1.04), 0.0, 0.0)
tD8 = dam(tVol,tCap,0.0,0.0,0.0, 1.04) 

d9 = dam(19.0, 25.0, C1, C2(38.0+2.2+3.6+24.0+8.7), 0.0, 0.0)
tD9 = dam(tVol,tCap,0.0,0.0,0.0,38.0) 

d10 = dam(18.0, 25.0, C1, C2(2.2), 0.0, 0.0)
tD10 = dam(tVol,tCap,0.0,0.0,0.0,2.2) 

d11 = dam(19.8, 25.0, C1, C2(3.6), 0.0, 0.0)
tD11 = dam(tVol,tCap,0.0,0.0,0.0,3.6) 

d12 = dam(17.0, 25.0, C1, C2(24.0), 0.0, 0.0)
tD12 = dam(tVol,tCap,0.0,0.0,0.0,24.0) 

d13 = dam(18.0, 25.0, C1, C2(8.7), 0.0, 0.0)
tD13 = dam(tVol,tCap,0.0,0.0,0.0,8.7) 


#Define dam topology and provide dam list
dTree = [kariba,[victoria,[d8,[tD8]],[d9,[d10,[tD10]],[d11,[tD11]],[d12,[tD12]],[d13,[tD13]],[tD9]],[tVictoria]],[tKariba]]
dList = [kariba,victoria,d8,d9,d10,d11,d12,d13]


inflow = dam(10000000.0,1000000000000.0,0.0,0.0,0.0,1.0)
a = dam(0.0,10.0,0.05,0.1,0.0,0.0)
b = dam(0.0,10.0,0.05,0.1,0.0,0.0)
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
    