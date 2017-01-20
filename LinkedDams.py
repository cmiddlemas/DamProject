# -*- coding: utf-8 -*-
"""
Simple, stupid model for linked dams. Assumes constant flow rate.
Handles tributaries and spillways, integrates by forward stepping.

Author: Tim Middlemas
"""

import numpy as np
import matplotlib as mpl
import itertools

class dam():
    def __init__(self, V, R, Vmax):
        self.V = V #current volume of water in dam
        self.R = R #rate of outflow in dam
        self.Vmax = Vmax #maximum carrying capacity of dam
        
def step(damTree,dt):
    #Base Case, resolve flow:
    if len(damTree) == 1:
        flowVol = dt*damTree[0].R
        underflow = flowVol - damTree[0].V
        #Make sure there's enough water in the dam to satisfly flow        
        if underflow > 0.0:
            outflow = damTree[0].V
            damTree[0].V = 0.0
            return outflow
        else:
            damTree[0].V -= flowVol
            return flowVol
    #Else Case, recursively resolve flow
    else:
        #Add the flows from the child nodes
        vol = sum(map(lambda x: step(x, dt), damTree[1:]))
        #Does the dam overflow?
        if vol + damTree[0].V > damTree[0].Vmax :
            return vol
        else:
            damTree[0].V += vol
            flowVol = dt*damTree[0].R
            underflow = flowVol - damTree[0].V
            #Make sure there's enough water in the dam to satisfly flow        
            if underflow > 0.0:
                outflow = damTree[0].V
                damTree[0].V = 0.0
                return outflow
            else:
                damTree[0].V -= flowVol
                return flowVol
                
def extract_vol(damList):
    return map(lambda x: x.V, damList)
        
def make_plots(times,vols):
    mpl.pyplot.plot(times,vols)
    nDams = len(vols[0])
    mpl.pyplot.legend(map(lambda x: str(x), range(nDams)))
    mpl.pyplot.axis([0.0,30.0,0.0,0.7])
    
def run_simulation(damTree,dt,nSteps,damList):
    t = 0.0
    vols = []
    times = []
    for i in range(nSteps):
        vols += [extract_vol(damList)]
        times += [t]
        step(damTree,dt)
        t += dt
    vols += [extract_vol(damList)]
    times += [t]
    make_plots(times,vols)
    return

#Specify data here

#Define dams with initial parameters
a1 = dam(1.0,0.1,1.0)
a2 = dam(1.0,0.1,1.0)
b = dam(0.0,0.1,5.0)
c = dam(0.0,0.0,0.5)

#Define dam topology and provide dam list
dTree = [c,[b,[a1],[a2]]]
dList = [c,b,a1,a2]