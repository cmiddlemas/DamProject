import numpy as np

def swarmDefaultParams():
    """
        PSO Equation of State:
            v[n+1] = Chi*(A*v[n] + B*(pbest - x[n]) + C*(lbest - x[n]))
            where [A,B,C] are given by [INERTIA, PBESTC, LBESTC]

        Parameters values taken from Bratton & Kennedy 1997 "Defining a Standard for PSO"

        NEIGHBORHOOD= f(swarmX,swarmF)->swarmL
            Returns the lbest of each particle given its position and fitness value
    """
    defaultParams={'TSTOP':200,'NP':50,'CHI':0.719,'INERTIA':1,'PBESTC':2.05,'LBESTC':2.05,'IMASS':0.9,'FMASS':0.5,'NEIGHBORHOOD':ring}
    return defaultParams

def fitDefaultParams():
    """
        Gives parameters of the fitness function.
        Fitness Function parameters:
        DIMENSIONS=finite number of dimensions (N)
        MAPFUNC=a 1:1 mapping from [0,1)^N to parameter space
        FITFUNC=takes in a parameter space vector and returns a scalar to be minimized.
    """
    defaultParams={'DIMENSIONS':3,'MAPFUNC':defaultMapping,'FITFUNC':defaultFitness}
    return defaultParams

def fitDefaultParams2():
    """
        Fitness function: L1 norm of the difference |x - y| in 2D
        
    """

    defaultParams={'DIMENSIONS':2,'MAPFUNC':defaultMapping,'FITFUNC':ellipse}
    return defaultParams

def defaultMapping(unitParams):
    return unitParams * 2

def defaultFitness(params):
    return np.inner(params,params)

def l1diff(params):
    return np.abs(params[0] - params[1])

def ellipse(params):
    params[1] = 10*params[1]
    return np.inner(params,params)
def ring(swarmX,swarmF):
    
    """ Implement the ring topology
    """

    NP = len(swarmX)
    swarmL = np.full(swarmX.shape,0) #Preallocate the lbest array
    #First handle edge cases
    #print(np.argmin([swarmF[-1],swarmF[ 0],swarmF[1]])-1)
    swarmL[0]  = swarmX[(np.argmin([swarmF[-1],swarmF[ 0],swarmF[1]])-1)%NP]
    swarmL[NP-1] = swarmX[(np.argmin([swarmF[-2],swarmF[-1],swarmF[0]])-2)%NP]
    
    #Handle middle of array
    for i in range(1,NP-1):
        #Populate swarmL with the minimizer of swarmF
        swarmL[i] = swarmX[i-1+np.argmin([swarmF[i-1],swarmF[i],swarmF[i+1]])]
    return swarmL



def pso(swarm=swarmDefaultParams(),fit=fitDefaultParams()):
    """
        Runs the PSO code based on parameters in psoParams dictionary.
        (1) Initializes swarm randomly
        (2) Evolve swarm in time
        (3) Periodic boundary conditions
        (4) Return best found fitness
    """

    #Compose and bind functions into local namespace for speed and concision
    fitnessF = fit['FITFUNC']
    mapF   = fit['MAPFUNC']
    hoodF  = swarm['NEIGHBORHOOD']
    PSOF   = lambda x: fitnessF(mapF(x))

    #Initialize swarm location, position, and pbest
    swarmX = np.random.rand(swarm['NP'],fit['DIMENSIONS'])
    swarmV = np.random.rand(swarm['NP'],fit['DIMENSIONS'])
    swarmP = np.copy(swarmX)
    swarmPF = np.full((swarm['NP'],1),np.inf)
    #DEBUG
    #print(swarmX)
   
    #Evolve swarm in time
    for t in range(0,swarm['TSTOP']):
        #Evaluate fitness and update pbests (swarmP)
        swarmF = list(map(PSOF, swarmX))
        #print(swarmF)
        for i in range(0,swarm['NP']):
            if swarmF[i] < swarmPF[i]:
                swarmP[i] = swarmX[i]
                swarmPF[i] = swarmF[i]
                #Might have to update Gbest
                swarmG = swarmX[np.argmin(swarmF)]

        #Evaluate lbest
        swarmL = hoodF(swarmX, swarmF)

        #DEBUG
        #print(np.hstack((swarmX,swarmL)))

        swarmV = swarm['CHI']*(
            swarm['INERTIA']*swarmV +
            swarm['PBESTC']*(swarmP-swarmX)+
            swarm['LBESTC']*(swarmL-swarmX)
            )
        swarmX += swarmV
        swarmX = np.modf(swarmX)[0]
        swarmXF = np.array([PSOF(x) for x in swarmX])
    return (swarmG,PSOF(swarmG),swarmX,PSOF(swarmX))
def test():
    swarmParams = swarmDefaultParams()
    swarmParams['TSTOP'] = 100
    fitParams = fitDefaultParams2()
    x = pso(swarmParams,fitParams)
    return x

def __main__():
    if __name__ == "__main__":
        test()

