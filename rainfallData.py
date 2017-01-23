import pandas as pd
import numpy as np
from numpy import matlib
from scipy import interpolate
rainfall = pd.read_excel(open("basin_flow_data.xlsx","rb"), sheetname='Sheet1')
table = rainfall.values
byRegion = table.transpose()[1:,0:12]

Nyears = 200
byRegion = byRegion *  31536000 / 1000**3. # Convert from m^3 / s to km^3 / year
byRegion = matlib.repmat(byRegion,1,Nyears) # Generate Nyear time series
[normal8,drought8,normal9,drought9,normal10,drought10,normal11,drought11,normal12,drought12,normal13,drought13,normal6,drought6,normalvictoria,droughtvictoria,normalkariba,droughtkariba] = byRegion # Read off each region...
years = np.arange(Nyears * 12) / 12.

normal8 = interpolate.UnivariateSpline(years,normal8,k=3,s=0)
normal9 = interpolate.UnivariateSpline(years,normal9,k=3,s=0)
normal10 = interpolate.UnivariateSpline(years,normal10,k=3,s=0)
normal11 = interpolate.UnivariateSpline(years,normal11,k=3,s=0)
normal12 = interpolate.UnivariateSpline(years,normal12,k=3,s=0)
normal13 = interpolate.UnivariateSpline(years,normal13,k=3,s=0)
normalvictoria = interpolate.UnivariateSpline(years,normalvictoria,k=3,s=0)
normalkariba = interpolate.UnivariateSpline(years,normalkariba,k=3,s=0)

drought8 = interpolate.UnivariateSpline(years,drought8,k=3,s=0)
drought9 = interpolate.UnivariateSpline(years,drought9,k=3,s=0)
drought10 = interpolate.UnivariateSpline(years,drought10,k=3,s=0)
drought11 = interpolate.UnivariateSpline(years,drought11,k=3,s=0)
drought12 = interpolate.UnivariateSpline(years,drought12,k=3,s=0)
drought13 = interpolate.UnivariateSpline(years,drought13,k=3,s=0)
droughtvictoria = interpolate.UnivariateSpline(years,droughtvictoria,k=3,s=0)
droughtkariba = interpolate.UnivariateSpline(years,droughtkariba,k=3,s=0)

flood = np.zeros(years.shape)
fwhm = 222./365. # 222 day flood
maxFlow = 16000.* 31536000 / 1000**3 # in km^3 / yr
offset = 4.51 # wettest time of year in March...flood the fourth year...

# Allow for a flood in Year 1 of the simulation.
flood = maxFlow * np.exp(-4 * np.log(2) * (years - offset)**2 / fwhm )
floodYrFour = interpolate.UnivariateSpline(years,flood/8.,k=3,s=0)

def getFlow(region='kariba',condition='normal',flood=False):
    if flood:
        totalFlow = interpolate.UnivariateSpline(years,eval(condition + region)(years) + floodYrFour(years),k=3,s=0)
    else:
        totalFlow = eval(condition + region)
    return totalFlow
"""
Print = True
if Print:
    plt.figure()
    plt.title('normalal Conditions')
    plt.plot(years,normal6,label='6')
    plt.plot(years,normal8,label='8')
    plt.plot(years,normal9,label='9')
    plt.plot(years,normal10,label='10')
    plt.plot(years,normal11,label='11')
    plt.plot(years,normal12,label='12')
    plt.plot(years,normal13,label='13')
    plt.plot(years,normalvictoria,label='Victoria')
    plt.plot(years,normalkariba,label='Kariba')
    
if Print:
    plt.figure()
    plt.title('Drought Conditions')
    plt.plot(years,drought6,label='6')
    plt.plot(years,drought8,label='8')
    plt.plot(years,drought9,label='9')
    plt.plot(years,drought10,label='10')
    plt.plot(years,drought11,label='11')
    plt.plot(years,drought12,label='12')
    plt.plot(years,drought13,label='13')
    plt.plot(years,droughtvictoria,label='Victoria')
    plt.plot(years,droughtkariba,label='Kariba')
"""
