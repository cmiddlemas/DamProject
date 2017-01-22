import pandas as pd
import numpy as np
from numpy import matlib
from scipy import interpolate
rainfall = pd.read_excel(open("basin_flow_data.xlsx","rb"), sheetname='Sheet1')
table = rainfall.values
byRegion = table.transpose()[1:,0:12]

Nyears = 20
byRegion = byRegion *  31536000 / 1000**3. # Convert from m^3 / s to km^3 / year
byRegion = matlib.repmat(byRegion,1,Nyears) # Generate Nyear time series
[normal8,drought8,normal9,drought9,normal10,drought10,normal11,drought11,normal12,drought12,normal13,drought13,normal6,drought6,normalvictoria,droughtvictoria,normalkariba,droughtkariba] = byRegion # Read off each region...
years = np.arange(Nyears * 12) / 12.

normal8 = interpolate.interp1d(years,normal8,kind = 'cubic')
normal9 = interpolate.interp1d(years,normal9,kind = 'cubic')
normal10 = interpolate.interp1d(years,normal10,kind = 'cubic')
normal11 = interpolate.interp1d(years,normal11,kind = 'cubic')
normal12 = interpolate.interp1d(years,normal12,kind = 'cubic')
normal13 = interpolate.interp1d(years,normal13,kind = 'cubic')
normalvictoria = interpolate.interp1d(years,normalvictoria,kind = 'cubic')
normalkariba = interpolate.interp1d(years,normalkariba,kind = 'cubic')

drought8 = interpolate.interp1d(years,drought8,kind = 'cubic')
drought9 = interpolate.interp1d(years,drought9,kind = 'cubic')
drought10 = interpolate.interp1d(years,drought10,kind = 'cubic')
drought11 = interpolate.interp1d(years,drought11,kind = 'cubic')
drought12 = interpolate.interp1d(years,drought12,kind = 'cubic')
drought13 = interpolate.interp1d(years,drought13,kind = 'cubic')
droughtvictoria = interpolate.interp1d(years,droughtvictoria,kind = 'cubic')
droughtkariba = interpolate.interp1d(years,droughtkariba,kind = 'cubic')

flood = np.zeros(years.shape)
fwhm = 222./365. # 222 day flood
maxFlow = 16000.* 31536000 / 1000**3 # in km^3 / yr
offset = 0.51 # wettest time of year in March...
for n in range(Nyears):
    flood += maxFlow * np.exp(-4 * np.log(2) * (years - (n + offset))**2 / fwhm )
    
floodAny = interpolate.interp1d(years,flood/8.,kind = 'cubic')

def getFlow(region='kariba',condition='normal'):
    if 'flood' in condition:
        return floodAny
    return eval(condition + region)
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
