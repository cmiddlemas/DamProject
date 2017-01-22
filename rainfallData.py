import pandas as pd
import numpy as np
from numpy import matlib
from scipy import interpolate
rainfall = pd.read_excel(open("basin_flow_data.xlsx","rb"), sheetname='Sheet1')
table = rainfall.values
byRegion = table.transpose()[1:,0:12]

years = 5
byRegion = matlib.repmat(byRegion,1,years)

[norm8,drought8,norm9,drought9,norm10,drought10,norm11,drought11,norm12,drought12,norm13,drought13,norm6,drought6,normvictoria,droughtvictoria,normkariba,droughtkariba] = byRegion
byRegion = matlib.repmat(byRegion,1,years)
years = np.arange(years * 12) / 12.


norm8 = interpolate.interp1d(years,norm8,kind = 'cubic')
norm9 = interpolate.interp1d(years,norm9,kind = 'cubic')
norm10 = interpolate.interp1d(years,norm10,kind = 'cubic')
norm11 = interpolate.interp1d(years,norm11,kind = 'cubic')
norm12 = interpolate.interp1d(years,norm12,kind = 'cubic')
norm13 = interpolate.interp1d(years,norm13,kind = 'cubic')
normvictoria = interpolate.interp1d(years,normvictoria,kind = 'cubic')
normvictoria = interpolate.interp1d(years,normkariba,kind = 'cubic')

drought8 = interpolate.interp1d(years,drought8,kind = 'cubic')
drought9 = interpolate.interp1d(years,drought9,kind = 'cubic')
drought10 = interpolate.interp1d(years,drought10,kind = 'cubic')
drought11 = interpolate.interp1d(years,drought11,kind = 'cubic')
drought12 = interpolate.interp1d(years,drought12,kind = 'cubic')
drought13 = interpolate.interp1d(years,drought13,kind = 'cubic')
droughtvictoria = interpolate.interp1d(years,droughtvictoria,kind = 'cubic')
droughtvictoria = interpolate.interp1d(years,droughtkariba,kind = 'cubic')
"""
Print = True
if Print:
    plt.figure()
    plt.title('Normal Conditions')
    plt.plot(years,norm6,label='6')
    plt.plot(years,norm8,label='8')
    plt.plot(years,norm9,label='9')
    plt.plot(years,norm10,label='10')
    plt.plot(years,norm11,label='11')
    plt.plot(years,norm12,label='12')
    plt.plot(years,norm13,label='13')
    plt.plot(years,normvictoria,label='Victoria')
    plt.plot(years,normkariba,label='Kariba')
    
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
