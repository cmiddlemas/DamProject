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
months = np.arange(years * 12)


norm8 = interpolate.interp1d(months,norm8,kind = 'cubic')
norm9 = interpolate.interp1d(months,norm9,kind = 'cubic')
norm10 = interpolate.interp1d(months,norm10,kind = 'cubic')
norm11 = interpolate.interp1d(months,norm11,kind = 'cubic')
norm12 = interpolate.interp1d(months,norm12,kind = 'cubic')
norm13 = interpolate.interp1d(months,norm13,kind = 'cubic')
normvictoria = interpolate.interp1d(months,normvictoria,kind = 'cubic')
normvictoria = interpolate.interp1d(months,normkariba,kind = 'cubic')

drought8 = interpolate.interp1d(months,drought8,kind = 'cubic')
drought9 = interpolate.interp1d(months,drought9,kind = 'cubic')
drought10 = interpolate.interp1d(months,drought10,kind = 'cubic')
drought11 = interpolate.interp1d(months,drought11,kind = 'cubic')
drought12 = interpolate.interp1d(months,drought12,kind = 'cubic')
drought13 = interpolate.interp1d(months,drought13,kind = 'cubic')
droughtvictoria = interpolate.interp1d(months,droughtvictoria,kind = 'cubic')
droughtvictoria = interpolate.interp1d(months,droughtkariba,kind = 'cubic')
"""
Print = True
if Print:
    plt.figure()
    plt.title('Normal Conditions')
    plt.plot(months,norm6,label='6')
    plt.plot(months,norm8,label='8')
    plt.plot(months,norm9,label='9')
    plt.plot(months,norm10,label='10')
    plt.plot(months,norm11,label='11')
    plt.plot(months,norm12,label='12')
    plt.plot(months,norm13,label='13')
    plt.plot(months,normvictoria,label='Victoria')
    plt.plot(months,normkariba,label='Kariba')
    
if Print:
    plt.figure()
    plt.title('Drought Conditions')
    plt.plot(months,drought6,label='6')
    plt.plot(months,drought8,label='8')
    plt.plot(months,drought9,label='9')
    plt.plot(months,drought10,label='10')
    plt.plot(months,drought11,label='11')
    plt.plot(months,drought12,label='12')
    plt.plot(months,drought13,label='13')
    plt.plot(months,droughtvictoria,label='Victoria')
    plt.plot(months,droughtkariba,label='Kariba')
"""
