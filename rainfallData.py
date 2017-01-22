import pandas as pd


rainfall = pd.read_excel(open("basin_flow_data.xlsx","rb"), sheetname='Sheet1')
table = rainfall.values
byRegion = table[3:15,:].transpose()

[norm8,drought8,norm9,drought9,norm10,drought10,norm11,drought11,norm12,drought12,norm13,drought13,norm6,drought6,normvictoria,droughtvictoria,normkariba,droughtkariba]   = byRegion[1:]
months = np.arange(12) + 1
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
