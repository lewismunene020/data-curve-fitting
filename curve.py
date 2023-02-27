import numpy as np 

import matplotlib.pyplot as  plt 
import numpy as np 

from scipy.optimize import curve_fit  

# function for perfoming the curve fitting and   plotting the graphs
def line_func(x ,a ,b):
    return a*np.cos(2*np.pi +b ) 




# read the CSV file as a text file
data = np.loadtxt('temperature.csv', delimiter=',', dtype=str, skiprows=1)

# extract the columns
date = data[:, 0]
max_temp = data[:, 1]
min_temp = data[:, 2]
avg_temp = data[:, 3]



# print the first 5 rows of the data
print(data[:5])  

# convert date string to a numerical value from  0 to the last date
x = np.arange(len(date)).astype(float)

# convert avg_temp strings to floats
y = avg_temp.astype(float)  


# perform curve fitting using the dataset and the function
popt, pcov = curve_fit(line_func, x, y)

# print the parameters of the best fit curve
print(f'Best fit parameters:{popt}')

#printing the  point covariance pcov
print(f"Point of variance matrix {pcov}")  




# plot the original data
plt.plot(x, y, '.', label='Original Data')
plt.legend(loc='best') 
plt.title("ORIGINAL DATA")
plt.xlabel('Date')
plt.ylabel('Average Temperature')
plt.show() 







# plot the best fit curve

y1 = x + line_func(x ,*popt) 
plt.plot(x, y1 , '.', label='Best Fit Curve')
plt.legend(loc='best')
plt.title("CURVE OF BEST FIT")
plt.xlabel("Date")
plt.ylabel("Average Temperature")

plt.show()


