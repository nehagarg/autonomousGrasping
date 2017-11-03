import os
import sys
import time 
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats

def load_data(file_name ="transparent_green_glass_uncertainty.data" ):
    with open(file_name, 'r') as f:
        content = f.readlines()
        content = [x.strip() for x in content] 
    x = [float(k.split(' ')[0]) for k in content]
    y = [float(k.split(' ')[1]) for k in content]
    return x,y

def remove_outliers(x,y):
    x_new = []
    y_new= []
    for i in range(0,len(x)):
        #if x[i] < 0.5 and x[i] > -0.6 and y[i] < -0.09 and y[i] > -0.21:
        if x[i] != -1:
            x_new.append(x[i])
            y_new.append(y[i])
    return x_new,y_new

#file_name ="transparent_green_glass_uncertainty.data"
file_name = "orange_cup_uncertainty.data"
x_,y_ = load_data(file_name)
x,y = remove_outliers(x_,y_)
mean_x = np.mean(x)
mean_y = np.mean(y)
print (repr(mean_x) + " " + repr(mean_y))

plt.scatter(x,y)
plt.show()

plot_val = x
size = len(x)
xx = scipy.arange(size)
count, bins, ignored = plt.hist(plot_val, 50, normed=True)
#plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
#plt.show()
print len(x)

dist_names = ['norm', 'uniform']

for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(plot_val)
    #p-value = scipy.stats.kstest(y,dist_name)
    print param
    pdf_fitted = dist.pdf(bins, loc=param[-2], scale=param[-1])
    sse = np.sum(np.power(count - pdf_fitted[1:], 2.0))
    print sse
    plt.plot(bins,pdf_fitted, label=dist_name)
    #plt.xlim(0,47)
plt.legend(loc='upper right')
plt.show()
