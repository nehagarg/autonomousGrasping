import numpy as np
import matplotlib.pyplot as plt

#Performance
L2PT10Train = [0,0,0,0,0]
L2PT1Train = [2,1,2,2,3]
L3Train = [0,19, 13, 13, 0]  #incomplete data
PT10Train = [0, 0, 0, 0, 0]
PT1Train = [0, 1, 1, 5, 6]

L2PT10Test = [1,1,1,1,0]
L2PT1Test = [1,1,3,1,0]
L3Test = [12,11,16,12,12]
PT10Test = [1,3 ,6, 3, 8]
PT1Test = [7,5,10,4 ,26 ]

"""
##>25 for train or 30 for test
L2PT10Train = [17,31,30,27,17]
L2PT1Train = [19,37,34,29,21]
L3Train = [17,49,48,42,17]  #incomplete data
PT10Train = [0, 3, 11, 7, 1]
PT1Train = [60, 27, 29, 30, 80]

L2PT10Test = [12,9,14,15,11]
L2PT1Test = [10,17,20,17,14]
L3Test = [27,31,46,29,27]
PT10Test = [8,23 ,21, 25, 26]
PT1Test = [53,38,39,36 ,85]
"""
trainMeans = (np.mean(PT10Train), np.mean(PT1Train), np.mean(L3Train), np.mean(L2PT10Train), np.mean(L2PT1Train))
trainStd = (np.std(PT10Train), np.std(PT1Train), np.std(L3Train), np.std(L2PT10Train), np.std(L2PT1Train))

testMeans = (np.mean(PT10Test), np.mean(PT1Test), np.mean(L3Test), np.mean(L2PT10Test), np.mean(L2PT1Test))

testStd = (np.std(PT10Test), np.std(PT1Test), np.std(L3Test), np.std(L2PT10Test), np.std(L2PT1Test))


N = len(trainMeans)               # number of data entries
ind = np.arange(N)              # the x locations for the groups
width = 0.35                    # bar width

fig, ax = plt.subplots()

rects1 = ax.bar(ind, trainMeans,                  # data
                width,                          # bar width
                color='MediumSlateBlue',        # bar colour
                yerr=trainStd,                  # data for error bars
                error_kw={'ecolor':'Black',    # error-bars colour
                          'linewidth':2})       # error-bar width

rects2 = ax.bar(ind + width, testMeans, 
                width, 
                color='Tomato', 
                yerr=testStd, 
                error_kw={'ecolor':'Black',
                          'linewidth':2})

axes = plt.gca()
axes.set_ylim([0, 30])             # y-axis bounds


ax.set_ylabel('#Failures')
#ax.set_ylabel('#Cases with history length > 25 (train) or > 30 (test)')
#ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('PT10', 'PT1', 'L3', 'L2PT10', 'L2PT1'))

ax.legend((rects1[0], rects2[0]), ('Training Cases', 'Test Cases'))


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center',            # vertical alignment
                va='bottom'             # horizontal alignment
                )

autolabel(rects1)
autolabel(rects2)

plt.show()                              # render the plot

"""
Simple demo of a scatter plot.
"""
"""
min_x_o = 0.4586  #range for object location
max_x_o = 0.5517; #range for object location
min_y_o = 0.0829; #range for object location
max_y_o = 0.2295; #range for object location
x = []
y = []
colors = []
for i in range(0,10):
    for j in range(0,10):
        count = len(x)
        filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/ros/apc/rosmake_ws/despot_vrep_glue/results/despot_logs/VrepData_t20_n10_state_' 
        filename = filename  + repr(count) + '.log'
        txt = open(filename).read()
        x.append(min_x_o + (i*(max_x_o - min_x_o)/9.0))
        y.append(min_y_o + (j*(max_y_o - min_y_o)/9.0))
        
        if 'Reward = 20' in txt :
            colors.append('green')
        elif 'Reward = -100' in txt  :
            colors.append('red')
        elif 'Step 89' in txt  :
            colors.append('yellow')
        else:
            colors.append(0)
        area = np.pi * (15 * 1)**2  # 0 to 15 point radiuses

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Time step 20 sec')
plt.show()
"""
