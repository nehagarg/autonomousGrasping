import numpy as np
import matplotlib.pyplot as plt
from log_file_parser import ParseLogFile
import os
import csv
import sys

def get_mean_std_for_numbers_in_file(filename):
    import numpy as np
    import math
    a = []
    sum2 = 0.0
    with open(filename, 'r') as f:
        for line in f:
          a.append(float(line)) 
          sum2 = sum2+ a[-1]*a[-1]
    mean = np.mean(a)
    std = np.std(a) #standard deviation)
    std2 = math.sqrt((sum2/(len(a)*len(a))) - (mean*mean/len(a))) #standard error
    #print mean
    #print std
    #print std2
    return (mean, std, std2)

def generate_reward_file(dir_name, pattern_list, reward_file_size, reward_file_name):
    prev_path = os.getcwd()
    os.chdir(dir_name)
    #r = os.popen("cd " + dir_name).read()
    #print r
    i = 0
    for pattern in pattern_list:
        out_str = '>>'
        if i == 0:
            out_str = '>'
        system_command = "grep 'Total undiscounted reward = ' "
        system_command = system_command  + pattern + " | cut -d'=' -f2 "
        system_command = system_command + out_str + " " + reward_file_name
        os.system(system_command)
        i = i+1
    
    a = int(os.popen("cat " + reward_file_name + " | wc -l").read())
    if(a!=reward_file_size):
        print a 
        print reward_file_size
        print dir_name
        print reward_file_name
        #assert(False)
    os.chdir(prev_path)

def generate_average_step_file(dir_name, pattern_list, reward_file_size, reward_file_name, reward_value):
    os.system("cd " + dir_name)
    i = 0
    for pattern in pattern_list:
        out_str = '>>'
        if i == 0:
            out_str = '>'
        system_command = "grep -B 5 'Simulation terminated in' "
        system_command = system_command + pattern + " | grep -A 5 'Reward = "
        system_command = system_command + repr(reward_value) + "' | "
        system_command = system_command + "grep 'Simulation terminated in' | cut -d' ' -f4 "
        system_command = system_command + out_str + " " + reward_file_name
        os.system(system_command)
        i = i+1
    
    a = int(os.popen("cat " + reward_file_name + " | wc -l").read())
    if(a!=reward_file_size):
        print a 
        print reward_file_size
        assert(False)
    os.system("cd -")

def get_success_failure_cases(dir_name, pattern_list, reward_value):
    prev_path = os.getcwd()
    os.chdir(dir_name)
    cases = []
    for pattern in pattern_list:
        system_command = "grep -B 5 'Simulation terminated in' "
        system_command = system_command + pattern + " | grep 'Reward = "
        system_command = system_command + repr(reward_value) + "' | wc -l"
        success_cases = os.popen(system_command).read()
        cases.append(float(success_cases))
    os.chdir(prev_path)
    return cases


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center',            # vertical alignment
                va='bottom'             # horizontal alignment
                )
#means 2D array
#stds 2D array
def plot_bar_graph_with_std_error(means, stds, colors):
    N = len(means[0])               # number of data entries
    ind = np.arange(N)              # the x locations for the groups
    width = 0.35                    # bar width
    
    fig, ax = plt.subplots()
    rects = []
    for i in (0,len(means)):
        rects1 = ax.bar(ind, means[i],                  # data
                width,                          # bar width
                color=colors[i],        # bar colour
                yerr=stds[i],                  # data for error bars
                error_kw={'ecolor':'Black',    # error-bars colour
                          'linewidth':2})       # error-bar width
        rects.append(rects1)
    
    axes = plt.gca()
    axes.set_ylim([0, 100])             # y-axis bounds

    ax.set_ylabel('#Cases with history length > 30')
    ax.set_title('Test Cases')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('PT10', 'PT1', 'L3', 'L2PT10', 'L2PT1'))

    #ax.legend((rects1[0], rects2[0]), ('Training Cases', 'Test Cases'))

    for rect in rects:
        autolabel(rect)


    plt.show()                              # render the plot


def plot_line_graph_with_std_error(means,stds,title, legend, xlabel, colors = None):
  
    if colors is None:
        colors = ['blue', 'red', 'yellow', 'green', 'magenta']
    N = len(means[0])   # number of data entries

    ind = np.arange(N)
    fig, ax = plt.subplots()
    print len(ind)
    print len(means[0])
    #plt.plot(ind,np.array(means[0]))
    for i in range(0,len(means)):
        print i
        plt.errorbar(ind,means[i], stds[i], color = colors[i % len(colors)])
        #plt.errorbar(ind,means[i], stds[i])
        #plt.plot(ind,np.array(means[i]))
    plt.title(title)
    ax.set_xticklabels(xlabel)
    plt.legend(legend)
    plt.show()


def plot_scatter_graph(x,y,area):
    area = np.pi * (15 * 1)**2  # 0 to 15 point radiuses

    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.scatter(x, y, s=area)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Time step 20 sec')
    plt.show()
    

def generate_average_reward_csv_for_vrep_multi_object_cases(csv_file_name, dir_name = None, testCases = True):
    

    if dir_name is None:
        dir_name = "/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10"
    time_steps = [1,5]
    sampled_scenarios = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    learning_versions = [6]
    patterns = ['*8cm*.log', '*9cm*.log', '*7cm*.log']
    reward_file_name = 'reward.txt'
    reward_file_size = 1500
    if testCases:
        patterns = ['*85mm*.log', '*75mm*.log']
        reward_file_name = 'reward_test.txt'
        reward_file_size = 1000
    #means = []
    #stds = []
    csv_file = open(csv_file_name,'w')
    
    csv_file.write("Average undiscounted reward")
    for n in sampled_scenarios:
          csv_file.write(",n" +  repr(n))
    csv_file.write("\n")
    
    for t in time_steps:
        csv_file.write("T" + repr(t))
        #means.append([])
        #stds.append([])
        for n in sampled_scenarios:
            new_dir_name = dir_name + "/t" + repr(t)+ "_n" + repr(n)
            generate_reward_file(new_dir_name, patterns, reward_file_size, reward_file_name)
            (mean, stddev, stderr) = get_mean_std_for_numbers_in_file(new_dir_name + "/" + reward_file_name)
            
            csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
        csv_file.write("\n")
        
    for l in learning_versions:
        csv_file.write("L" + repr(l))
        for n in sampled_scenarios:
            new_dir_name = dir_name + "/learning/version" + repr(l)
            generate_reward_file(new_dir_name, patterns, reward_file_size, reward_file_name)
            (mean, stddev, stderr) = get_mean_std_for_numbers_in_file(new_dir_name + "/" + reward_file_name)
            csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
        csv_file.write("\n")
        for t in time_steps:
            csv_file.write("L" + repr(l) + "T" + repr(t))
        #means.append([])
        #stds.append([])
            for n in sampled_scenarios:
                new_dir_name = dir_name + "/learning/version" + repr(l) + "/combined/t" + repr(t)+ "_n" + repr(n)
                generate_reward_file(new_dir_name, patterns, reward_file_size, reward_file_name)
                (mean, stddev, stderr) = get_mean_std_for_numbers_in_file(new_dir_name + "/" + reward_file_name)
            
                csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
            csv_file.write("\n")

def generate_success_cases_csv_for_vrep_multi_object_cases(csv_file_name, dir_name = None, testCases = True):
    if dir_name is None:
        dir_name = "/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10"
    time_steps = [1,5]
    sampled_scenarios = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    learning_versions = [6]
    patterns = ['*8cm*.log', '*9cm*.log', '*7cm*.log']
    #reward_file_name = 'reward.txt'
    reward_file_size = 1500
    if testCases:
        patterns = ['*85mm*.log', '*75mm*.log']
        #reward_file_name = 'reward_test.txt'
        reward_file_size = 1000
    #means = []
    #stds = []
    csv_file = open(csv_file_name,'w')
    
    csv_file.write("Success failure cases")
    for n in sampled_scenarios:
          csv_file.write(",n" +  repr(n))
    csv_file.write("\n")
    
    for t in time_steps:
        csv_file.write("T" + repr(t))
        #means.append([])
        #stds.append([])
        for n in sampled_scenarios:
            new_dir_name = dir_name + "/t" + repr(t)+ "_n" + repr(n)
            success_cases = sum(get_success_failure_cases(new_dir_name,patterns, 100))
            failure_cases = sum(get_success_failure_cases(new_dir_name,patterns, -10))
            stuck_cases = reward_file_size - (success_cases + failure_cases)
            csv_file.write("," + repr(success_cases) + ":" + repr(failure_cases)+":" + repr(stuck_cases))
        csv_file.write("\n")
        
    for l in learning_versions:
        csv_file.write("L" + repr(l))
        for n in sampled_scenarios:
            new_dir_name = dir_name + "/learning/version" + repr(l)
            success_cases = sum(get_success_failure_cases(new_dir_name,patterns, 100))
            failure_cases = sum(get_success_failure_cases(new_dir_name,patterns, -10))
            stuck_cases = reward_file_size - (success_cases + failure_cases)
            csv_file.write("," + repr(success_cases) + ":" + repr(failure_cases)+":" + repr(stuck_cases))

        csv_file.write("\n")
        for t in time_steps:
            csv_file.write("L" + repr(l) + "T" + repr(t))
        #means.append([])
        #stds.append([])
            for n in sampled_scenarios:
                new_dir_name = dir_name + "/learning/version" + repr(l) + "/combined/t" + repr(t)+ "_n" + repr(n)
                success_cases = sum(get_success_failure_cases(new_dir_name,patterns, 100))
                failure_cases = sum(get_success_failure_cases(new_dir_name,patterns, -10))
                stuck_cases = reward_file_size - (success_cases + failure_cases)
                csv_file.write("," + repr(success_cases) + ":" + repr(failure_cases)+":" + repr(stuck_cases))

            csv_file.write("\n")
   

def plot_graph_from_csv(csv_file, data_type = 'reward'):
    plt_title = None
    xlabels = None
    legend = []
    means = []
    stds = []
    with open(csv_file) as f:
        line = f.readline().rstrip('\n').split(",")
        plt_title = line[0]
        xlabels = line[1:]
        for line in f:
            data = line.rstrip('\n').split(",")
            legend.append(data[0])
            means.append([])
            stds.append([])
            for value in data[1:]:
                mean = float(value.split(':')[0])
                stderr = 0
                if data_type == 'reward':
                    stderr = float(value.split(':')[2])
                means[-1].append(mean)
                stds[-1].append(stderr)
    plot_line_graph_with_std_error(means, stds, plt_title, legend, xlabels)
    
dir_name = None
data_type = 'reward'
plot_graph = 'yes'
if(len(sys.argv) > 1):
    plot_graph = sys.argv[1]
if(len(sys.argv) > 2):
    data_type = sys.argv[2]
if(len(sys.argv) > 3):
    dir_name = sys.argv[3]


csv_file_name = 'multi_object_' + data_type + '_test.csv'
generate_csv = True
if os.path.exists(csv_file_name):
    ans = raw_input("Csv file already exists. Overwrite it?y or n")
    if ans == 'n':
        generate_csv = False
        
if generate_csv:    
    if data_type == 'reward':
        generate_average_reward_csv_for_vrep_multi_object_cases(csv_file_name, dir_name, True)
    if data_type == 'success_cases':
        generate_success_cases_csv_for_vrep_multi_object_cases(csv_file_name, dir_name, True)

if plot_graph == 'yes':
    plot_graph_from_csv(csv_file_name, data_type)

    
    
"""
#Performance toy problem
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
axes.set_ylim([0, 100])             # y-axis bounds


#ax.set_ylabel('#Failures')
ax.set_ylabel('#Cases with history length > 30')
ax.set_title('Test Cases')
ax.set_xticks(ind + width)
ax.set_xticklabels(('PT10', 'PT1', 'L3', 'L2PT10', 'L2PT1'))

#ax.legend((rects1[0], rects2[0]), ('Training Cases', 'Test Cases'))


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center',            # vertical alignment
                va='bottom'             # horizontal alignment
                )

#autolabel(rects1)
autolabel(rects2)

plt.show()                              # render the plot
"""
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
#Draw belief
log_filename = '/home/neha/WORK_FOLDER/phd2013/phdTopic/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/vrep_simulator/TableScene_9cm_gaussian_belief_with_state_in_belief_t10_n5_trial_2.log'
lfp =  ParseLogFile(log_filename, 'vrep', 0, 'vrep')
l = len(lfp.stepInfo_[0]['belief'])
for i in range(0, l):
    x.append(lfp.stepInfo_[0]['belief'][i]['state'].o_x)
    y.append(lfp.stepInfo_[0]['belief'][i]['state'].o_y)
    
"""    
     
"""
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
"""




