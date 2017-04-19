import numpy as np
import matplotlib.pyplot as plt
from log_file_parser import ParseLogFile
import os
import csv
import sys, getopt

def get_mean_std_for_array(a, sum2 = None):
    if sum2 is None:
        sum2 = sum([x*x for x in a ])
    if(len(a) == 0):
        print "zero length array"
        std2 = 0
        mean = 0
        std = 0
    else:
        mean = np.mean(a)
        std = np.std(a) #standard deviation)
        std2 = math.sqrt((sum2/(len(a)*len(a))) - (mean*mean/len(a)))  #stderr
    return (mean, std, std2)
    
def get_mean_std_for_numbers_in_file(filename):
    import numpy as np
    import math
    a = []
    sum2 = 0.0
    with open(filename, 'r') as f:
        for line in f:
          a.append(float(line)) 
          sum2 = sum2+ a[-1]*a[-1]
    if(len(a) == 0):
        print filename
    
    (mean, std, std2) = get_mean_std_for_array(a, sum2)
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
    prev_path = os.getcwd()
    os.chdir(dir_name)
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
        print dir_name
        print reward_file_name
        #assert(False)
    os.chdir(prev_path)

def get_regex_from_number_range(start_number, end_number):
    #Assuming end_number - start_number + 1 is a multiple of 10
    end_number_string = str(end_number)
    start_number_string = str(start_number)
    num_digits_end_number = len(str(end_number_string))
    num_digits_start_number = len(str(start_number_string))
    
    pattern = '_'
    for i in range(0,num_digits_start_number-num_digits_end_number):
        pattern = pattern + '[0-'+end_number_string[i]+']?'
    for i in range(num_digits_start_number-num_digits_end_number, num_digits_end_number):
        pattern = pattern + '[' + start_number_string[i] + '-' + end_number_string[i] + ']'
    return pattern
    
def get_success_failure_cases(dir_name, pattern_list, reward_value, index_step, end_index ):
    prev_path = os.getcwd()
    os.chdir(dir_name)
    all_cases = []
    num_iterations = end_index/index_step
    for i in range(0, num_iterations):
        start_number = i*index_step + 0
        end_number = (i+1)*index_step -1
        number_pattern = get_regex_from_number_range(start_number, end_number)
        cases=[]
        for pattern in pattern_list:
            new_pattern = pattern.replace('.log', number_pattern + '.log')
            system_command = "grep -B 5 'Simulation terminated in' "
            system_command = system_command + new_pattern + " | grep 'Reward = "
            system_command = system_command + repr(reward_value) + "' | wc -l"
            success_cases = os.popen(system_command).read()
            cases.append(float(success_cases))
        all_cases.append(cases)
    os.chdir(prev_path)
    return all_cases


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
        colors = ['blue', 'red', 'yellow', 'teal', 'magenta', 'cyan', 'hotpink', 'lightblue', 'pink']
    N = len(means[0])   # number of data entries

    ind = np.arange(N)
    fig, ax = plt.subplots()
    print len(ind)
    print len(means[0])
    #plt.plot(ind,np.array(means[0]))
    for i in range(0,len(means)):
        print i
        plt.errorbar(ind,means[i], stds[i], marker = 'o', color = colors[i % len(colors)], linewidth=4.0)
        #plt.errorbar(ind,means[i], stds[i])
        #plt.plot(ind,np.array(means[i]))
    plt.title(title)
    ax.set_xticklabels(xlabel)
    plt.legend(legend)
    plt.show()


def plot_scatter_graph(x,y, colors):
    area = np.pi * (15 * 1)**2  # 0 to 15 point radiuses

    #plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.scatter(x, y, s=area, c = colors)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Time step 20 sec')
    plt.show()
    


def write_statistics_to_csv_files(new_dir_name, test_pattern, csv_files, index_step, end_index):
    
    if test_pattern == 'train':
        patterns = ['*8cm*.log', '*9cm*.log', '*7cm*.log']
    elif test_pattern == 'test':
        patterns = ['*85mm*.log', '*75mm*.log']
    else :
        patterns = ['*' + test_pattern + '*.log']
    reward_file_size = end_index * len(patterns)
    max_success_cases = index_step *  len(patterns)
    
    average_step_file_name = 'average_step_' + test_pattern
    reward_file_name = 'reward_' + test_pattern +'.txt'
    
    reward_csv_file = csv_files[0]
    success_csv_file = csv_files[1]
    av_step_success_file = csv_files[2]
    av_step_failure_file = csv_files[3]
    failure_csv_file = csv_files[4]
    stuck_csv_file = csv_files[5]
    
    success_cases_array = get_success_failure_cases(new_dir_name,patterns, 100)
    success_cases_per_index_step = [sum(x) for x in success_cases_array]
    (mean, stddev, stderr) = get_mean_std_for_array(success_cases_per_index_step)        
    success_csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
    
    success_cases = sum(success_cases_per_index_step)
    generate_average_step_file(new_dir_name, patterns, success_cases, average_step_file_name + '_success.txt' , 100)
    (mean, stddev, stderr) = get_mean_std_for_numbers_in_file(new_dir_name + "/" + average_step_file_name + '_success.txt')
    av_step_success_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
            
    failure_cases_array = get_success_failure_cases(new_dir_name,patterns, -10)
    failure_cases_per_index_step =  [sum(x) for x in failure_cases_array]
    (mean, stddev, stderr) = get_mean_std_for_array(failure_cases_per_index_step)        
    failure_csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
    
    failure_cases = sum(failure_cases_per_index_step)
    generate_average_step_file(new_dir_name, patterns, failure_cases, average_step_file_name + '_failure.txt' , -10)
    (mean, stddev, stderr) = get_mean_std_for_numbers_in_file(new_dir_name + "/" + average_step_file_name + '_failure.txt')
    av_step_failure_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))

    stuck_cases_per_index_step = [(max_success_cases - x) for x in success_cases_per_index_step ]
    stuck_cases_per_index_step = [(max_success_cases - x) for x in failure_cases_per_index_step ]
    (mean, stddev, stderr) = get_mean_std_for_array(stuck_cases_per_index_step)        
    stuck_csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))
    
    generate_reward_file(new_dir_name, patterns, reward_file_size, reward_file_name)
    (mean, stddev, stderr) = get_mean_std_for_numbers_in_file(new_dir_name + "/" + reward_file_name)
    reward_csv_file.write("," + repr(mean) + ":" + repr(stddev)+":" + repr(stderr))


def generate_csv_file(csv_file_name, dir_name, test_pattern, time_steps,sampled_scenarios, learning_versions, combined_policy_versions, begin_index, end_index, index_step):
    if dir_name is None:
        dir_name = "/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10"    
    
    #means = []
    #stds = []
    csv_files = []
    reward_csv_file = open(csv_file_name[0], 'w')
    csv_files.append(reward_csv_file)
    success_csv_file = open(csv_file_name[1],'w')
    csv_files.append(success_csv_file)
    av_step_success_file = open(csv_file_name[2], 'w')
    csv_files.append(av_step_success_file)
    av_step_failure_file = open(csv_file_name[3], 'w')
    csv_files.append(av_step_failure_file)
    failure_csv_file = open(csv_file_name[4],'w')
    csv_files.append(failure_csv_file)
    stuck_csv_file = open(csv_file_name[5],'w')
    csv_files.append(stuck_csv_file)
    
    reward_csv_file.write("Average undiscounted reward")
    success_csv_file.write("Success cases")
    av_step_success_file.write("Average Steps Success")
    av_step_failure_file.write("Average Steps Failure")
    failure_csv_file.write("Failure cases")
    stuck_csv_file.write("Stuck cases")
    
    for n in sampled_scenarios:
        for csv_file in csv_files:
          csv_file.write(",n" +  repr(n))
    
    for csv_file in csv_files:
        csv_file.write("\n")
    
    for t in time_steps:
        for csv_file in csv_files:
            csv_file.write("T" + repr(t))
        
        #means.append([])
        #stds.append([])
        for n in sampled_scenarios:
            new_dir_name = dir_name + "/t" + repr(t)+ "_n" + repr(n)
            write_statistics_to_csv_files(new_dir_name, test_pattern, csv_files, index_step, end_index)
        for csv_file in csv_files:
            csv_file.write("\n")
        
        
    for l in learning_versions:
        for csv_file in csv_files:
            csv_file.write("L" + repr(l))
        
        for n in sampled_scenarios:
            new_dir_name = dir_name + "/learning/version" + repr(l)
            write_statistics_to_csv_files(new_dir_name, test_pattern, csv_files, index_step, end_index)
        for csv_file in csv_files:    
            csv_file.write("\n")
        
        for c in combined_policy_versions:
            for t in time_steps:
                for csv_file in csv_files:    
                    csv_file.write("L" + repr(l) + "T" + repr(t) + "S" + repr(c))
                
                #means.append([])
                #stds.append([])
                for n in sampled_scenarios:
                    new_dir_name = dir_name + "/learning/version" + repr(l) + "/combined_" + repr(c)+"/t" + repr(t)+ "_n" + repr(n)
                    write_statistics_to_csv_files(new_dir_name, test_pattern, csv_files, index_step, end_index)                
                
                for csv_file in csv_files: 
                    csv_file.write("\n")
                
   

def plot_graph_from_csv(csv_file, plt_error):
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
                if plt_error:
                    stderr = float(value.split(':')[2])
                else:
                    stderr = float(value.split(':')[1])
                means[-1].append(mean)
                stds[-1].append(stderr)
    plot_line_graph_with_std_error(means, stds, plt_title, legend, xlabels)
    

def get_params_and_generate_or_plot_csv(plot_graph, csv_name_prefix, dir_name, pattern):
    
    
    #data_type = 'reward'
    #input_data_type = raw_input("Data type: reward or success_cases ?")
    data_types = ['reward', 'success_cases', 'av_step_success', 'av_step_failure', 'failure_cases', 'stuck_cases']
    #if input_data_type in data_types:
    #    data_type = input_data_type
    #else:
    #    print "Invalid data type. Setting data_type to reward"
    csv_file_names = []
    for data_type in data_types:
        csv_file_names.append(csv_name_prefix + '_' + data_type + '_' + pattern + '.csv')

    #csv_file_names = [csv_name_prefix + '_' + data_type + '_' + pattern + '.csv']
    #if data_type == 'success_cases':
    #    csv_file_names.append(csv_name_prefix + '_' + 'av_step_success' + '_' + pattern + '.csv')
    #    csv_file_names.append(csv_name_prefix + '_' + 'av_step_failure' + '_' + pattern + '.csv')
    
    
    file_name_begin_range = 0
    file_name_end_range = len(csv_file_names)
    if plot_graph == 'yes':         
        plot_type = int(raw_input("Plot reward [0] success_cases[1] or average_step_success[2] or average_step_failure[3] or failure_cases[4] or stuck_cases[5]?"))
        file_name_begin_range = plot_type
        file_name_end_range = plot_type + 1
       
    
    csv_file_names_for_generation = csv_file_names[:]
    generate_csv = False
    
    for i  in range(file_name_begin_range, file_name_end_range):
        if os.path.exists(csv_file_names[i]):
            ans = raw_input("Csv file " + csv_file_names[i] + " already exists. Overwrite it[y or n]?")
            if ans == 'y':
                generate_csv = True
            else:
                #generate_csv = False and generate_csv
                csv_file_names_for_generation[i] = 'dummy'+ repr(i)+'.csv'
            
                
        else:
            generate_csv = True  



    #csv_file_name = 'multi_object_' + data_type + '_test.csv'


    if generate_csv:
        time_steps = [1,5]
        time_steps = get_list_input(time_steps, "Planning times")
    
        sampled_scenarios = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
        sampled_scenarios = get_list_input(sampled_scenarios, "Sampled scenarios")
    
        
        learning_versions = [6]
        learning_versions =  get_list_input(learning_versions, "Learning versions")
    
    
        combined_policy_versions = [0, 1, 2]
        combined_policy_versions = get_list_input(combined_policy_versions, "Combined policy versions")
        
        begin_index = 0
        end_index = 1000
        end_index_input = raw_input("End index (default 1000):")
        if end_index_input:
            end_index = int(end_index_input)
        
        index_step = 1000
        index_step_input = raw_input("Index step(default 1000):")
        if index_step_input:
            index_step = int(index_step_input)
        
        
        print "dir_name is: " + dir_name
        #if data_type == 'reward':
        #    generate_average_reward_csv_for_vrep_multi_object_cases(csv_file_names_for_generation[0], dir_name, pattern, time_steps, sampled_scenarios, learning_versions, combined_policy_versions, begin_index, end_index, index_step)
        #if data_type == 'success_cases':
        generate_csv_file(csv_file_names_for_generation, dir_name, pattern, time_steps, sampled_scenarios, learning_versions, combined_policy_versions, begin_index, end_index, index_step)

    if plot_graph == 'yes':
        plt_error = True
        if plot_type > 0:
            plt_error = False
        plot_graph_from_csv(csv_file_names[plot_type], plt_error)
    




def get_and_plot_success_failure_cases_for_vrep(dir_name, pattern):
    
    min_x_o = 0.4586  #range for object location
    max_x_o = 0.5517; #range for object location
    min_y_o = 0.0829; #range for object location
    max_y_o = 0.2295; #range for object location
    max_reward = 100
    min_reward = -10
    x = []
    y = []
    colors = []
    time_step = raw_input('Time step?')
    
    scenarios = raw_input('Sccenarios?')
    time_scenario_string = 't'+ time_step + '_n' + scenarios
    for i in range(0,500):
        log_filename = dir_name +'/' +time_scenario_string + '/TableScene_cylinder_'+ pattern +'_gaussian_belief_' + time_scenario_string + '_trial_' + repr(i) + '.log'
        #log_filename = dir_name +'/' +time_scenario_string + '/TableScene_cylinder_'+ pattern +'_gaussian_belief_with_state_in_belief_' + time_scenario_string + '_trial_' + repr(i) + '.log'
        
        fullData =  ParseLogFile(log_filename, 'vrep', 0, 'vrep').getFullDataWithoutBelief()
        x.append(fullData['roundInfo']['state'].o_x)
        y.append(fullData['roundInfo']['state'].o_y)
        if fullData['stepInfo'][-1]['reward'] == max_reward:
            colors.append('green')
        elif fullData['stepInfo'][-1]['reward'] == min_reward:
            colors.append('red')
        else:
            colors.append('yellow')
            
    plot_scatter_graph(x, y, colors)
    
def get_list_input(sampled_scenarios, command):
    while True:
        input = raw_input(command + " are " + " ".join(map(str, sampled_scenarios)) + " To add type a <no>. To remove type r <no>. To stop type s.")
        if 's' in input:
            break
        if 'a' in input:
            sampled_scenarios.append(int(input.split(' ')[1]))
            sampled_scenarios = sorted(set(sampled_scenarios))
        if 'r' in input:
            sampled_scenarios.remove(int(input.split(' ')[1]))
    return sampled_scenarios        

if __name__ == '__main__':
    plot_graph = 'no'
    csv_name_prefix = 'multi_object'
    dir_name = "/home/neha/WORK_FOLDER/ncl_dir_mount/neha_github/autonomousGrasping/grasping_ros_mico/results/despot_logs/multiObjectType/belief_cylinder_7_8_9_reward100_penalty10"
        
    opts, args = getopt.getopt(sys.argv[1:],"hpd:f:",["dir=","csv_prefix="])
    #print opts
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'plot_despot_results.py -p -d <directory_name> -f <csv_file_prefix>'
         sys.exit()
      elif opt == '-p':
          plot_graph='yes'
      elif opt in ("-d", "--dir"):
         dir_name = arg
      elif opt in ("-f", "--csv_prefix"):
         csv_name_prefix = arg
         
    pattern = 'test'
    input_pattern = raw_input("Pattern type: train or test or file identifier?")
    input_patterns = ['test', 'train', '9cm', '8cm', '7cm', '75mm', '85mm']
    if input_pattern in input_patterns:
        pattern = input_pattern
    else :
        print "Invalid pattern. Setting pattern as test"
    
        
    get_params_and_generate_or_plot_csv(plot_graph, csv_name_prefix, dir_name, pattern)
    #get_and_plot_success_failure_cases_for_vrep(dir_name, pattern)
    
    
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




