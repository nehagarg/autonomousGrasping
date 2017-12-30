import sys
import getopt
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(b * x) + c

def sigmoid(x, b, c):
  return 1 / (1 + math.exp(-b*(x+c)))



def get_plot_value(val):
    f_val = float(val)
    deg_val = f_val*180/3.14
    ans = float("{:.2f}".format(deg_val))
    ans = f_val
    #if ans < 0:
    #    ans = 0
    return ans
def plot_finger(command_file, finger, start_index, end_index):
    print finger
    actual_joint_values = []
    dummy_joint_values = []
    i = 0
    with open(command_file, 'r') as f:
        for line in f:
            joint_values = line.strip().split(' ')
            actual_value = get_plot_value(joint_values[0+(2*finger)])
            dummy_value = get_plot_value(joint_values[1+(2*finger)])
            if(actual_value < 3):
                actual_joint_values.append(actual_value)
                dummy_joint_values.append(dummy_value)
                if(actual_joint_values[-1] >3 or dummy_joint_values[-1] >3 ):
                    print i
                    print actual_joint_values[-1]
                    print dummy_joint_values[-1]
            i = i+1
    #popt, pcov = curve_fit(func, dummy_joint_values,actual_joint_values )
    popt=(1.0, 0.01, -1.0)
    #plt.scatter(actual_joint_values[start_index:end_index],dummy_joint_values[start_index:end_index])
    plt.scatter(dummy_joint_values[start_index:end_index],actual_joint_values[start_index:end_index] )
    #plt.plot(dummy_joint_values, actual_joint_values, 'ko', label="Original Noised Data")
    #plt.plot(dummy_joint_values, func(dummy_joint_values, *popt), 'r-', label="Fitted Curve")
    plt.show()

def get_color_for_action(action_id, reward, touch_values, rel_x, rel_y, finger_joint_values, ver):
    color = None
    if action_id == 10:
        color = 'green'
        if reward < 0:
            color = 'red'
        if rel_x < 0 or rel_x > 0.3 or rel_y > 0.1 or rel_y < -0.1 or reward < -999:
        #if reward < -999:
                
                color = None
        #if float(touch_values[0]) > 5 or float(touch_values[1]) > 5:
        #    color ='blue'
    elif action_id == 8:
        finger_joint1 =   float(finger_joint_values[0])*180/3.14
        finger_joint2 =   float(finger_joint_values[2])*180/3.14
        finger_joint12 =   float(finger_joint_values[1])*180/3.14
        finger_joint22 =   float(finger_joint_values[3])*180/3.14
        if ver in ['ver5', 'ver6']:
            color = 'yellow'
            
            if finger_joint1 > 57 and finger_joint2 > 57 :

                color = 'red'
            elif finger_joint1 > 1 and finger_joint2 > 1:
                color = 'green'

            else:
                color='yellow'
            
        else:
            if finger_joint1 > 22 and finger_joint12 > 85 and finger_joint2 > 22 and finger_joint22 > 85:
               color = 'red' 
            elif finger_joint12 > 25 and finger_joint22 > 25:
               color = 'green'
            else:
                color = 'yellow'
        #if rel_x < -0.1 or rel_x > 0.3 or rel_y > 0.1 or rel_y < -0.1:
        if reward < -999:
                    color = None
        if reward < -101:
            color = None
    elif action_id == 20:
        color = 'yellow'
        if reward == 20:
            color = 'green'
        if reward == -100:
            color = 'red'
        if reward < -999:
            color = None
            
    else:
        #print action_id
        if reward < -101:
            color = None
        elif reward < -2: #-1 or reward == -1.5:
            color = 'red'
        elif reward < -0.6:
            color = 'yellow'
        elif reward < 0:
            color = 'green'
        #if float(touch_values[0]) > 10 or float(touch_values[1]) > 10:
        #    color ='blue'
    return color
    
def plot_pick_success(object_file_name, action_id= 10):
    y = []
    x = []
    colors = []
    index = [0,0]
    reward_index = 0
    
    if(os.path.isdir(object_file_name)):
        files = [os.path.join(object_file_name, f) for f in os.listdir(object_file_name) if f.endswith('.txt')]
    elif(not os.path.exists(object_file_name)):
        obj_dir_name = os.path.dirname(object_file_name)
        file_prefix = os.path.basename(object_file_name)
        files = [os.path.join(obj_dir_name, f) for f in os.listdir(obj_dir_name) if f.startswith(file_prefix) and f.endswith('.txt')]
    else:
        files = [object_file_name]
    for command_file in files:
        print command_file
        with open(command_file, 'r') as f:
            for line in f:
                sasor = line.strip().split('*')
                init_state = sasor[0].split('|')
                init_state_gripper = init_state[0].split(' ')[2:]
                init_state_object = init_state[1].split(' ')
                action = int(sasor[1])
                reward = float(sasor[-1])
                rel_x = float(init_state_object[0]) - float(init_state_gripper[0])
                rel_y = float(init_state_object[1]) - float(init_state_gripper[1])
                touch_values = sasor[-2].split('|')[-1].split(' ')
                finger_joint_values = sasor[2].split('|')[-1].split(' ')
                index_line = init_state[0].split(' ')[0:2]
                if(index_line == index):
                    if reward < reward_index:
                        reward_index = reward
                else:
                    reward_index = 0
                    index[0] = index_line[0]
                    index[1] = index_line[1]

                if init_state[0].split(' ')[0] == '7' and init_state[0].split(' ')[1] == '2':
                    if action == action_id:
                        print reward
                ver = 'ver4'
                if 'ver5' in command_file:
                    ver = 'ver5'
                if 'ver6' in command_file:
                    ver = 'ver6'
                if action == action_id:
                    if reward_index < -999:
                        reward = reward_index
                    color = get_color_for_action(action_id, reward, touch_values, rel_x, rel_y, finger_joint_values, ver)
                    if action_id == 10 and rel_x < 0.04 and color == 'red':
                        print '---------'
                        print init_state
                        print '----------'
                    if action_id == 8 and rel_x < 0.04 and color =='yellow':
                        print line
                    if color is not None: #color=='green':
                        y.append(rel_y)
                        x.append(rel_x)
                        colors.append(color)
                        if color == 'green' and action_id != 8:
                                print init_state
                    else:
                        if action_id == 10:
                            print init_state
                    
                        
                
                   
    area = np.pi * (5 * 1)**2 
    return y,x,area,colors
    
    

if __name__ == '__main__':
    finger = 0
    start_index = 0
    end_index = -1
    command_file = None
    opts, args = getopt.getopt(sys.argv[1:],"hf:s:e:")
    for opt, arg in opts:
        if opt == '-h':
             print 'check_finger_joint_correlation.py -f <finger id 0|1> finger_data_file'
             sys.exit()
        elif opt == '-f':
             fingers = arg.split('-')
        elif opt == '-s':
             start_index = int(arg)
        elif opt == '-e':
             end_index = int(arg)
    if len(args) > 0:
        command_files = args
    print len(command_files)
        
    #plot_finger(command_file, finger, start_index, end_index)
    fig, ax = plt.subplots(nrows=1, ncols=len(command_files))
    i = 1
    for command_file in command_files:
        y,x,area,colors = plot_pick_success(command_file, int(fingers[i-1]))
        plt.subplot(1,len(command_files), i)
        plt.scatter(y,x,s = area, c = colors)
        i = i+1
    plt.show()