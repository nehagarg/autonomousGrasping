import sys
import getopt
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
    if ans < 0:
        ans = 0
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
            if(actual_value < 10):
                actual_joint_values.append(actual_value)
                dummy_joint_values.append(dummy_value)
                if(actual_joint_values[-1] >0.3 and dummy_joint_values[-1] < 1 ):
                    print i
            i = i+1
    #popt, pcov = curve_fit(func, dummy_joint_values,actual_joint_values )
    popt=(1.0, 0.01, -1.0)
    #plt.scatter(actual_joint_values[start_index:end_index],dummy_joint_values[start_index:end_index])
    plt.scatter(dummy_joint_values[start_index:end_index],actual_joint_values[start_index:end_index] )
    #plt.plot(dummy_joint_values, actual_joint_values, 'ko', label="Original Noised Data")
    #plt.plot(dummy_joint_values, func(dummy_joint_values, *popt), 'r-', label="Fitted Curve")
    plt.show()
    
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
             finger = int(arg)
        elif opt == '-s':
             start_index = int(arg)
        elif opt == '-e':
             end_index = int(arg)
    if len(args) > 0:
        command_file = args[0]
        
    plot_finger(command_file, finger, start_index, end_index)