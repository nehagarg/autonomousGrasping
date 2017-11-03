import sys
import getopt
import os

import rospkg

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import numpy as np

PICK_ACTION_ID = 10
OPEN_ACTION_ID = 9
CLOSE_ACTION_ID = 8
NUM_PREDICTIONS = 18

def get_float_array(a):
    return [float(x) for x in a]

def get_data_from_line(line):   
    ans = {}
    sasor = line.strip().split('*')
    init_state = sasor[0].split('|')
    next_state = sasor[2].split('|')
    ans['index'] = get_float_array(init_state[0].split(' ')[0:2])
    ans['init_gripper'] = get_float_array(init_state[0].split(' ')[2:])
    ans['init_object'] = get_float_array(init_state[1].split(' '))
    init_joint_values = get_float_array(init_state[2].split(' '))
    ans['init_joint_values'] = [init_joint_values[0], init_joint_values[2]]
    ans['next_gripper'] = get_float_array(next_state[0].split(' '))
    ans['next_object'] = get_float_array(next_state[1].split(' '))
    next_joint_values =  get_float_array(next_state[2].split(' '))
    ans['next_joint_values'] = [next_joint_values[0], next_joint_values[2]]
    ans['action'] = int(sasor[1])
    ans['reward'] = float(sasor[-1 ])  
    ans['touch'] = get_float_array(sasor[-2].split('|')[-1].split(' '))
    return ans


def load_data_file(object_name, data_dir):
    saso_string = "/SASOData_Cylinder_"
    if 'SASO' in object_name:
        saso_string = ""
        
    object_file_name1 = data_dir + saso_string + object_name + "_allActions.txt"
    object_file_name2 = data_dir + saso_string + object_name + "_openAction.txt"
    print "Loading files" + object_file_name1 + object_file_name2
    ans={}
    with open(object_file_name1, 'r') as f:
        for line in f:
            sasor = get_data_from_line(line.strip())
            if sasor['action'] != OPEN_ACTION_ID:
                if sasor['action']==PICK_ACTION_ID:
                    sasor['touch_prev'] = ans[CLOSE_ACTION_ID][-1]['touch']
                if sasor['action'] not in ans:
                    ans[sasor['action']]= []
                ans[sasor['action']].append(sasor)
                
    with open(object_file_name2, 'r') as f:
        for line in f:
            sasor = get_data_from_line(line.strip())
            if sasor['action'] == OPEN_ACTION_ID:
                if sasor['action'] not in ans:
                    ans[sasor['action']]= []
                ans[sasor['action']].append(sasor)
    return ans
                
                
def get_prediction_value(sasor, p):
    if p < 7:
        return sasor['next_gripper'][p]
    if p < 14:
        return sasor['next_object'][p - 7]
    if p < 16:
        return sasor['next_joint_values'][p-14]
    if p < 18:
        return sasor['touch'][p-16]
    

"""
object_name : name of the object for which model is being learned e.g 9cm cylinder 1001, 1084 etc
data_dir : Directory containing SASO*.txt file
train_type: 1. Pick probability, 
            2. next action joint values, 
            3. next action gripper state, 
            4. next action object state
            5. next action touch values
"""

def train(object_name, data_dir, train_type, classifier_type,learned_model = None, debug = False):
    ans = None
    saso_data = load_data_file(object_name, data_dir)
    if train_type == 'pick_success_probability':
        x=[]
        y=[]
        x_index = []
        for sasor in saso_data[PICK_ACTION_ID]:
            x_entry = sasor['touch_prev'] + sasor['init_joint_values'] 
            x_entry =  x_entry + sasor['init_gripper'] + sasor['init_object']
            x.append(x_entry)
            x_index.append(sasor['index'])
            if sasor['reward'] > 0:
                y.append(1)
            else:
                y.append(0)
        if learned_model is not None:
            logistic = learned_model
        else:
            print classifier_type
            if classifier_type == 'DTC':
                logistic = DecisionTreeClassifier()
            else:
                logistic = linear_model.LogisticRegression(max_iter = 400, C = 1.0)
            logistic.fit(x,y)
        ans = logistic
        print logistic.score(x,y)
        print logistic.get_params()
        if classifier_type != 'DTC':
            print logistic.coef_
            print logistic.intercept_
        else:
            print logistic.feature_importances_
        if debug:
            for i in range(0,len(x)):
                y_bar = logistic.predict([x[i]])
                if y_bar != y[i]:
                    print x_index[i]
                    print x[i] 
                    print y[i]
                    print logistic.predict_proba([x[i]])
                    if classifier_type != 'DTC':
                        print logistic.decision_function([x[i]])
                        prob  = (np.dot(logistic.coef_[0], x[i]) + logistic.intercept_[0])
                        print prob
                        prob *= -1
                        prob = np.exp(prob)
                        prob += 1
                        prob = np.reciprocal(prob)
                        print prob
    if 'next_state' in train_type:
        actions = range(10)
         
        #  predictions can be 18, 7 for gripper pose, 7 for objct pose
        # 2 for joint values
        # 2 for touch values
        predictions = range(NUM_PREDICTIONS)
        
        train_type_array = train_type.split('_')
        for s in train_type_array:
            if 'action' in s:
                actions = s.split('-')[1:]
            if 'pred' in s:
                predictions = s.split('-')[1:]
        ans = {}
        for action_ in actions:
            action = int(action_)
            x = []
            y = []
            l_reg = []
            x_index = []
            for i in range(0,NUM_PREDICTIONS):
                y.append([])
                l_reg.append('')
            for sasor in saso_data[action]: 
                if sasor['reward'] > -999: #discard invalid states
                    x_entry = sasor['init_joint_values'] 
                    x_entry =  x_entry + sasor['init_gripper'][0:7] + sasor['init_object'][0:7]
                    x.append(x_entry)
                    x_index.append(sasor['index'])
                    for p_ in predictions:
                        p = int(p_)
                        y[p].append(get_prediction_value(sasor,p))
            print len(x)
            ans[action] = {}
            for p_ in predictions:
                p = int(p_)
                if learned_model is not None:
                    l_reg[p] = learned_model[action][p]
                else:
                    if classifier_type == 'ridge':
                        l_reg[p] = linear_model.Ridge(alpha = 0.5, normalize = True)
                    elif classifier_type == 'SVR':
                        l_reg[p] = SVR( epsilon = 0.2) 
                    elif classifier_type == 'DTR':
                         l_reg[p] = DecisionTreeRegressor()
                    else:
                        l_reg[p] = linear_model.LinearRegression()
                    l_reg[p].fit(x,y[p])
                ans[action][p] = l_reg[p]
                print repr(action) + " " + repr(p) + " " + repr(l_reg[p].score(x,y[p]))
                print l_reg[p].get_params()
                if classifier_type not in [ 'SVR', 'DTR']:
                    print l_reg[p].coef_
                if classifier_type not in ['DTR']:
                    print l_reg[p].intercept_
                if classifier_type == 'DTR':
                    print l_reg[p].feature_importances_
                
                if debug:
                    for i in range(0,len(x)):
                        y_bar = l_reg[p].predict([x[i]])
                        error_val = 0.01
                        if p >15:
                            error_val = .05
                        if p > 13:
                            error_val = 2*3.14/180.0

                        if y_bar - y[p][i] > error_val or y_bar - y[p][i] < -1*error_val:
                            print x_index[i]
                            print x[i] 
                            print repr(y[p][i]) + ' Prediction ' + repr(y_bar)
                        
    return ans    
        
        

            

           

        
#Status on 23rd october
#linear regression not learning a good function to predict next state
#Will try again with more data.
#Till then will put model learning on hold
#DTR gives promising result
def main():
    # running parameters
    #batch_size = 1
    #test_dataGenerator(batch_size)
    #train()
    
    object_name = '9cm'
    rospack = rospkg.RosPack()
    grasping_ros_mico_path = rospack.get_path('grasping_ros_mico')
    classifier_type = 'linear'
    
    data_dir = grasping_ros_mico_path + "/data_low_friction_table_exp_ver5"
    train_type = 'pick_success_probability'
    training_types = ['pick_success_probability']
    action = 'train'
    action_list = ['train', 'test']
    
    opts, args = getopt.getopt(sys.argv[1:],"ha:d:t:o:c:",["action=","datadir=","type=", "object=", "classifier="])
    #print opts
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'grasping_dynamic_model.py -a <train|test|testBatch> -d <input and output  data dir>  -o <object_name> -t <training type>'
         sys.exit()
      elif opt in ("-a","--action" ):
          action = arg
          if action not in action_list:
              action = raw_input("Please specify correction action[train|test]:")
      elif opt in ("-d", "--datadir"):
         data_dir = arg
      elif opt in ("-t", "--type"):
          if arg.isdigit():
              train_type = training_types[int(arg)]
          else:
            train_type = arg
      elif opt in ("-o", "--object"):
          object_name = arg
      elif opt in ("-c", "--classifier"):
          classifier_type = arg
              
              

  
          
    if(action == 'train'):
        train_object_name = object_name
        if os.path.exists(data_dir + "/SASOData_0-005_Cylinder_" + object_name + "_allActions.txt"):
            train_object_name = "/SASOData_0-005_Cylinder_" + object_name
        
        ans = train(train_object_name, data_dir, train_type, classifier_type)
        train(object_name, data_dir, train_type, classifier_type,ans, False)    
    
    #test()
    #test_dataGenerator(1,logfileName)   

if __name__ == '__main__':
    main()
