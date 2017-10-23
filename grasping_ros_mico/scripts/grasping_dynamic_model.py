import sys
import getopt

import rospkg

from sklearn import linear_model
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
    
    object_file_name1 = data_dir + "/SASOData_Cylinder_" + object_name + "_allActions.txt"
    object_file_name2 = data_dir + "/SASOData_Cylinder_" + object_name + "_openAction.txt"
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

def train(object_name, data_dir, train_type):
    
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
        logistic = linear_model.LogisticRegression(max_iter = 400, C = 1.0)
        logistic.fit(x,y)
        print logistic.score(x,y)
        print logistic.get_params()
        print logistic.coef_
        print logistic.intercept_
        for i in range(0,len(x)):
            y_bar = logistic.predict([x[i]])
            if y_bar != y[i]:
                print x_index[i]
                print x[i] 
                print y[i]
                print logistic.predict_proba([x[i]])
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
                    x_entry =  x_entry + sasor['init_gripper'] + sasor['init_object']
                    x.append(x_entry)
                    x_index.append(sasor['index'])
                    for p_ in predictions:
                        p = int(p_)
                        y[p].append(get_prediction_value(sasor,p))
                    
            for p_ in predictions:
                p = int(p_)
                l_reg[p] = linear_model.LinearRegression()
                l_reg[p].fit(x,y[p])
                print repr(action) + " " + repr(p) + " " + repr(l_reg[p].score(x,y[p]))
                print l_reg[p].get_params()
                print l_reg[p].coef_
                print l_reg[p].intercept_
                for i in range(0,len(x)):
                    y_bar = l_reg[p].predict([x[i]])
                    if y_bar - y[p][i] > 0.01 or y_bar - y[p][i] < -0.01:
                        print x_index[i]
                        print x[i] 
                        print repr(y[p][i]) + ' Prediction ' + repr(y_bar)
                        
        
        
        
        
def test(model_name, svm_model_prefix, model_input, rnn_state, action = 'test'):
    
   pass
            

           

        
#Status on 23rd october
#linear regression not learning a good function to predict next state
#Will try again with more data.
#Till then will put model learning on hole
def main():
    # running parameters
    #batch_size = 1
    #test_dataGenerator(batch_size)
    #train()
    
    object_name = '9cm'
    rospack = rospkg.RosPack()
    grasping_ros_mico_path = rospack.get_path('grasping_ros_mico')
    
    data_dir = grasping_ros_mico_path + "/data_low_friction_table_exp_ver5"
    train_type = 'pick_success_probability'
    training_types = ['pick_success_probability']
    action = 'train'
    action_list = ['train', 'test']
    
    opts, args = getopt.getopt(sys.argv[1:],"ha:d:t:o:",["action=","datadir=","type=", "object="])
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

  
          
    if(action == 'train'):
        train(object_name, data_dir, train_type)
    
    #test()
    #test_dataGenerator(1,logfileName)   

if __name__ == '__main__':
    main()
