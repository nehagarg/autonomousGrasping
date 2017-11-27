
import os
from sklearn.externals import joblib
from sklearn.utils import check_array

PICK_ACTION_ID = 10
OPEN_ACTION_ID = 9
CLOSE_ACTION_ID = 8
NUM_PREDICTIONS = 18

#C++ function

def load_model(classifier_type, action, model_dir, num_predictions):
    model_filename = model_dir + "/" + classifier_type + "-" + repr(action)
    model_filenames = [model_filename + "-" + repr(i) + ".pkl" for i in range(0,num_predictions)]
    model_filenames.append(model_filename + '.pkl')
    
    models = [joblib.load(m)  for m in model_filenames if os.path.exists(m)]
    return models
    
def get_model_prediction(models, x,predict_prob = 0):
    y = []
    for model in models:
        if(predict_prob == 0 and len(models) > 1):
            
            y_ = model.predict([x])
            y.append(y_[0])
            if(len(y) == 15):
                debug_decision_tree(model,[list(x), list(x)])
        else:
            if(predict_prob == 0):
                y=model.predict([x])[0].tolist()
            else:
                y=model.predict_proba([x])[0].tolist()
            
        
    #print y
    return y


def debug_decision_tree(estimator,X_test ):
    import numpy as np
    # Using those arrays, we can parse the tree structure:
    
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold
    value = estimator.tree_.value
    print value.shape
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    """
    print("The binary tree structure has %s nodes and has "
          "the following tree structure:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()
    """
    # First let's retrieve the decision path of each sample. The decision_path
    # method allows to retrieve the node indicator functions. A non zero element of
    # indicator matrix at the position (i, j) indicates that the sample i goes
    # through the node j.

    node_indicator = estimator.decision_path(X_test)

    # Similarly, we can also have the leaves ids reached by each sample.

    leave_id = estimator.apply(X_test)
    print leave_id
    print threshold[leave_id]
    print value[leave_id]
    # Now, it's possible to get the tests that were used to predict a sample or
    # a group of samples. First, let's make it for the sample.

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    print('Rules used to predict sample %s: ' % sample_id)
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if(X_test[sample_id][feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        
        print("decision id node %s : (X_test[%s, %s] (= %s) %s %s)"
              % (node_id,
                 sample_id,
                 feature[node_id],
                 X_test[sample_id][feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))
    print "Here"
    # For a group of samples, we have the following common node.
    """
    sample_ids = [0, 1]
    common_nodes = (node_indicator.toarray()[sample_ids].sum(axis=0) ==
                    len(sample_ids))

    common_node_id = np.arange(n_nodes)[common_nodes]

    print("\nThe following samples %s share the node %s in the tree"
          % (sample_ids, common_node_id))
    print("It is %s %% of all nodes." % (100 * len(common_node_id) / n_nodes,))
    """


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
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if saso_string + object_name in f and f.endswith('.txt') and '_24_' in f]    
    #object_file_name1 = data_dir + saso_string + object_name + "_allActions.txt"
    #object_file_name2 = data_dir + saso_string + object_name + "_openAction.txt"
    #print "Loading files" + object_file_name1 + object_file_name2
    ans={}
    bad_list = ['data_for_regression/Cylinder_85/SASOData_0-005_Cylinder_85_24_37-38--1--1_allActions.txt']#238
    bad_list.append('data_for_regression/Cylinder_85/SASOData_0-005_Cylinder_85_24_10-11--1--1_allActions.txt')#211
    bad_list.append('data_for_regression/Cylinder_8/SASOData_0-005_Cylinder_8_24_26-27--1--1_allActions.txt') #77
    bad_list.append('data_for_regression/Cylinder_8/SASOData_0-005_Cylinder_8_24_33-34--1--1_allActions.txt') #84
    bad_list.append('data_for_regression/Cylinder_7/SASOData_0-005_Cylinder_7_24_19-20--1--1_openAction.txt') #620
    bad_list.append('data_for_regression/Cylinder_7/SASOData_0-005_Cylinder_7_24_32-33--1--1_openAction.txt') #633
    #data_for_regression/Cylinder_75/SASOData_0-005_Cylinder_75_24_15-16--1--1_allActions.txt #166
    #data_for_regression/Cylinder_75/SASOData_0-005_Cylinder_75_24_24-25--1--1_allActions.txt #175
    #data_for_regression/Cylinder_75/SASOData_0-005_Cylinder_75_24_26-27--1--1_allActions.txt #177
    #data_for_regression/Cylinder_75/SASOData_0-005_Cylinder_75_24_30-31--1--1_allActions.txt #181
    #data_for_regression/Cylinder_75/SASOData_0-005_Cylinder_75_24_32-33--1--1_openAction.txt #683
    #data_for_regression/Cylinder_75/SASOData_Cylinder_75_24_18-19--1--1_allActions.txt #419
    #data_for_regression/Cylinder_85/SASOData_0-005_Cylinder_85_24_26-27--1--1_allActions.txt #227
    
    print "Loading files" + " ".join(files)
    for file in files:
        print file
        #if  file not in bad_list:
        #    continue
        file_nan_count = 0
        with open(file, 'r') as f:
            for line in f:
                if 'nan' in line:
                    file_nan_count = file_nan_count + 1
                    continue
                sasor = get_data_from_line(line.strip())
                if('openAction' not in file):
                    if sasor['action'] != OPEN_ACTION_ID:
                        if sasor['action']==PICK_ACTION_ID:
                            #Assuming pick action will always be after a close action
                            sasor['touch_prev'] = ans[CLOSE_ACTION_ID][-1]['touch']
                        if sasor['action'] not in ans:
                            ans[sasor['action']]= []
                        if(sasor['reward'] > -999):
                            ans[sasor['action']].append(sasor)
                else:
                    if sasor['action'] == OPEN_ACTION_ID:
                        if sasor['action'] not in ans:
                            ans[sasor['action']]= []
                        if(sasor['reward'] > -999):
                            ans[sasor['action']].append(sasor)
        
        print file_nan_count        
        assert(file_nan_count < 5)                    
    return ans
def approx_equal(x1,x2,error):
    return (x1 - x2) < error and (x1 -x2) > -1*error
        
    
def write_config_in_file(filename, ans):
    
    from yaml import dump
    try:
        from yaml import CDumper as Dumper
    except ImportError:
        from yaml import Dump
    output = dump(ans, Dumper=Dumper)
    f = open(filename, 'w')
    f.write(output)                
def get_prediction_value(sasor, p, rel = True):
    if p < 7:
        if rel:
            ans = sasor['next_gripper'][p] - sasor['init_gripper'][p]
        else:
            ans =  sasor['next_gripper'][p]
        return round(ans,4)
    if p < 14:
        if rel:
            ans = sasor['next_object'][p - 7] -  sasor['init_object'][p - 7]
        else:
            ans = sasor['next_object'][p - 7]
        return round(ans,4)
    if p < 16:
        if rel:
            ans = sasor['next_joint_values'][p-14] - sasor['init_joint_values'][p-14]
        else:
            ans = sasor['next_joint_values'][p-14]
        return round(ans,5)
    if p < 18:
        ans = sasor['touch'][p-16]
        return round(ans,2)
def get_default_value(sasor, p): 
    if(sasor['action'] == 0):
        if p == 0:
            return sasor['init_gripper'][p]+0.01
    if(sasor['action'] == 1):
        if p == 0:
            return sasor['init_gripper'][p]+0.08
    if(sasor['action'] == 2):
        if p == 0:
            return sasor['init_gripper'][p]-0.01
    if(sasor['action'] == 3):
        if p == 0:
            return sasor['init_gripper'][p]-0.08
    if(sasor['action'] == 4):
        if p == 1:
            return sasor['init_gripper'][p]+0.01
    if(sasor['action'] == 5):
        if p == 1:
            return sasor['init_gripper'][p]+0.08
    if(sasor['action'] == 6):
        if p == 1:
            return sasor['init_gripper'][p]-0.01
    if(sasor['action'] == 7):
        if p == 1:
            return sasor['init_gripper'][p]-0.08
    if(sasor['action'] == 8):
        if p < 16 and p>=15:
            return 61.15*3.14/180
    if(sasor['action'] == 8):
        if p < 16 and p>=15:
            return 0
    if p ==16:
        if sasor['action'] == 8:
            return 0.91
        else:
            return 0.11
    if p== 17:
        if sasor['action'] == 8:
            return 1.01
        else:
            return 0.12
    return get_prediction_value(sasor, p)

def is_correct(p,actual,predicted):
    error_val = 0.005
    if p >15:
        error_val = .05
    if p > 13:
        error_val = 2*3.14/180.0

    if actual- predicted < error_val and actual - predicted > -1*error_val:
        return 1
    else:
        return 0

def get_yaml_earth(pred):
    yaml_out = {}
    yaml_out['coef'] = pred.coef_.tolist()[0]
    yaml_out['bf_knot'] = []
    yaml_out['bf_variable'] = []
    yaml_out['bf_reverse'] = []
    yaml_out['bf_type'] = []
    for bf in pred.basis_:
        if(not bf.is_pruned()):
            if('Intercept' in str(bf)):
                yaml_out['bf_type'] = ['Intercept']
                yaml_out['bf_knot'].append(0)
                yaml_out['bf_variable'].append(0)
                yaml_out['bf_reverse'].append(False)
            else:
                yaml_out['bf_type'] = ['Hinge']
                yaml_out['bf_knot'].append(bf.get_knot())
                yaml_out['bf_variable'].append(bf.get_variable())
                yaml_out['bf_reverse'].append(bf.get_reverse())
            
            
    return yaml_out
    
    
def predict_earth(pred,x):
    import numpy as np
    y_my = 0
    val_array = []
    i = 0
    for bf in pred.basis_:

        if(not bf.is_pruned()):
            val = 0
            if('Intercept' in str(bf)):
                val = 1.0
            else:
                if bf.get_reverse():
                    val = np.where(x[bf.get_variable()]  > bf.get_knot(), 0.0, bf.get_knot() - x[bf.get_variable()])
                else:
                    val = np.where(x[bf.get_variable()] <= bf.get_knot(), 0.0, x[bf.get_variable()] - bf.get_knot())
            y_my = y_my + pred.coef_[0][i]*val
            val_array.append(val)
            i = i+1

             

    if not approx_equal(y_my , pred.predict([x])[0], 0.000001):
        print "Here"
        print y_my
        print pred.predict([x[j]])[0]
        X , missing= pred._scrub_x([x], None)
        print X
        print x
        B = pred.transform(X, missing)
        print B
        print val_array
        print np.dot(B, l_reg[p].coef_.T)
"""
object_name : name of the object for which model is being learned e.g 9cm cylinder 1001, 1084 etc
data_dir : Directory containing SASO*.txt file
train_type: 1. Pick probability, 
            2. next action joint values, 
            3. next action gripper state, 
            4. next action object state
            5. next action touch values
"""

def train(object_name, data_dir, output_dir, train_type, classifier_type,learned_model = None, debug = False):
    from sklearn import linear_model, tree
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostRegressor
    if classifier_type == 'Earth':
        from pyearth import Earth
    import numpy as np
    have_graphviz = True
    try:
        import graphviz
    except:
        have_graphviz = False
    ans = None
    saso_data = load_data_file(object_name, data_dir)
    if train_type =='gripper_status':
        action_str = 'gs'
        actions = range(CLOSE_ACTION_ID + 1)
        x=[]
        y=[]
        x_index = []
        for action in actions:
            for sasor in saso_data[action]:
                #x_entry = sasor['touch_prev'] + sasor['init_joint_values'] 
                x_entry = sasor['next_joint_values'] 
                x_entry =  x_entry + sasor['next_gripper'] + sasor['next_object']
                x_entry.append(sasor['next_object'][0]-sasor['next_gripper'][0])
                x_entry.append(sasor['next_object'][1]-sasor['next_gripper'][1])
                x.append(x_entry)
                x_index.append(sasor['index'])
                if action == CLOSE_ACTION_ID :
                    y.append(1)
                else:
                    y.append(0) #gripper open
    if train_type == 'pick_success_probability':
        action_str = repr(PICK_ACTION_ID)
        x=[]
        y=[]
        x_index = []
        for sasor in saso_data[PICK_ACTION_ID]:
            #x_entry = sasor['touch_prev'] + sasor['init_joint_values'] 
            x_entry = sasor['init_joint_values'] 
            x_entry =  x_entry + sasor['init_gripper'][0:3] + sasor['init_object'][0:3]
            x_entry.append(sasor['init_object'][0]-sasor['init_gripper'][0])
            x_entry.append(sasor['init_object'][1]-sasor['init_gripper'][1])
            x.append(x_entry)
            x_index.append(sasor['index'])
            if sasor['reward'] > 0:
                y.append(1)
            else:
                y.append(0)
    if train_type in ['pick_success_probability', 'gripper_status']:
        if learned_model is not None:
            logistic = learned_model
        else:
            print classifier_type
            if classifier_type == 'DTC':
                logistic = DecisionTreeClassifier(criterion='entropy')
            else:
                logistic = linear_model.LogisticRegression(max_iter = 400, C = 1.0)
            logistic.fit(x,y)
            joblib.dump(logistic, output_dir + '/' + classifier_type + '-' + action_str + '.pkl') 
        ans = logistic
        print logistic.score(x,y)
        print logistic.get_params()
        print len(x)
        if classifier_type != 'DTC':
            print logistic.coef_
            print logistic.intercept_
            yaml_out = {}
            yaml_out['coef'] = logistic.coef_.tolist()[0]
            yaml_out['intercept'] = logistic.intercept_.tolist()[0]
            write_config_in_file(output_dir + '/' +  classifier_type + '-' + action_str+".yaml", yaml_out)
        else:
            print logistic.feature_importances_
             
            #feature_names=['t1','t2', 'j1', 'j2']
            feature_names=['j1', 'j2'] #Touch not required when object coordinates are known
            feature_names = feature_names + ['gx', 'gy','gz','gxx','gyy','gzz','gw'][0:3]
            feature_names = feature_names + ['ox', 'oy','oz','oxx','oyy','ozz','ow'][0:3]
            feature_names = feature_names + ['xrel','yrel']
            if have_graphviz:
                dot_data = tree.export_graphviz(logistic, out_file=None,
                                        feature_names = feature_names, filled=True) 
                graph = graphviz.Source(dot_data) 
                graph.render(output_dir + '/' +  classifier_type + '-' + action_str ) 
            yaml_out = {}
            yaml_out["max_depth"] = logistic.tree_.max_depth
            yaml_out["values"] = logistic.tree_.value
            yaml_out['n_nodes'] = logistic.tree_.node_count
            yaml_out['children_left'] = logistic.tree_.children_left
            yaml_out['children_right'] = logistic.tree_.children_right
            yaml_out['feature'] = logistic.tree_.feature
            yaml_out['threshold'] = logistic.tree_.threshold
            write_config_in_file(output_dir + '/' +  classifier_type + '-' + action_str+".yaml", yaml_out)
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
            y_c = []
            l_reg = []
            l_reg_c = []
            x_index = []
            for i in range(0,NUM_PREDICTIONS):
                y.append([])
                y_c.append([])
                l_reg.append('')
                l_reg_c.append('')
            for sasor in saso_data[action]: 
                if sasor['reward'] > -999: #discard invalid states
                    x_entry = sasor['init_joint_values'] 
                    x_entry =  x_entry + sasor['init_gripper'][0:3] + sasor['init_object'][0:3]
                    x_entry.append(sasor['init_object'][0]-sasor['init_gripper'][0])
                    x_entry.append(sasor['init_object'][1]-sasor['init_gripper'][1])
                    x.append(x_entry)
                    x_index.append(sasor['index'])
                    for p_ in predictions:
                        p = int(p_)
                        y[p].append(get_prediction_value(sasor,p))
                        y_default = get_default_value(sasor,p)
                        y_c[p].append(is_correct(p, y[p][-1], y_default))
                        """
                        try:
                            check_array(x)
                            check_array(y[p])
                        except:
                            print x[-1]
                            print y[p][-1]
                            print sasor['index']
                            assert(0==1)
                        """
                            
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
                    elif classifier_type in ['DTR', 'DTRM']:
                         l_reg[p] = DecisionTreeRegressor()
                    elif classifier_type == 'DTC':
                         l_reg[p] = DecisionTreeClassifier()
                    elif classifier_type == 'Earth':
                         l_reg[p] = Earth()
                    elif classifier_type == 'AdaLinear':
                         l_reg[p] = AdaBoostRegressor(linear_model.LinearRegression())
                    else:
                        l_reg[p] = linear_model.LinearRegression()
                    if classifier_type == 'DTRM':
                        l_reg[p].fit(x,np.transpose(np.array(y)))
                    elif classifier_type == 'DTC':
                         l_reg[p].fit(x,y_c[p])
                    else:
                        l_reg[p].fit(x,y[p])
                    joblib.dump(l_reg[p], output_dir + '/' + classifier_type+ "-"+repr(action)+"-"+repr(p) +'.pkl') 
                ans[action][p] = l_reg[p]
                
                if classifier_type == 'DTRM':
                    print repr(action) + " " + repr(p) + " " + repr(l_reg[p].score(x,np.transpose(np.array(y))))
                elif classifier_type == 'DTC':
                    print repr(action) + " " + repr(p) + " " + repr(l_reg[p].score(x,y_c[p]))
                else:
                    print repr(action) + " " + repr(p) + " " + repr(l_reg[p].score(x,y[p]))
                print l_reg[p].get_params()
                if classifier_type not in [ 'SVR', 'DTR', 'DTRM', 'AdaLinear', 'DTC']:
                    print l_reg[p].coef_
                if classifier_type not in ['DTR', 'DTRM', 'AdaLinear', 'DTC','Earth']:
                    print l_reg[p].intercept_
                if classifier_type in ['Earth']:
                    for j in range(0,len(x)):
                        predict_earth(l_reg[p],x[j])
                    print l_reg[p].summary()
                if learned_model is None:    
                    if classifier_type in ['DTR', 'DTRM','AdaLinear','DTC']:

                        print l_reg[p].feature_importances_
                        
                        feature_names=['j1', 'j2']
                        feature_names = feature_names + ['gx', 'gy','gz','gxx','gyy','gzz','gw'][0:3]
                        feature_names = feature_names + ['ox', 'oy','oz','oxx','oyy','ozz','ow'][0:3]
                        feature_names = feature_names + ['xrel','yrel']
                        if have_graphviz:
                            dot_data = tree.export_graphviz(l_reg[p], out_file=None,
                                            feature_names = feature_names, filled=True) 
                            graph = graphviz.Source(dot_data) 
                            graph.render(output_dir + '/' + classifier_type+"-"+repr(action)+"-"+repr(p))
                        yaml_out = {}
                        yaml_out['max_depth'] = l_reg[p].tree_.max_depth
                        yaml_out["values"] = l_reg[p].tree_.value.tolist()
                        yaml_out['n_nodes'] = l_reg[p].tree_.node_count
                        yaml_out['children_left'] = l_reg[p].tree_.children_left.tolist()
                        yaml_out['children_right'] = l_reg[p].tree_.children_right.tolist()
                        yaml_out['feature'] = l_reg[p].tree_.feature.tolist()
                        yaml_out['threshold'] = l_reg[p].tree_.threshold.tolist()
                        write_config_in_file(output_dir + '/' + classifier_type+"-"+repr(action)+"-"+repr(p)+".yaml", yaml_out)
                    if classifier_type in ['Earth']:
                        yaml_out = get_yaml_earth(l_reg[p])
                        write_config_in_file(output_dir + '/' + classifier_type+"-"+repr(action)+"-"+repr(p)+".yaml", yaml_out)
                    
                    
                if classifier_type == 'DTRM':
                    i = 0
                    y_bar = l_reg[p].predict([x[i]])
                    print x_index[i]
                    print x[i] 
                    y_t = np.transpose(np.array(y))
                    print repr(y_t[i]) + ' Prediction ' + repr(y_bar)
                    break
                if debug:
                    for i in range(0,len(x)):
                        y_bar = l_reg[p].predict([x[i]])
                        if classifier_type=='DTC':
                            if y_bar != y_c[p][i]:
                                print x_index[i]
                                print x[i] 
                                print y_c[p][i]
                                print y[p][i]
                                print l_reg[p].predict_proba([x[i]])
                        else:
                            if is_correct(p,y_bar,y[p][i]) == 0:
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
    import rospkg
    import sys
    import getopt
    
    object_name = '9cm'
    rospack = rospkg.RosPack()
    grasping_ros_mico_path = rospack.get_path('grasping_ros_mico')
    classifier_type = 'linear'
    output_dir = grasping_ros_mico_path +'/scripts/decision_trees'
    output_dir = 'data_low_friction_table_exp_ver5/regression_models'
    
    #data_dir = grasping_ros_mico_path + "/data_low_friction_table_exp_ver5"
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
              
           

    #output_dir = output_dir+"/"+object_name
          
    if(action == 'train'):
        #train_object_name = object_name
        #test_object_name = object_name
        #if os.path.exists(data_dir + "/SASOData_0-005_Cylinder_" + object_name + "_allActions.txt"):
        #    train_object_name = "/SASOData_0-005_Cylinder_" + object_name
        
        train_object_name = 'SASOData_0-005_' + object_name
        test_object_name =  'SASOData_' + object_name
        data_dir = data_dir + "/" + object_name
        output_dir = output_dir + "/" + object_name
        ans = train(train_object_name, data_dir, output_dir, train_type, classifier_type)
        train(test_object_name, data_dir, output_dir, train_type, classifier_type,ans, True)    
    
    #test()
    #test_dataGenerator(1,logfileName)   

if __name__ == '__main__':
    main()
