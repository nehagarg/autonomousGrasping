
import os
#from sklearn.externals import joblib
#from sklearn.utils import check_array
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import time
import yaml
from tensorflow.keras import backend as K
rom tensorflow.keras.utils import plot_model

PICK_ACTION_ID = 10
OPEN_ACTION_ID = 9
CLOSE_ACTION_ID = 8
NUM_PREDICTIONS = 18


#Lines contain name of vision file also
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


def load_data(object_name_list, data_dir, object_id_mapping_file):
    object_id_mapping = yaml.load(file(object_id_mapping_file, 'r'))
    ans = {}
    for object_name in object_name_list:
        final_data_dir = os.path.join(data_dir , o
        files = [os.path.join(final_data_dir, f) for f in os.listdir(data_dir) if object_name in f and f.endswith('.txt') and '_24_' in f]
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
                    sasor['object_id'] = object_id_mapping
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


def train(object_name_list, data_dir, object_id_mapping_file):
    pass


def map_actions(x):
    y = K.switch(K.greater(x,8), x-4,x*0.5)
    return K.flatten(K.one_hot(y,3))



def keras_functional_transition_model():
    num_object_bits = 2 # For only 3 objects
    num_action_bits = 3 # For 11 actions

    #State inputs
    gripper_2D_pos = tf.keras.Input(shape=(2,), name='gripper_2D_pos_input')
    object_2D_pos = tf.keras.Input(shape=(2,), name='object_2D_pos_input')
    object_thetaz = tf.keras.Input(shape=(1,), name='object_thetaz_input')
    object_id = tf.keras.Input(shape = (num_object_bits,), name='object_id_input')
    previous_action_id = tf.keras.Input(shape = (num_action_bits,), name='previous_action_id_input')
    gripper_finger_joint_angles = tf.keras.Input(shape = (2,), name='gripper_finger_joint_angle_input')

    action_id = tf.keras.Input(shape = (1,), name='action_id_input')
    mapped_action = layers.Lambda(map_actions)(action_id)
    transition_model = tf.keras.Model(inputs=action_id, outputs=mapped_action)
    return transition_model

def get_keras_state_transition_model():

    input_size = 5 + num_object_bits
    model = tf.keras.Sequential()
    model.add(layers.Dense(16, activation='sigmoid', input_shape=(input_size,), name = 'Intermediate'))
    output_bits = input_size - num_object_bits
    output_bits = 200*160
    model.add(layers.Dense(output_bits, name='Output'))
    return model

def get_keras_observation_model():
    pass

def train_model():
    #Currently generating random data
    #sess = K.get_session()
    model = keras_functional_transition_model()
    model.summary()
    X = np.array()[[1,2,3,4,5,6,7,8,9,10]]).transpose()
    Y = model.predict(X)
    print Y
    plot_model(model,
               to_file='vae_mlp.png',
               show_shapes=True)

def save_model():
    sess = K.get_session()
    model = get_keras_state_transition_model()
    print model.name
    model.summary()
    node_names = [node.op.name for node in model.outputs]
    print node_names
    node_names = [node.op.name for node in model.inputs]
    print node_names
    sess.run(tf.global_variables_initializer())

    #X = np.array([[0.1,0.2,0.3,0.4,0.5,0,1]])
    X = np.random.random((100, 7))
    print X.shape
    X1 = X.transpose()
    print X1.shape
    for i in range(0,10):
        start = time.time()
        Y = model.predict(X)
        end = time.time()
        print repr(end-start)
    #print Y
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess, './exported/my_model')
    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb", as_text=False)
    tf.train.write_graph(sess.graph, '.', "./exported/graph.pb_txt", as_text=True)

if __name__ == '__main__':
    #save_model()
    object_id_mapping_file = 'ObjectNameToIdMapping.yaml'
    data_dir = 'data_low_friction_table_exp_wider_object_workspace_ver8/data_for_regression'
    object_id_mapping = yaml.load(file(object_id_mapping_file, 'r'))
    object_name_list = object_id_mapping.keys()
    train(object_name_list, data_dir, object_id_mapping_file)
