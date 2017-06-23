import sys
import getopt
sys.path.append('../')
import tensorflow as tf
import json
import os
#import math
import time
import deepLearning_data_generator as traces
from plot_despot_results import plot_scatter_graph
#from tensorflow.python.framework import dtypes
#from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn, rnn_cell
import random
import config
from sklearn import svm
from sklearn.externals import joblib
import model as rnn_model
import matplotlib.pyplot as plt
#from model import Encoder, Seq2SeqModel, DataGenerator, is_stump, is_pad, PROBLEM_NAME

import numpy as np

class Seq2SeqModelExt(rnn_model.Seq2SeqModel):
    def predict_and_return_state(self, input_values, summary = False):
        input_feed = {}
        for i, inp in enumerate(self.testing_graph.inputs):
            #print inp.name
            #print input_values[:,i,:]
            input_feed[inp.name] = input_values[:, i, :]
        #probs, states, output_summary = self.session.run([self.testing_graph.probs, self.testing_graph._outputs, self.testing_graph.merged], input_feed)
        output_summary = None
        if summary:
            output_summary, probs, states = self.session.run([self.testing_graph.merged, self.testing_graph.probs, self.testing_graph._outputs], feed_dict=input_feed)
        else:
            probs, states = self.session.run([ self.testing_graph.probs, self.testing_graph._outputs], feed_dict=input_feed)
        
        probs = np.array(probs)
        probs = np.transpose(probs, (1,0,2))
        #output_summary=None
        return probs, states, output_summary

def get_svm_file_name(file_prefix,nu=0.1, kernel="rbf", gamma=0.1 ):
    filename = 'nu_'+ repr(nu).replace('.', '_') + '_kernel_' + kernel +  '_gamma_' + repr(gamma).replace('.', '_') + '_'
    filename = filename + file_prefix + '_svm.pkl'
    return filename

def compute_svm(data,output_dir,file_prefix, nu=0.1, kernel="rbf", gamma=0.1):
    clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    clf.fit(data)
    filename = get_svm_file_name(file_prefix, nu, kernel, gamma)
    joblib.dump(clf, output_dir + '/' + filename) 
    return clf

def get_learning_model_output(model_name, raw_input, batch_size):
    data_generator = rnn_model.DataGenerator(batch_size, raw_input)
    seq_length = data_generator.seq_length 
    print len(data_generator.xseqs)
    print data_generator.seq_length
    print 'data number of batches', data_generator.num_batches
    with tf.Session(config=config.get_tf_config()) as sess:
        h_to_a_model = load_model(model_name, sess, seq_length, data_generator.batch_size)
        x, y = data_generator.next_batch()
        target = np.argmax(y, axis=2) #target  = batch size*seq length *1
        probs, outputs, image_summary = h_to_a_model.predict_and_return_state(x, summary = False) #output = seqlength*batch size* hiddenunits
        return (probs, outputs , target ,x)
def get_svm_model_output(svm_model_prefix, svm_model_input):
    correct_prediction_svm = joblib.load(svm_model_prefix+ 'correct_prediction_svm.pkl') 
    wrong_prediction_svm = joblib.load(svm_model_prefix + 'wrong_prediction_svm.pkl')
    y_correct_predict = correct_prediction_svm.predict(svm_model_input)   
    y_wrong_predict = wrong_prediction_svm.predict(svm_model_input)
    return (y_correct_predict, y_wrong_predict)
    
def train(model_name, output_dir, model_input= None):
    
    data_generator = rnn_model.DataGenerator(1, model_input)
    num_val_batches = data_generator.num_batches
    print len(data_generator.xseqs)
    print data_generator.seq_length
    print 'data number of batches', data_generator.num_batches
    #num_val_batches = 1
    seq_length = data_generator.seq_length
    #generate training data for two svms
    with tf.Session(config=config.get_tf_config()) as sess:
        h_to_a_model = load_model(model_name, sess, seq_length)
        #summary_writer = tf.summary.FileWriter('output', sess.graph, seq_length)
        
        
        correct_prediction_outputs = []
        wrong_prediction_outputs = []
        for _ in xrange(num_val_batches):
            x, y = data_generator.next_batch() # x/y = batch size*seq length*input_length/output length
            target = np.argmax(y, axis=2) #target  = batch size*seq length *1
            probs, outputs, image_summary = h_to_a_model.predict_and_return_state(x, summary = False) #output = seqlength*batch size* hiddenunits
            #summary_writer.add_summary(image_summary)
            prediction = np.argmax(probs, axis=2) # prediction = batch size*seq length * 1
            #print data_generator.xseqs
            #print y
            #print target[0]
            #print prediction[0]
            #print outputs[0:2]
            correct_prediction = target==prediction #batch size *seq length * 1
            for i in xrange(len(outputs)):
                if (not rnn_model.is_stump(x[0][i]) ) and (not rnn_model.is_pad(x[0][i]) ): #not stump nd pad
                    if correct_prediction[0][i]:
                        correct_prediction_outputs.append(outputs[i][0])
                    else:
                        wrong_prediction_outputs.append(outputs[i][0])
        
        print 'num correct prediction traces', len(correct_prediction_outputs)
        print 'num wrong prediction traces', len(wrong_prediction_outputs)
        correct_prediction_svm = compute_svm(correct_prediction_outputs, output_dir, 'correct_prediction')
        #y_correct_predict = correct_prediction_svm.predict(correct_prediction_outputs)
        wrong_prediction_svm = compute_svm(wrong_prediction_outputs, output_dir, 'wrong_prediction')    
        #y_wrong_predict = wrong_prediction_svm.predict(wrong_prediction_outputs)
        
        #print y_correct_predict
        #print y_wrong_predict


def test(model_name, svm_model_prefix, model_input, action = 'test'):
    data_generator = rnn_model.DataGenerator(1, model_input)
    num_val_batches = data_generator.num_batches
    if action == 'testBatch':
        print len(data_generator.xseqs)
        print data_generator.seq_length
        print 'data number of batches', data_generator.num_batches
    #num_val_batches = 1
    seq_length = data_generator.seq_length   
    with tf.Session(config=config.get_tf_config()) as sess:
        h_to_a_model = load_model(model_name, sess, seq_length)
        #summary_writer = tf.summary.FileWriter('output', sess.graph)
        
        
        prediction_outputs = []
        num_seen_predictions_correct_svm = 0
        num_unseen_prediction_correct_svm = 0
        num_seen_predictions_wrong_svm = 0
        num_unseen_prediction_wrong_svm = 0
        
        for _ in xrange(num_val_batches):
            x, y = data_generator.next_batch() # x/y = batch size*seq length*input_length/output length
            #target = np.argmax(y, axis=2) #target  = batch size*seq length *1
            probs, outputs, image_summary = h_to_a_model.predict_and_return_state(x, summary = False) #probs = batch size*seq length * output length #output = seqlength*batch size* hiddenunits

                
        
            probs_without_dummy_actions = [i[:-2] for i in probs[0] ]
            prediction = np.argmax([probs_without_dummy_actions], axis=2)
            #prob_prediction = get_prob_prediction(probs_without_dummy_actions)
        
        
            #print prediction[0]
            
            #if prob_prediction == -1:
            #    print prediction[0][-2]
            #else:
            #    print prob_prediction
            if action in [ 'testModel']:
                print prediction[0][-2]
                print ' '.join(str(p) for p in probs_without_dummy_actions[-2])
                print prediction[0][-2]
                print prediction[0]
                print data_generator.yseqs
                print model_name
                return
        
        
            #summary_writer.add_summary(image_summary)
            #prediction = np.argmax(probs, axis=2) # prediction = batch size*seq length * 1
            #print data_generator.xseqs
            #print y
            #print target[0]
            #print prediction[0]
            #print outputs[0:2]

            for i in xrange(len(outputs)):
                if (not rnn_model.is_stump(x[0][i]) ) and (not rnn_model.is_pad(x[0][i]) ): #not stump nd pad
                    prediction_outputs.append(outputs[i][0])
        
        if len(prediction_outputs) == 0: #Initial stump
            print 1
            print -1
            print prediction[0][-2]
        else:
            correct_prediction_svm = joblib.load(svm_model_prefix+ 'correct_prediction_svm.pkl') 
            wrong_prediction_svm = joblib.load(svm_model_prefix + 'wrong_prediction_svm.pkl') 

            svm_input = [prediction_outputs[-1]]
            if action == 'testBatch':
                svm_input = prediction_outputs

            y_correct_predict = correct_prediction_svm.predict(svm_input)   
            y_wrong_predict = wrong_prediction_svm.predict(svm_input)
            
            if action != 'testBatch':
                print y_correct_predict[-1]
                print y_wrong_predict[-1]
                print prediction[0][-2]
                #if(y_correct_predict[-1] == 1) and (y_wrong_predict[-1] == -1):
                #    print 1
                #else:
                #    print 0
                print y_correct_predict
                print y_wrong_predict
                #print data_generator.xseqs
            if action == 'testBatch':
                num_seen_predictions_correct_svm = num_seen_predictions_correct_svm + sum(xx for xx in y_correct_predict if xx == 1)
                num_unseen_prediction_correct_svm = num_unseen_prediction_correct_svm + sum(xx for xx in y_correct_predict if xx == -1)
                num_seen_predictions_wrong_svm = num_seen_predictions_wrong_svm + sum(xx for xx in y_wrong_predict if xx == 1)
                num_unseen_prediction_wrong_svm = num_unseen_prediction_wrong_svm + sum(xx for xx in y_wrong_predict if xx == -1)
        if action != 'testBatch' :

            print ' '.join(str(p) for p in probs_without_dummy_actions[-2])
            print prediction[0][-2]
            print prediction[0]
            print data_generator.yseqs
            print model_name

        if action == 'testBatch':
            print "Num seen predictions (correct svm, wrong svm):" + repr(num_seen_predictions_correct_svm) + "," + repr(num_seen_predictions_wrong_svm)
            print "Num unseen predictions (corect svm, wrong svm):" + repr(num_unseen_prediction_correct_svm) + "," + repr(num_unseen_prediction_wrong_svm)
            #print x

def debug_with_pca(model_name, model_input, svm_model_prefix, model_input_test = None):
    pca = joblib.load("toy_pca_model.pkl")
    (probs,outputs,target,x) = get_learning_model_output(model_name, model_input, -1)
    
    prediction = np.argmax(probs, axis=2)
    correct_prediction = target==prediction
    batch_size = len(probs)
    correct_prediction_outputs = []
    wrong_prediction_outputs = []
    for j in xrange(batch_size):
            i = 1
            if (not rnn_model.is_stump(x[j][i]) ) and (not rnn_model.is_pad(x[j][i]) ): #not stump nd pad
                if correct_prediction[j][i]:
                    correct_prediction_outputs.append(outputs[i][j])
                else:
                    wrong_prediction_outputs.append(outputs[i][j])
    
    print len(correct_prediction_outputs + wrong_prediction_outputs)
   
    (y_correct_predict, y_wrong_predict) = get_svm_model_output(svm_model_prefix, correct_prediction_outputs + wrong_prediction_outputs)
    transformed_correct_prediction_outputs = pca.transform(correct_prediction_outputs)
    transformed_wrong_prediction_outputs = pca.transform(wrong_prediction_outputs)
    scatter_x = []
    scatter_y = []
    scatter_colors = []
    scatter_markers = []
    csv_file = open('toy_test_outputs.csv', 'w')
    csv_file.write("x,y,type,correct_predict,wrong_predict \n")
    
    for i in range(0,len(correct_prediction_outputs + wrong_prediction_outputs)):
        data_type = 'correct'
        if i < len(correct_prediction_outputs):
            scatter_x.append(transformed_correct_prediction_outputs[i][0])
            scatter_y.append(transformed_correct_prediction_outputs[i][1])
            scatter_colors.append('green')
        else:
            data_type = 'wrong'
            scatter_x.append(transformed_wrong_prediction_outputs[i-len(transformed_correct_prediction_outputs)][0])
            scatter_y.append(transformed_wrong_prediction_outputs[i-len(transformed_correct_prediction_outputs)][1])
            scatter_colors.append('red')
        
        csv_file.write(repr(scatter_x[-1]) + "," + repr(scatter_y[-1]) + ",")
        csv_file.write(data_type + "," + repr(y_correct_predict[i]) + "," + repr(y_wrong_predict[i]) + "\n")
        
        if(y_correct_predict[i]*y_wrong_predict[i] == 1):
            if(y_correct_predict[i] == 1):
                scatter_markers.append('^')
            else:
                scatter_markers.append('v')
        else:
            if(y_correct_predict[i] == 1):
                scatter_markers.append('+')
            else:
                scatter_markers.append('o')
    
    csv_file.close()        
    area = np.pi * (15 * 1)**2
    m = np.array(scatter_markers)

    unique_markers = set(m)  # or yo can use: np.unique(m)

    for um in unique_markers:
        mask = m == um 
        # mask is now an array of booleans that van be used for indexing  
        plt.scatter(np.array(scatter_x)[mask], np.array(scatter_y)[mask], marker=um, c = np.array(scatter_colors)[mask], s = area)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data')
    plt.show()
    
    
def debug_with_pca_train(model_name, model_input, svm_model_prefix, model_input_test = None):
    (probs,outputs,target,x) = get_learning_model_output(model_name, model_input, -1)
    
    prediction = np.argmax(probs, axis=2)
    correct_prediction = target==prediction
    batch_size = len(probs)
    correct_prediction_outputs = []
    wrong_prediction_outputs = []
    for j in xrange(batch_size):
        for i in xrange(len(outputs)):
            if (not rnn_model.is_stump(x[j][i]) ) and (not rnn_model.is_pad(x[j][i]) ): #not stump nd pad
                if correct_prediction[j][i]:
                    correct_prediction_outputs.append(outputs[i][j])
                else:
                    wrong_prediction_outputs.append(outputs[i][j])
    
    #print len(correct_prediction_outputs + wrong_prediction_outputs)
   
    (y_correct_predict, y_wrong_predict) = get_svm_model_output(svm_model_prefix, correct_prediction_outputs + wrong_prediction_outputs)
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2)
    pca.fit(correct_prediction_outputs + wrong_prediction_outputs)
    joblib.dump(pca, "toy_pca_model.pkl") 
    
    print pca.explained_variance_ratio_
    #print pca.explained_variance_
    print len(pca.components_)
    transformed_correct_prediction_outputs = pca.transform(correct_prediction_outputs)
    transformed_wrong_prediction_outputs = pca.transform(wrong_prediction_outputs)
    scatter_x = []
    scatter_y = []
    scatter_colors = []
    scatter_markers = []
    csv_file = open('toy_outputs.csv', 'w')
    csv_file.write("x,y,type,correct_predict,wrong_predict \n")
    
    for i in range(0,len(correct_prediction_outputs + wrong_prediction_outputs)):
        data_type = 'correct'
        if i < len(correct_prediction_outputs):
            scatter_x.append(transformed_correct_prediction_outputs[i][0])
            scatter_y.append(transformed_correct_prediction_outputs[i][1])
            scatter_colors.append('green')
        else:
            data_type = 'wrong'
            scatter_x.append(transformed_wrong_prediction_outputs[i-len(transformed_correct_prediction_outputs)][0])
            scatter_y.append(transformed_wrong_prediction_outputs[i-len(transformed_correct_prediction_outputs)][1])
            scatter_colors.append('red')
        
        csv_file.write(repr(scatter_x[-1]) + "," + repr(scatter_y[-1]) + ",")
        csv_file.write(data_type + "," + repr(y_correct_predict[i]) + "," + repr(y_wrong_predict[i]) + "\n")
        
        if(y_correct_predict[i]*y_wrong_predict[i] == 1):
            if(y_correct_predict[i] == 1):
                scatter_markers.append('^')
            else:
                scatter_markers.append('v')
        else:
            if(y_correct_predict[i] == 1):
                scatter_markers.append('+')
            else:
                scatter_markers.append('o')
    
    csv_file.close()        
    area = np.pi * (15 * 1)**2
    m = np.array(scatter_markers)

    unique_markers = set(m)  # or yo can use: np.unique(m)

    for um in unique_markers:
        mask = m == um 
        # mask is now an array of booleans that van be used for indexing  
        plt.scatter(np.array(scatter_x)[mask], np.array(scatter_y)[mask], marker=um, c = np.array(scatter_colors)[mask], s = area)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data')
    plt.show()
    
    
    
    

def test_with_server(model_input, action = 'testServerWithSwitching', random_id  = None):
    #This code has only been tested for learning model
    #Needs to be tested for switching server
    import rospy        
    from learning_ros_server.srv import LearningServer, UnseenScenarioServer
    learning_server_name = get_learning_server_name(random_id)
    switching_server_name = get_switching_server_name(random_id)
    
    rospy.wait_for_service(learning_server_name)
    learning_predictor = rospy.ServiceProxy(learning_server_name, LearningServer)
    resp = learning_predictor(model_input)
    prediction = resp.prediction
    last_output = resp.last_activation
    print prediction[resp.ans_index]
    
    if action == 'testServerWithSwitching':
        if len(last_output) == 0:
            print 1
            print -1
        else:
    
            #req = UnseenScenarioServerRequest()
            #req.rnn_activations_batch = svm_input
            rospy.wait_for_service(switching_server_name)
            switching_predictor = rospy.ServiceProxy(switching_server_name, UnseenScenarioServer)
            resp = switching_predictor(last_output)
            y_correct_predict = resp.correct_predictions
            y_wrong_predict = resp.wrong_predctions
            print y_correct_predict[-1]
            print y_wrong_predict[-1]
            #if(y_correct_predict[-1] == 1) and (y_wrong_predict[-1] == -1):
            #    print 1
            #else:
            #    print 0
            print y_correct_predict
            print y_wrong_predict
    

    print prediction
    print random_id

    
    
def get_server_name(server_name, random_id):
    ans = server_name
    if random_id is not None:
        ans = server_name + "_" + random_id
    return ans

def get_learning_server_name(random_id):
    return get_server_name("learning_server", random_id)

def get_switching_server_name(random_id):
    return get_server_name("unseen_scenario_server", random_id)

def load_model(model_name, sess, seq_length = None, batch_size = 1):
    model_dir = os.path.dirname(model_name + '.meta')
    model_config_file = './output/' + model_dir+ "/params.yaml"
    import yaml
    with open(model_config_file, 'r') as stream:
        model_params = yaml.load(stream)
    model = model_params['model']
    hidden_units = model_params['hidden_units']
    num_layers = model_params['num_layers']
    prob_config = config.get_problem_config(rnn_model.PROBLEM_NAME)
    if seq_length is None:
        seq_length = prob_config['max_sequence_length'] + 1
    observation_length = prob_config['input_length']
    action_length = prob_config['output_length']
    
    
    encoder = rnn_model.Encoder(action_length, observation_length)
    input_length = encoder.size_x()
    print input_length
    output_length = encoder.size_y()
    
    start = time.time()
    model = Seq2SeqModelExt(session=sess,
                hidden_units=hidden_units,
                model=model,
                num_layers=num_layers,
                seq_length=seq_length,
                input_length=input_length,
                output_length=output_length,
                batch_size= batch_size,
                scope="model")
    end = time.time()
    model_create_time = end-start
    #model.load('vrep/version1/model.ckpt-967')
    model.load(model_name)
    start = time.time()
    model_load_time = start-end
    return model


def handle_unseen_scenario_server_request(req):
    global server_correct_prediction_svm
    global server_wrong_prediction_svm
    from learning_ros_server.srv import UnseenScenarioServerResponse
    y_correct_predict = correct_prediction_svm.predict([req.rnn_activations_batch])   
    y_wrong_predict = wrong_prediction_svm.predict([req.rnn_activation_batch])
    ans = UnseenScenarioServerResponse()
    ans.correct_predictions = y_correct_predict
    ans.wrong_predictions = y_wrong_predict
    return ans
    
    
    
    
def launch_unseen_scenario_server(svm_model_prefix, random_id):
    import rospy
    from learning_ros_server.srv import UnseenScenarioServer
    server_name = get_switching_server_name(random_id)
    rospy.init_node(server_name)
    global server_correct_prediction_svm
    global server_wrong_prediction_svm
    server_correct_prediction_svm = joblib.load(svm_model_prefix+ 'correct_prediction_svm.pkl') 
    server_wrong_prediction_svm = joblib.load(svm_model_prefix + 'wrong_prediction_svm.pkl') 
    rospy.Service(server_name, UnseenScenarioServer, handle_unseen_scenario_server_request)
    rospy.spin()


def handle_learning_server_request(req):
    global h_to_a_model
    from learning_ros_server.srv import LearningServerResponse
    prob_config = config.get_problem_config(rnn_model.PROBLEM_NAME)
    my_seq_length = prob_config['max_sequence_length'] + 1
    data_generator = rnn_model.DataGenerator(1, req.input, my_seq_length)
    x,y = data_generator.next_batch()
    ans_index = len(data_generator.seqs[0]) - 1
    probs, outputs, image_summary = h_to_a_model.predict_and_return_state(x, summary = False) #probs = batch size*seq length * output length #output = seqlength*batch size* hiddenunits
            #summary_writer.add_summary(image_summary)
    probs_without_dummy_actions = [i[:-2] for i in probs[0] ]
    prediction = np.argmax([probs_without_dummy_actions], axis=2)
    prediction_outputs = []
    ans = LearningServerResponse()
    ans.last_activation = []
    for i in xrange(len(outputs)):
        if (not rnn_model.is_stump(x[0][i]) ) and (not rnn_model.is_pad(x[0][i]) ): #not stump nd pad
            prediction_outputs.append(outputs[i][0])
    if len(prediction_outputs) > 0:
        ans.last_activation = prediction_outputs[-1]
    
    
    ans.prediction = prediction[0]
    ans.ans_index = ans_index
    print data_generator.yseqs
    
    return ans
    
    
def launch_learning_server(model_name, random_id):
    import rospy
    from learning_ros_server.srv import LearningServer 
    
    server_name = get_learning_server_name(random_id)
    rospy.init_node(server_name)

    global h_to_a_model
    
    with tf.Session(config=config.get_tf_config()) as sess:
        h_to_a_model = load_model(model_name, sess)
        rospy.Service(server_name, LearningServer, handle_learning_server_request)
        rospy.spin()
    
    
    
    
def main():
    # running parameters
    #batch_size = 1
    #test_dataGenerator(batch_size)
    #train()
    
    action = None
    model_name = None
    model_input = None
    output_dir = None
    random_id = None
    #global PROBLEM_NAME
    action_list = ['testModel', 'testServer', 'testServerWithSwitching', "train", "test", 'testBatch', 'launch_learning_server', 'launch_unseen_scenario_server']
    action_list.append('pca_debug')
    opts, args = getopt.getopt(sys.argv[1:],"ha:m:i:o:p:r:",["action=","model=","input=", "outdir=", "problem="])
    #print opts
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'joint_training_model.py -a <train|test|testBatch> -m <rnn_model_name> -i <logfilename|seq|training data version> -o <svm output dir for train|svm model file prefix for test>'
         sys.exit()
      elif opt in ("-a","--action" ):
          action = arg
          if action not in action_list:
              action = raw_input("Please specify correction action[train|test]:")
      elif opt in ("-m", "--model"):
         model_name = arg
      elif opt in ("-i", "--input"):
         model_input = arg
      elif opt in ("-o", "--outdir"):
          output_dir = arg
      elif opt in ("-p", "--problem"):
          #PROBLEM_NAME = arg
          rnn_model.PROBLEM_NAME = arg
      elif opt == '-r':
          random_id = arg    
        
    if action == 'train':
        train(model_name, output_dir, model_input)
    elif action in ['test', 'testBatch', 'testModel']:
        test(model_name, output_dir, model_input, action)
    elif action == 'launch_learning_server':
        launch_learning_server(model_name, random_id)
    elif action == 'launch_unseen_scenario_server':
        launch_unseen_scenario_server(output_dir, random_id)
    elif action in ['testServer', 'testServerWithSwitching']:
        test_with_server(model_input, action, random_id)
    elif action =='pca_debug':
        debug_with_pca(model_name, model_input, output_dir, None)
    
    #test()
    #test_dataGenerator(1,logfileName)   

if __name__ == '__main__':
    main()
