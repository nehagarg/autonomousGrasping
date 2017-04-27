import sys
import getopt
sys.path.append('../')
import tensorflow as tf
import json
import os
#import math
import time
import deepLearning_data_generator as traces
#from tensorflow.python.framework import dtypes
#from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn, rnn_cell
import random
import config
from sklearn import svm
from sklearn.externals import joblib
from model import Encoder, Seq2SeqModel, DataGenerator, is_stump, is_pad, PROBLEM_NAME

import numpy as np

class Seq2SeqModelExt(Seq2SeqModel):
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
    
def train(model_name, output_dir, model_input= None):
    
    data_generator = DataGenerator(1, model_input)
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
                if x[0][i][-1] == 0 and x[0][i][-2] == 0 : #not stump nd pad
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
    #generate training data for two svms
    data_generator = DataGenerator(1, model_input)
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
            probs, outputs, image_summary = h_to_a_model.predict_and_return_state(x, summary = False) #output = seqlength*batch size* hiddenunits
            #summary_writer.add_summary(image_summary)
            prediction = np.argmax(probs, axis=2) # prediction = batch size*seq length * 1
            #print data_generator.xseqs
            #print y
            #print target[0]
            #print prediction[0]
            #print outputs[0:2]
            
            for i in xrange(len(outputs)):
                if (not is_stump(x[0][i]) ) and (not is_pad(x[0][i]) ): #not stump nd pad
                    prediction_outputs.append(outputs[i][0])
        
        if len(prediction_outputs) == 0: #Initial stump
            print 1
            print -1
        else:
            correct_prediction_svm = joblib.load(svm_model_prefix+ 'correct_prediction_svm.pkl') 
            wrong_prediction_svm = joblib.load(svm_model_prefix + 'wrong_prediction_svm.pkl') 
            if action == 'test' :
                y_correct_predict = correct_prediction_svm.predict([prediction_outputs[-1]])   
                y_wrong_predict = wrong_prediction_svm.predict([prediction_outputs[-1]])
        
                print y_correct_predict[-1]
                print y_wrong_predict[-1]
                #if(y_correct_predict[-1] == 1) and (y_wrong_predict[-1] == -1):
                #    print 1
                #else:
                #    print 0
                print y_correct_predict
                print y_wrong_predict
                #print data_generator.xseqs
            if action == 'testBatch':
                y_correct_predict = correct_prediction_svm.predict(prediction_outputs)   
                y_wrong_predict = wrong_prediction_svm.predict(prediction_outputs)
                num_seen_predictions_correct_svm = num_seen_predictions_correct_svm + sum(xx for xx in y_correct_predict if xx == 1)
                num_unseen_prediction_correct_svm = num_unseen_prediction_correct_svm + sum(xx for xx in y_correct_predict if xx == -1)
                num_seen_predictions_wrong_svm = num_seen_predictions_wrong_svm + sum(xx for xx in y_wrong_predict if xx == 1)
                num_unseen_prediction_wrong_svm = num_unseen_prediction_wrong_svm + sum(xx for xx in y_wrong_predict if xx == -1)
        if action == 'test' :
            print data_generator.yseqs
            print prediction[0]
        if action == 'testBatch':
            print "Num seen predictions (correct svm, wrong svm):" + repr(num_seen_predictions_correct_svm) + "," + repr(num_seen_predictions_wrong_svm)
            print "Num unseen predictions (corect svm, wrong svm):" + repr(num_unseen_prediction_correct_svm) + "," + repr(num_unseen_prediction_wrong_svm)
        #print x
            

    
    
    


def load_model(model_name, sess, seq_length = None):
    model_dir = os.path.dirname(model_name + '.meta')
    model_config_file = './output/' + model_dir+ "/params.yaml"
    import yaml
    with open(model_config_file, 'r') as stream:
        model_params = yaml.load(stream)
    model = model_params['model']
    hidden_units = model_params['hidden_units']
    num_layers = model_params['num_layers']
    ### TODO: Get this information from a separate config
    prob_config = config.get_problem_config(PROBLEM_NAME)
    if seq_length is None:
        seq_length = prob_config['max_sequence_length']
    observation_length = prob_config['input_length']
    action_length = prob_config['output_length']
    
    
    encoder = Encoder(action_length, observation_length)
    input_length = encoder.size_x()
    output_length = encoder.size_y()
    
    start = time.time()
    model = Seq2SeqModelExt(session=sess,
                hidden_units=hidden_units,
                model=model,
                num_layers=num_layers,
                seq_length=seq_length,
                input_length=input_length,
                output_length=output_length,
                batch_size=1,
                scope="model")
    end = time.time()
    model_create_time = end-start
    #model.load('vrep/version1/model.ckpt-967')
    model.load(model_name)
    start = time.time()
    model_load_time = start-end
    return model
    
    
    
def main():
    # running parameters
    #batch_size = 1
    #test_dataGenerator(batch_size)
    #train()
    
    action = None
    model_name = None
    model_input = None
    output_dir = None
    global PROBLEM_NAME
    
    opts, args = getopt.getopt(sys.argv[1:],"ha:m:i:o:p:",["action=","model=","input=", "outdir=", "problem="])
    #print opts
    for opt, arg in opts:
      # print opt
      if opt == '-h':
         print 'joint_training_model.py -a <train|test|testBatch> -m <rnn_model_name> -i <logfilename|seq|training data version> -o <svm output dir for train|svm model file prefix for test>'
         sys.exit()
      elif opt in ("-a","--action" ):
          action = arg
          if action not in ("train", "test", 'testBatch'):
              action = raw_input("Please specify correction action[train|test]:")
      elif opt in ("-m", "--model"):
         model_name = arg
      elif opt in ("-i", "--input"):
         model_input = arg
      elif opt in ("-o", "--outdir"):
          output_dir = arg
      elif opt in ("-p", "--problem"):
          PROBLEM_NAME = arg
        
    if action == 'train':
        train(model_name, output_dir, model_input)
    else:
        test(model_name, output_dir, model_input, action)
    #test()
    #test_dataGenerator(1,logfileName)   

if __name__ == '__main__':
    main()
