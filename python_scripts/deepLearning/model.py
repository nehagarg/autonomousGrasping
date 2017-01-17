import sys
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

import numpy as np
STUMP = '$'
PAD = '*'
class Encoder(object):
    def __init__(self, num_actions=14, log_num_observations=13):
        self.num_actions = num_actions
        self.len_obs = log_num_observations
        self.total_len = self.num_actions + self.len_obs + 2

    def size_x(self):
        return self.total_len

    def size_y(self):
        return self.num_actions + 2

    def transform_x(self, x):
        #print x
        #for toy problem
        #trans_x = np.zeros(self.size_x(), dtype=np.int8)
        #for vrep problem
        trans_x = np.zeros(self.size_x(), dtype=np.float32)
        
        if x == STUMP:
            trans_x[-2] = 1
        elif x == PAD:
            trans_x[-1] = 1
        else:
            act = x[0]
            obs = x[1]
            trans_x[0:self.len_obs] = obs
            trans_x[self.len_obs + int(act)] = 1
        #print trans_x
        return trans_x

    def transform_y(self, y):
        trans_y = np.zeros(self.size_y(), dtype=np.bool)
        if y == STUMP:
            trans_y[self.num_actions] = 1
        elif y == PAD:
            trans_y[self.num_actions+1] = 1
        else:
            trans_y[int(y)] = 1
        return trans_y

class Seq2SeqGraph(object):
    def __init__(self,
                is_training=False,
                hidden_units=128,
                model='lstm',
                num_layers=1,
                seq_length=10,
                input_length=10,
                output_length=10,
                keep_prob=0.6, #dropout keep probability
                learning_rate=0.002,
                weight_amplitude=0.08,
                batch_size=32):
        self.inputs = []
        self.outputs = []
        self.batch_size = batch_size
        self.output_length = output_length
        tf.constant(seq_length, name='seq_length')

        # sequence data
        for i in xrange(seq_length):
            #for toy problem
            #self.inputs.append(tf.placeholder(tf.int8, shape=(None, input_length), name="input_{0}".format(i)))
            #for vrep problem
            self.inputs.append(tf.placeholder(tf.float32, shape=(None, input_length), name="input_{0}".format(i)))
            self.outputs.append(tf.placeholder(tf.bool, shape=(None, output_length), name="output_{0}".format(i)))
        ### for c++ calling
        # valid action mask for action filtering in sampling
        self.valid_actions = tf.placeholder(tf.bool, shape=(None, self.output_length), name='valid')
        ### end for c++ calling

        def random_uniform():
            return tf.random_uniform_initializer(-weight_amplitude, weight_amplitude)

        if model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        if model == 'lstm':
            cell = cell_fn(hidden_units, state_is_tuple=True)
            self.cells = rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        else:
            cell = cell_fn(hidden_units)
            self.cells = rnn_cell.MultiRNNCell([cell] * num_layers)

        self.softmax_w = tf.get_variable('softmax_w', shape=(hidden_units, output_length), initializer=random_uniform())
        self.softmax_b = tf.get_variable('softmax_b', shape=(output_length,), initializer=random_uniform())

        state = self.cells.zero_state(batch_size=batch_size, dtype=tf.float32)
        inputs = [tf.cast(inp, tf.float32) for inp in self.inputs]
        _outputs, _ = rnn.rnn(self.cells, inputs, state)
        self.output_logits = [tf.matmul(_output, self.softmax_w) + self.softmax_b for _output in _outputs]
        self.probs = [tf.nn.softmax(logit, name='prob_{}'.format(i)) for i, logit in enumerate(self.output_logits)]
        #self.tops = [tf.argmax(prob, 1, name='top_{}'.format(i)) for i, prob in enumerate(self.probs)]
        #self.samples = [self.batch_sample_with_temperature(prob, i) for i, prob in enumerate(self.probs)]
        #compact variable for cpp
        self.tensor_probs = tf.pack(self.probs, 0, 'probs')
        #self.tensor_tops = tf.pack(self.tops, 0, 'tops')
        #self.tensor_samples = tf.pack(self.samples, 0, 'samples')

        if is_training:
            self.cells = tf.nn.rnn_cell.DropoutWrapper(self.cells, output_keep_prob=keep_prob)
            self.targets = [tf.cast(oup, tf.float32) for oup in self.outputs]#[1:]
            #tf.Print(self.targets, [self.targets], message="testing tf print")
            #print output_logits
            self.losses = [tf.nn.softmax_cross_entropy_with_logits(logit, target) for logit, target in zip(self.output_logits, self.targets)]
            #self.losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, target) for logit, target in zip(self.output_logits, self.targets)]
            
            self.loss = tf.reduce_sum(tf.add_n(self.losses))
            self.cost = self.loss/seq_length/batch_size
            self.lr = tf.Variable(learning_rate, trainable=False)
            #train_vars = tf.trainable_variables()
            #grads = tf.gradients(self.cost, train_vars)
            #optimizer = tf.train.AdamOptimizer(self.lr)
            #self.train_op = optimizer.apply_gradients(zip(grads, train_vars))
            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.minimize(self.cost)


    def cpp_saver(self, cpp_dir):
        def save(sess, global_step=0):
            for variable in tf.trainable_variables():
                tensor = tf.constant(variable.eval())
                tf.assign(variable, tensor, name="nWeights")

            # This does not work in tensorflow with python3 now,
            # but we defenetely need to save graph as binary!
            tf.train.write_graph(sess.graph_def, cpp_dir, 'graph_{}.pb'.format(global_step), as_text=False)
        return save

class Seq2SeqModel(object):
    def __init__(self,
                session,
                hidden_units=128,
                model='lstm',
                num_layers=1,
                seq_length=10,
                input_length=10,
                output_length=10,
                keep_prob=0.6, #dropout keep probability
                learning_rate=0.001,
                weight_amplitude=0.08,
                batch_size=32,
                scope="seq2seq_model"):
        self.session = session
        self.batch_size = batch_size

        with tf.variable_scope(scope, reuse=None):
            self.training_graph = Seq2SeqGraph(
                                            is_training=True,
                                            hidden_units=hidden_units,
                                            model=model,
                                            num_layers=num_layers,
                                            seq_length=seq_length,
                                            input_length=input_length,
                                            output_length=output_length,
                                            keep_prob=keep_prob, #dropout keep probability
                                            learning_rate=learning_rate,
                                            batch_size=batch_size)
        with tf.variable_scope(scope, reuse=True):
            self.testing_graph = Seq2SeqGraph(
                                            is_training=False,
                                            hidden_units=hidden_units,
                                            model=model,
                                            num_layers=num_layers,
                                            seq_length=seq_length,
                                            input_length=input_length,
                                            output_length=output_length,
                                            keep_prob=keep_prob, #dropout keep probability
                                            learning_rate=learning_rate,
                                            batch_size=batch_size)

    def init_variables(self):
        tf.initialize_all_variables().run()

    def _fit_batch(self, input_values, targets):
        assert targets.shape[0] == input_values.shape[0] == self.batch_size
        assert len(self.training_graph.inputs) == input_values.shape[1]
        assert len(self.training_graph.outputs) == targets.shape[1]
        input_feed = {}
        for i, inp in enumerate(self.training_graph.inputs):
            input_feed[inp.name] = input_values[:, i, :]
        # delegate stump to the data generator
        for i, oup in enumerate(self.training_graph.outputs):
            input_feed[oup.name] = targets[:, i, :]
        #print input_feed
        input_feed[self.training_graph.valid_actions.name] = np.ones((self.training_graph.batch_size, self.training_graph.output_length))
        #print self.training_graph.valid_actions.name
        #print input_feed[self.training_graph.valid_actions.name]

        train_loss, _ = self.session.run([self.training_graph.cost, self.training_graph.train_op], feed_dict=input_feed)
        #print self.training_graph.targets
        return train_loss

    def set_learning_rate(self, learning_rate):
        self.session.run(tf.assign(self.training_graph.lr, learning_rate))

    def get_learning_rate(self):
        return self.training_graph.lr.eval()

    def fit(self,
            data_generator,
            num_epochs = 30,
            batches_per_epoch = 256,
            lr_decay = 0.95,
            num_val_batches=120,
            output_dir='output',
            cpp_dir='output_cpp'):
        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
            cpp_saver = self.training_graph.cpp_saver(cpp_dir)

        history = []
        prev_error_rate = np.inf
        val_error_rate = np.inf
        best_val_error_rate = np.inf
        val_set = [data_generator.next_batch(validation=True) for _ in xrange(num_val_batches)]

        epochs_since_init = 0

        for e in xrange(num_epochs):
            start = time.time()
            for b in xrange(batches_per_epoch):
                inputs, targets = data_generator.next_batch(validation=False)
                train_loss = self._fit_batch(inputs, targets)
            end = time.time()

            val_error_rate = self.validate(val_set)

            '''
            if epochs_since_init == 1000 and val_error_rate > 0.85:
                self.init_variables()
                epochs_since_init = 0
                print 'Restarting'
                continue
            '''
            epochs_since_init += 1

            print("Epoch {}: train_loss = {:.3f}, val_error_rate = {:.3f}, time/epoch = {:.3f}"
                    .format(e, train_loss, val_error_rate, end - start))

            if best_val_error_rate > val_error_rate or (val_error_rate < 0.01 or e % 400 == 0):
                save_path = saver.save(self.session, "{}/model.ckpt".format(output_dir), global_step=e)
                cpp_saver(self.session, global_step=e)
                print "model saved : " + repr(best_val_error_rate) + "," + repr(val_error_rate)
                best_val_error_rate = val_error_rate

            if val_error_rate > prev_error_rate and self.get_learning_rate() > 0.0001:
                self.set_learning_rate(self.get_learning_rate() * lr_decay)
                print "Decreasing Learning Rate to {:.5f}".format(self.get_learning_rate())
            elif val_error_rate < 0.10:
                val_set = [data_generator.next_batch() for _ in xrange(num_val_batches)]

            history.append({
                    'epoch' : e,
                    'val_error_rate':float(val_error_rate),
                    'train_loss': float(train_loss),
                    'learning_rate': float(self.get_learning_rate())
                })
            with open('{}/history.json'.format(output_dir), 'w') as outfile:
                json.dump(history, outfile)

            prev_error_rate = val_error_rate

    def sample(self, input_values):
        input_feed = {}
        for i, inp in enumerate(self.testing_graph.inputs):
            input_feed[inp.name] = input_values[:, i, :]
        mask = np.zeros((self.batch_size, self.testing_graph.output_length))
        mask[0][1] = 1
        mask[0][4] = 1
        mask[1][2] = 1
        mask[1][3] = 1
        input_feed[self.testing_graph.valid_actions.name] = mask
        #probs, sample= self.session.run([self.testing_graph.probs, self.testing_graph.sample], input_feed)
        probs = self.session.run(self.testing_graph.tensor_probs, input_feed)
        sample, EXP, X, U, diff = self.session.run([self.testing_graph.tensor_samples, self.testing_graph.exponent_raised, self.testing_graph.matrix_X, self.testing_graph.matrix_U, self.testing_graph.diff], input_feed)
        print 'probs[0]', probs.shape
        print 'sample', sample
        print 'EXP, X, U, diff', EXP, X, U, diff

    def predict(self, input_values):
        input_feed = {}
        for i, inp in enumerate(self.testing_graph.inputs):
            input_feed[inp.name] = input_values[:, i, :]
        probs = self.session.run(self.testing_graph.probs, input_feed)
        probs = np.array(probs)
        probs = np.transpose(probs, (1,0,2))
        return probs

    def validate_old(self, val_set):
        num_correct = 0
        num_samples = 0
        for batch in val_set:
            x, y = batch
            target = np.argmax(y, axis=2)
            prediction = np.argmax(self.predict(x), axis=2)
            
            num_correct += sum([int(np.all(t==p)) for t, p in zip(target, prediction)])
            num_samples += len(x)
            #if num_correct < num_samples :
            print "NC:" + repr(num_correct) + " NS:" + repr(num_samples) + " Target: " + repr(target) + " Prediction: " + repr(prediction)
        return 1.0 - float(num_correct)/num_samples
    
    def validate(self, val_set):
        num_correct = 0
        num_samples = 0
        for batch in val_set:
            x, y = batch
            target = np.argmax(y, axis=2)
            prediction = np.argmax(self.predict(x), axis=2)
            
            num_correct += sum([int(sum(t==p)) for t, p in zip(target, prediction)])
            num_samples += sum([len(x1) for x1 in x])
            #if num_correct < num_samples :
            print "NC:" + repr(num_correct) + " NS:" + repr(num_samples) + " Target: " + repr(target) + " Prediction: " + repr(prediction)
        return 1.0 - float(num_correct)/num_samples
    
    def debug(self, data_generator):
        input_values, targets = data_generator.current_batch()
        input_feed = {}
        for i, inp in enumerate(self.training_graph.inputs):
            input_feed[inp.name] = input_values[:, i, :]
        # delegate stump to the data generator
        for i, oup in enumerate(self.training_graph.outputs):
            input_feed[oup.name] = targets[:, i, :]
        #print input_feed
        input_feed[self.training_graph.valid_actions.name] = np.ones((self.training_graph.batch_size, self.training_graph.output_length))
        #print self.training_graph.valid_actions.name
        #print input_feed[self.training_graph.valid_actions.name]
        output_logits = self.session.run(self.training_graph.output_logits, feed_dict=input_feed)
        print np.array(output_logits)
        output_targets = self.session.run(self.training_graph.targets, feed_dict=input_feed)
        print np.array(output_targets)
        losses = self.session.run(self.training_graph.losses, feed_dict=input_feed)
        print np.array(losses)
        
    def load(self, checkpoint_file, output_dir='output'):
        saver = tf.train.Saver()
        saver.restore(self.session, os.path.join(output_dir, checkpoint_file))

def parse_data(fileName):
    if(fileName is None) or (fileName.endswith('log')):
        seqs = traces.parse(fileName)
    else:
        seqs = [[]]
        for act_obs_string in fileName.split('*'):
            values = act_obs_string.split(",")
            act = int(values[0])
            obs = [float(x) for x in values[1:]]
            seqs[0].append((act,obs))
    #print seqs
    #seqs = traces.parse('canadian_bridge_trace', 'canadian_bridge')
    st = [STUMP]
    xseqs = [(st + seq)[:-1] for seq in seqs]
    yseqs = [[t[0] for t in seq] for seq in seqs]
    '''
    xseqs = seqs
    yseqs = [[t[0] for t in seq][1:]+st for seq in seqs]
    '''
    maxlen = max(map(len, seqs)) + 1
    # extend to maxlen
    xseqs = [seq + [PAD]*(maxlen-len(seq)) for seq in xseqs]
    yseqs = [seq + [PAD]*(maxlen-len(seq)) for seq in yseqs]

    #encoder = Encoder(*get_problem_info(problem, size, number))
    #print xseqs[0]
    #print yseqs[0]
    #for toy problem
    #encoder = Encoder(10, 26)
    #for old vrep problem
    #encoder = Encoder(19, 8)
    #for vrep problem
    encoder = Encoder(11, 8)
    return (seqs, xseqs, yseqs, encoder, maxlen)

class DataGenerator(object):
    def __init__(self, batch_size, fileName=None):
        self.batch_size = batch_size
        self.seqs, self.xseqs, self.yseqs, self.encoder, self.seq_length = parse_data(fileName)
        #print len(self.xseqs)
        #print self.seq_length
        self.num_batches = len(self.xseqs)/self.batch_size
        #print 'data number of batches', self.num_batches
        self.batch_id = -1

    def _next_batch_id(self):
        self.batch_id += 1
        if self.batch_id >= self.num_batches:
            self.batch_id = 0
        return self.batch_id

    def parse_xs(self, xs):
        return np.array(map(self.encoder.transform_x, xs))

    def parse_ys(self, ys):
        return np.array(map(self.encoder.transform_y, ys))

    def next_batch(self, validation=False):
        idx = self._next_batch_id()*self.batch_size
        return self.get_batch_from_id(idx,validation)
    
    def current_batch(self):
        idx = self.batch_id
        return self.get_batch_from_id(idx)
        
    def get_batch_from_id(self, idx, validation=False):
        xs = self.xseqs[idx: idx+self.batch_size]
        ys = self.yseqs[idx: idx+self.batch_size]
        #print xs
        #print ys
        trans_xs = np.array([self.parse_xs(seq) for seq in xs])
        trans_ys = np.array([self.parse_ys(seq) for seq in ys])
        #print trans_xs[0][0:2]
        #print trans_ys[0][0:2]
        return (trans_xs, trans_ys)

def test_dataGenerator(batch_size, fileName = None):
    data = DataGenerator(batch_size, fileName)
    inp, tgt = data.next_batch()
    #print len(inp)
    #print inp.shape
    #print tgt.shape
    print data.xseqs
    print data.yseqs

def train():
    # training parameters
    hidden_units = 128
    num_layers = 2
    num_epochs = 1000
    training_batch_size = 20
    num_val_batches = 1
    keep_prob = 1.0
    learning_rate = 0.1
    #model='lstm'
    model='rnn'
    # data_genrator
    data_generator = DataGenerator(training_batch_size)
    print len(data_generator.xseqs)
    print data_generator.seq_length
    print 'data number of batches', data_generator.num_batches
    batches_per_epoch = data_generator.num_batches
    seq_length = data_generator.seq_length
    input_length = data_generator.encoder.size_x()
    output_length = data_generator.encoder.size_y()

    with tf.Session(config=config.get_tf_config()) as sess:
        print "building model"
        model = Seq2SeqModel(session=sess,
                hidden_units=hidden_units,
                model=model,
                num_layers=num_layers,
                seq_length=seq_length,
                input_length=input_length,
                output_length=output_length,
                keep_prob=keep_prob,
                learning_rate=learning_rate,
                batch_size=training_batch_size,
                scope="model")
        #summary_writer = tf.train.SummaryWriter('output', sess.graph)
        model.init_variables()

        print "finished building model"
        model.fit(data_generator,
                num_epochs=num_epochs,
                batches_per_epoch=batches_per_epoch,
                num_val_batches=num_val_batches)
        #model.debug(data_generator)
        print "finished training"
      
def get_prob_prediction(action_probs):
    r = random.random()
    prob = 0
    #sort_indices = np.argsort(action_probs[-2])
    
    #Remove actions with very less probability
    adjusted_probs = [x for x in action_probs[-2]]
    for i in range(0,len(action_probs[-2])):
        if adjusted_probs[i] < 0.1:
            adjusted_probs[i] = 0
        prob = prob + adjusted_probs[i]
    adjusted_probs = [x/prob for x in adjusted_probs]
    prob = 0
    for i in range(0,len(action_probs[-2])):
        if ((prob+adjusted_probs[i]) > r):
            return i
        prob = prob + adjusted_probs[i]
    return -1
def test(fileName=None):
    # training parameters
    hidden_units = 128
    num_layers = 2
    training_batch_size = 1 #100
    num_val_batches = 1
    model='rnn'
    # data_genrator
    data_generator = DataGenerator(training_batch_size,fileName)
    seq_length = data_generator.seq_length
    input_length = data_generator.encoder.size_x()
    output_length = data_generator.encoder.size_y()
    with tf.Session(config=config.get_tf_config()) as sess:
        #print "building model"
        model = Seq2SeqModel(session=sess,
                hidden_units=hidden_units,
                model=model,
                num_layers=num_layers,
                seq_length=seq_length,
                input_length=input_length,
                output_length=output_length,
                batch_size=training_batch_size,
                scope="model")
        model.load('vrep/version1/model.ckpt-967')

        #print "finished building and loading model"
        if fileName is None:
            num_val_batches = data_generator.num_batches
            for _ in xrange(num_val_batches):
                val_set = [data_generator.next_batch(validation=True) ]
                start = time.time()
                validate_loss = model.validate(val_set)
                end = time.time()
            print 'validate loss: ', validate_loss, ' time/sequence: {:.5f}'.format((end-start)/num_val_batches/training_batch_size)
        
        else:
            x,y = data_generator.next_batch(validation=True)
            probs = model.predict(x)
            #print probs.shape
            probs_without_dummy_actions = [i[:-2] for i in probs[0] ]
            prediction = np.argmax([probs_without_dummy_actions], axis=2)
            #prob_prediction = get_prob_prediction(probs_without_dummy_actions)
            
            #print prediction[0]
            print prediction[0][-2]
            #if prob_prediction == -1:
            #    print prediction[0][-2]
            #else:
            #    print prob_prediction
            print ' '.join(str(p) for p in probs_without_dummy_actions[-2])
            #print prob_prediction
            print prediction[0][-2]
            print prediction[0]
            
            #print data_generator.seqs
            #print data_generator.xseqs
            print data_generator.yseqs
	    print 'vrep/version/model.ckpt-967'
            #print np.argmax(y, axis=2)[0]
        '''
        X = ['$', (6, 1), (0, 0), (0, 0), (4, 0), (7, 2), (5, 2), (1, 0), (1, 0), (10, 1), (0, 0), (1, 0), (4, 0), (12, 2), (2, 0), (2, 0), (2, 0), (8, 1), (2, 0), (3, 0), (4, 0), (1, 0), (2, 0), (9, 1), (4, 0), (1, 0), (11, 1), (4, 0), (13, 2), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (14, 1), (4, 0), (1, 0), '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']
        y = [6, 0, 0, 4, 7, 5, 1, 1, 10, 0, 1, 4, 12, 2, 2, 2, 8, 2, 3, 4, 1, 2, 9, 4, 1, 11, 4, 13, 1, 1, 1, 1, 1, 14, 4, 1, 1, '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']
        for _ in xrange(len(X)):
            i = 20
            tmpX = X[:i] + [PAD]*(len(X)-i)
            print tmpX
            xs = [tmpX, tmpX]
            ys = [y, y]
            trans_xs = np.array([data_generator.parse_xs(tmp) for tmp in xs])
            #print trans_xs
            
            prediction = np.argmax(model.predict(trans_xs), axis=2)
            print 'y', y[:i]
            print 'pred', prediction[0][:i]
            print 'pred i', prediction[0][i]
            print 'i' , i
        '''
            #model.sample(trans_xs)


def main():
    # running parameters
    #batch_size = 1
    #test_dataGenerator(batch_size)
    #train()
    action = 'train'
    i = 0
    if len(sys.argv) > 1:
        action = 'test'
        try:
            i = int(sys.argv[1])
        except ValueError:
            action = 'testWithSeq'
    
    if action == 'train':
        train()
    else:
        if action == 'test' :
            #if len(sys.argv) > 1:
            #     i = int(sys.argv[1])
            #logfileName = '../../grasping_ros_mico/despot_logs/learning_trial_'+ repr(i) + '.log'
            logfileName = '../../grasping_ros_mico/results/learning/version1/TableScene_9cm_trial_' + repr(i) + '.log'
            #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/graspingV3_state_' + repr(i) + '_t20_obs_prob_change_particles_as_state_4objects.log'
            #test_dataGenerator(1,logfileName)
            #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version7/dagger_data/graspingV4_state_' + repr(i) + '_t10_n10_multi_runs_obs_prob_change_particles_as_state_4objects.log'
            #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/deepLearning_same_objects/version9/state_' + repr(i) + '_multi_runs.log'
            #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2/test.log'
            #test_dataGenerator(1,logfileName)
            #logfileName = '/home/neha/WORK_FOLDER/phd2013/phdTopic/despot/despot-0.2-server-version/4_objects_obs_prob_change_particles_as_state/adaboost_same_objects/sensor_observation_sum/state_' + repr(i) + '.log'
    
        else:
            logfileName = sys.argv[1]

        test(logfileName)
    #test()
    #test_dataGenerator(1,logfileName)
    

if __name__ == '__main__':
    main()
