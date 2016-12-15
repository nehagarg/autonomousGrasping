import os
import sys
import time

# set paths for storing models and results locally
#DATA_PATH='/media/data/mcdlp/trace'
#MODEL_PATH='/media/data/mcdlp/model'
#CPP_MODEL_PATH='/media/data/mcdlp/cpp'
#BASE='/media/psl-ctg/DATA2/mcdlp/'

def global_var_setting():
    BASE='/media/data/mcdlp/'
    DATA_PATH=BASE+'trace'
    MODEL_PATH=BASE+'model'
    CPP_MODEL_PATH=BASE+'cpp'

    ARCHS = [('L',1),
             ('LL', 2),
             ('LLL',3),
            ('LLLL', 4),
            ('BLL', 2),
            ('BLLT', 2),
            ('BGG', 2)]
    PROBLEMS = [('rocksample', 'appl', 7, 8),
            ('rocksample', 'appl', 11, 11),
            ('rocksample', 'despot_rocksample', 15, 15),
            ('pocman', 'despot', 11, 11)]
    if 'IDX' not in os.environ or 'PDX' not in os.environ or 'SDX' not in os.environ:
        print 'Please set IDX to 0, 1, 2 or 3 in the enviroment.'
        print 'Please set PDX to 0, 1, 2 or 3 in the enviroment.'
        print 'Please set SDX to 0, 1 for sampling (False, True) in the enviroment.'
        print 'IDX for ', ARCHS
        print 'PDX for', PROBLEMS
        sys.exit(-1)

    # architecture
    IDX = int(os.environ['IDX'])
    PDX = int(os.environ['PDX'])
    SDX = bool(int(os.environ['SDX']))

    PROBLEM, TRACE_TYPE, SIZE, NUMBER=PROBLEMS[PDX]
    print 'PROBLEM', PROBLEMS[PDX]
    print 'ARCH', ARCHS[IDX]

    # DATA Selection
    SUFFIX='test'
    assert SUFFIX in {'test', 'real'}
    # parameters for training
    BATCH_SIZE=512
    EPOCHES=1000
    SAMPLING=SDX # whether use sampling method for data selection
    HIDDEN=128
    LEARNING_RATE=0.1
    DROPOUT=0.2 # probability of dropping weights update

def get_architecture():
    return ARCHS[IDX]

def get_problem_name():
    return '%s_%d_%d_%s' % (PROBLEM, SIZE, NUMBER, SUFFIX)

def get_trace_path():
    path = os.path.join(DATA_PATH, get_problem_name())
    make_dir(path)
    return path

def get_model_path():
    path = os.path.join(MODEL_PATH, get_problem_name())
    make_dir(path)
    return path

def get_cpp_path():
    path = os.path.join(CPP_MODEL_PATH, get_problem_name())
    make_dir(path)
    return path

def get_tf_config(device=0, fraction=0.23):
    import tensorflow as tf
    # set visible GPU for running multiple programmes
    os.environ["CUDA_VISIBLE_DEVICES"]=repr(device)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Assume that you have 12GB of GPU memory and want to allocate ~2.4GB:
    config.gpu_options.per_process_gpu_memory_fraction=fraction
    return config

def make_dir(d):
    try:
        os.stat(d)
    except:
        os.mkdir(d)
    return d

if __name__ == "__main__":
    print get_trace_path()
