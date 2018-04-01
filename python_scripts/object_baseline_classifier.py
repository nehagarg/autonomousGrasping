import numpy as np
import get_initial_object_belief as giob
from gqcnn import Visualizer as vis
from combine_csvs import get_grasping_ros_mico_path
import perception as perception
import os
import time
from grasping_object_list import get_grasping_object_name_list

def reshape_keras_input(X_train):
    from keras import backend as K
    # input image dimensions
    img_rows, img_cols = 50, 40
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        #X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        #X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    return (X_train,input_shape)

def model_prediction(model,X_train):
    (X_train,input_shape) = reshape_keras_input(X_train)
    return model.predict(X_train)

def get_keras_cnn_model(X_train, Y_train, use_kmeans=False):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    from keras import backend as K
    np.random.seed(1)
    batch_size = 128
    nb_classes = 7
    if use_kmeans:
        nb_classes = 3
    nb_epoch = 30

    
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    (X_train,input_shape) = reshape_keras_input(X_train)

    

    #X_train = X_train.astype('bool')
    #X_test = X_test.astype('bool')

    print('X_train shape:', X_train.shape)  
    print(X_train.shape[0], 'train samples')
    #print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    #y_train_copy = [ord(x) - ord('a') for x in y_train]
    if use_kmeans:
        Y_train_categortical = np_utils.to_categorical(Y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    if use_kmeans:
        model.add(Activation('softmax'))

    if use_kmeans:
        model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    else:
        model.compile(loss='mean_squared_error',
              optimizer='adadelta',
              metrics=['accuracy'])
    start_time = time.time()
    if use_kmeans:
        model.fit(X_train, Y_train_categortical, validation_split=0.2, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
    else:
        model.fit(X_train, Y_train, validation_split=0.2, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1)
    end_time = time.time()
    
    print 'model train time : {:.5f}'.format(end_time -start_time)
    
    #logistic_train_predicted = np.argmax(model.predict(X_train), axis = 1)
    logistic_train_predicted = model.predict(X_train)
    #logistic_train_predicted = [chr(x+97) for x in logistic_train_predicted]
    #logistic_test_predicted = model.predict(X_test)
    #logistic_test_predicted = [chr(x+97) for x in logistic_test_predicted]
    #print logistic_train_predicted
    
    return (model, logistic_train_predicted)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])


def get_depth_image_thumbmail(depth_im, size_x = 16,size_y = 8, debug = False):
    depth_im_centered,_ = depth_im.center_nonzero()
    #if crop_point is None:
    #    crop_point = depth_im_centered.center
    #check
    depth_im_focus = depth_im_centered.focus(size_x,size_y)
    depth_im_focus_zero_pixels = depth_im_focus.zero_pixels()
    depth_im_centered_zero_pixels = depth_im_centered.zero_pixels()
    clipped = False
    try:
        assert np.array_equal(depth_im_focus_zero_pixels, depth_im_centered_zero_pixels)
    except:
        clipped = True
        #debug = True
        """
        print depth_im_focus_zero_pixels[0]
        print depth_im_centered_zero_pixels[0]
        a = np.append(depth_im_focus_zero_pixels,depth_im_centered_zero_pixels, axis=0)
        b,counts = np.unique(["{}-{}".format(i, j) for i, j in a], return_counts=True)
        #c = np.where(counts==1)
        print len(b)
        print len(counts)
        c = np.array([map(int,x.split('-')) for x in b[np.where(counts==1)]])
        print c
        print depth_im_focus.center
        print depth_im_centered[c[:,0],c[:,1]]
        """
    depth_im_crop = depth_im_centered.crop(size_x,size_y)
    depth_im_crop_thumbnail = depth_im_crop.resize(0.25)
    if debug:
        vis.figure()
        vis.subplot(1,4,1)
        vis.imshow(depth_im_crop)
        vis.subplot(1,4,2)
        vis.imshow(depth_im_centered)
        vis.subplot(1,4,3)
        vis.imshow(depth_im)
        vis.subplot(1,4,4)
        vis.imshow(depth_im_crop_thumbnail)
        vis.show()
    return depth_im_crop_thumbnail,clipped
    
def get_dir_list():
    dir_list = []
    for i in range(0,7):
        dir_list.append('belief_uniform_baseline_' + repr(i) + '_reward100_penalty10/simulator/fixed_distribution/')
    #dir_list.append('belief_uniform_cylinder_7_8_9_reward100_penalty10/use_discretized_data/simulator/fixed_distribution/')
    dir_list.append('belief_uniform_g3db_instances_train1_reward100_penalty10/use_discretized_data/use_weighted_belief/simulator/fixed_distribution/horizon90/')
    return dir_list

def get_baseline_labels(baseline_result_dir = 'data_low_friction_table_exp_ver6',use_kmeans=False , kmeans_label = ''):
    grasping_ros_mico_path = get_grasping_ros_mico_path()
    baseline_result_file_name = grasping_ros_mico_path + "/" + baseline_result_dir + "/baseline_results/"
    if(use_kmeans):
        baseline_result_file_name = baseline_result_file_name + "kmeans_object_labels_g3db_instances" + kmeans_label + ".csv"
    else:
        baseline_result_file_name = baseline_result_file_name + "a_success_cases_g3db_instances.csv"
    object_labels = {}
    with open(baseline_result_file_name) as f:
        if not use_kmeans:
            line = f.readline().rstrip('\n').split(",")
        for line in f:
            data = line.rstrip('\n').split(",")
            if use_kmeans:
                object_labels[data[0]] = int(data[1])
            else:
                object_labels[data[0]] = [float(x)/81.0 for x in data[1:8]]
            
    return object_labels
                
    
def get_data(use_kmeans=False,kmeans_label = '', for_test = False):   
    object_labels = get_baseline_labels(use_kmeans = use_kmeans, kmeans_label=kmeans_label)
    
    object_name = 'g3db_instances_non_test_version7'
    if for_test:
        object_name = 'g3db_instances_version7'
        #object_name = '39_beerbottle_final-13-Nov-2015-09-07-13_instance0'
    #object_name = '1_Coffeecup_final-10-Dec-2015-06-58-01_instance0'
    #object_name = 'Cylinder_7'
    #object_name = '39_beerbottle_final-13-Nov-2015-09-07-13_instance0'
    object_file_dir = '../grasping_ros_mico/point_clouds_for_classification'
    model_dir = object_file_dir + '/keras_model/'
    object_file_names = giob.get_object_filenames(object_name, object_file_dir)
    
    X = []
    Y = []
    object_names = []
    clipped_objects = []
    outfile_name = object_file_dir + "/clipped_object_list.txt"
    with open(outfile_name, 'r') as f:
        for line in f:
            clipped_objects.append(line.strip())
            
    for object_file_name_ in object_file_names:
        object_instance_name = object_file_name_.replace('.yaml','').split('/')[-1]
        
        for i in range(0,81):
            Y.append(object_labels[object_instance_name])
            object_names.append(object_instance_name+ "/" + repr(i))
            object_file_name = object_file_name_.replace('.yaml','') + "/" + repr(i)
            thumbnail_object_file_name = object_file_name + "_thumbnail.npy"
            if os.path.exists(thumbnail_object_file_name):
                print "Loading " + thumbnail_object_file_name
                depth_im_cropped = perception.DepthImage.open(thumbnail_object_file_name)
                if False:
                    vis.figure()
                    vis.subplot(1,1,1)
                    vis.imshow(depth_im_cropped)
                    vis.show()
            else:
                object_list = giob.load_object_file([object_file_name])
                (depth_im_cropped,clipped) = get_depth_image_thumbmail(object_list[0][0], 200,160,False)
                depth_im_cropped.save(thumbnail_object_file_name)
                if clipped:
                    clipped_objects.append(object_file_name)
            X.append(depth_im_cropped.data)
            
    #print clipped_objects
    outfile_name = object_file_dir + "/clipped_object_list.txt"
    with open(outfile_name, 'w') as f:
            f.write("\n".join(sorted(clipped_objects)))  
            f.write("\n")
    assert len(X) == len(Y)
    num_samples = len(X)
    print num_samples
    X = np.array(X)
    Y = np.array(Y)
    
    arr = np.arange(num_samples)
    np.random.shuffle(arr)
    #train_length = int(0.8*num_samples)
    #X_train = X[arr[0:train_length]]
    #Y_train = Y[arr[0:train_length]]
    
    #X_test = X[arr[train_length:num_samples]]
    #Y_test = Y[arr[train_length:num_samples]]
    X_shuf = X[arr[0:num_samples]]
    Y_shuf = Y[arr[0:num_samples]]
    #object_names_shuf = object_names[arr[0:num_samples]]
    return (X_shuf,Y_shuf, object_names ,arr,model_dir)
    
def train(use_kmeans = False, kmeans_label = ''):
        
    label_tag = ''
    if kmeans_label !='':
        label_tag = "label" + kmeans_label + "_"
    (X_shuf,Y_shuf, object_names, arr,model_dir) = get_data(use_kmeans,kmeans_label)
    (model,train_predicted) = get_keras_cnn_model(X_shuf,Y_shuf,use_kmeans)
    print train_predicted[0]
    print Y_shuf[0]
    print object_names[arr[0]]
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_file_name = model_dir + timestr + '.h5'
    if use_kmeans:
        model_file_name = model_dir + "kmeans_" +label_tag + timestr + '.h5'
    model.save(model_file_name)
    return (X_shuf,Y_shuf, model_dir,timestr)
    
def test(model_name, use_kmeans = False, kmeans_label = '' ):
    from keras.models import load_model
    (X_shuf,Y_shuf, object_names,arr,model_dir) = get_data(use_kmeans,kmeans_label, for_test = True)
    #if model_name is None:
    model_name = model_dir + model_name + '.h5'
    model = load_model(model_name)
    ans = model_prediction(model,X_shuf)
    model_prediction_filename = model_name.replace('.h5','.pred')
    with open(model_prediction_filename,'w') as f:
        for i in range(0,len(ans)):
            f.write(object_names[arr[i]] + " " + repr(ans[i]) + " " + 
            repr(np.argmax(ans[i])) + " " + repr(Y_shuf[i]) + "\n")
        
    #model = load_model(model_dir + model_name + '.h5')
    #y_predicted = model.predict(X_test)
    
def get_object_represention_and_weighted_belief(depth_im, 
object_group_name,keras_model_dir,keras_model_name, baseline_results_dir):
    from keras.models import load_model
    X = []
    (depth_im_cropped,clipped) = get_depth_image_thumbmail(depth_im, 200,160,False)
    X.append(depth_im_cropped.data)
    model_name = keras_model_dir + keras_model_name + '.h5'
    model = load_model(model_name)
    ans = model_prediction(model,np.array(X))[0]
    use_kmeans = False
    kmeans_label = ''
    if 'kmeans' in keras_model_name:
        use_kmeans = True
        if 'label' in keras_model_name:
            kmeans_label = "_" + keras_model_name.split('_')[2]
    object_labels = get_baseline_labels(baseline_results_dir,use_kmeans, kmeans_label)
    object_list = giob.get_object_filenames(object_group_name, "")
    object_list = [x.replace('.yaml',"").replace("/","") for x in object_list]
    if use_kmeans:
        probs = [ans[object_labels[o]] for o in object_list]
    else:
        
        object_list_pred = [np.square(np.subtract(ans, np.array(object_labels[o]))).mean() for o in object_list]
        print object_list_pred
        mean_error_list = [np.square(np.subtract(ans, np.array(object_labels[o]))).mean() for o in object_list]
        probs_unnormalized = [np.exp(1.0/((10*x) + 0.1)) for x in mean_error_list]
        z = sum(probs_unnormalized)
        probs = [x/z for x in probs_unnormalized]
    return list(ans),np.array(probs)

        
    

"""
def get_belief_for_objects(object_group_name, object_file_dir, clip_objects = -1, keras_model_name = None, baseline_results_dir = None, debug = False, start_node=True):
    if keras_model_name is None:
        obj_filenames = get_object_filenames(object_group_name, object_file_dir)
        giob = GetInitialObjectBelief(obj_filenames, debug, start_node)
        ans = giob.get_object_probabilities()
        print "<Object Probabilities>" + repr(ans)
        return ans
    else:
        #Load keras model
        giob = GetInitialObjectBelief(None, debug, start_node)
        (depth_im,cam_intr) = giob.get_object_point_cloud_from_sensor()
        import object_baseline_classifier as obc
        keras_model_dir = object_file_dir + "/" +keras_model 
        ans,object_beliefs = obc.get_object_represention_and_weighted_belief(depth_im,
        object_group_name,keras_model_dir,keras_model_name, baseline_results_dir)
        print "<Object Probabilities>" + repr(ans)
        print "<Object Beliefs>" + repr(object_beliefs)
        if clip_objects > 0:
            sort_ind = np.argsort(object_beliefs)
            object_beliefs[sort_ind[0:-1*clip_objects]] = 0
            object_beliefs_sum = np.sum(object_beliefs)
            object_beliefs = [x/object_beliefs_sum for x in object_beliefs]
        return object_beliefs
"""

def cluster_labels(num_clusters=3):
    from sklearn.cluster import KMeans
    object_labels = get_baseline_labels()
    object_key_to_array_index = {}
    X = []
    object_group_name = 'g3db_instances_non_test_version7'
    for object_name in get_grasping_object_name_list(object_group_name):
        sum_value = sum(object_labels[object_name])
        object_labels[object_name] = [x/sum_value for x in object_labels[object_name]]
        object_key_to_array_index[object_name] = len(X)
        X.append(object_labels[object_name])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)  
    transormed_X = kmeans.transform(X)
    
    
    for object_name in get_grasping_object_name_list(object_group_name):
        print repr(kmeans.labels_[object_key_to_array_index[object_name]])+':' + object_name + ':'  + repr(transormed_X[object_key_to_array_index[object_name]])
    
    baseline_result_dir = 'data_low_friction_table_exp_ver6'
    grasping_ros_mico_path = get_grasping_ros_mico_path()
    label_file_name = grasping_ros_mico_path + "/" + baseline_result_dir + "/baseline_results/kmeans_object_labels_g3db_instances_trial.csv"

    with open(label_file_name,'w') as f:
        for object_name in get_grasping_object_name_list(object_group_name):
            f.write(object_name+"," +repr(kmeans.labels_[object_key_to_array_index[object_name]]) + '\n' )
    return kmeans
    
def main():
    test('kmeans_label_1_20180223-105821', use_kmeans = True, kmeans_label = '_1' )
    #train(use_kmeans = True, kmeans_label = '_1')
    #cluster_labels(3)
    
if __name__ == '__main__':
    main()    