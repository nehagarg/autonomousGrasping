
import os
import sys
#from sklearn.externals import joblib
#from sklearn.utils import check_array
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import time
import yaml
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mse, binary_crossentropy
sys.path.append('../../python_scripts')

import prepare_data_for_transition_observation_training as dataProcessor
#from grasping_dynamic_model import get_float_array
import math
import argparse
import perception as perception
import matplotlib.pyplot as plt
from gqcnn import Visualizer as vis

PICK_ACTION_ID = 10
OPEN_ACTION_ID = 9
CLOSE_ACTION_ID = 8
NUM_PREDICTIONS = 18



def get_one_hot(x,num_classes):
    z = tf.cast(x,tf.uint8)
    return K.one_hot(z,num_classes)
def map_actions(x,num_classes):
    y = K.switch(K.greater(x,8), x-4, (x*0.5))

    return get_one_hot(y,num_classes)

def get_relative_pos(args):
    g = args[0]
    o = args[1]
    return o-g

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    #print repr(batch)
    #print repr(dim)

    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))

    #print epsilon
    return z_mean + K.exp(0.5 * z_log_var) * epsilon




def prepare_state_input_layer(latent_dim, input_s,add_action = True):
    num_object_classes = 3 # For only 3 objects
    num_action_classes = 7 # For 11 - 4 actions

    #latent_dim = 2





    #State inputs
    #gripper_2D_pos = input_s[0:2]
    #gripper_2D_pos = layers.Lambda( lambda x: tf.slice(x, [0,0], [-1,2]))(input_s)
    #gripper_2D_pos = tf.keras.Input(shape=(2,), name='gripper_2D_pos_input')
    #object_2D_pos = input_s[2:4]
    #object_2D_pos = layers.Lambda( lambda x: tf.slice(x, (2), (2)))(input_s)
    #object_2D_pos = tf.keras.Input(shape=(2,), name='object_2D_pos_input')

    gripper_2D_pos, object_2D_pos, object_thetaz, gripper_finger_joint_angles,object_id, action_id= layers.Lambda(lambda x: tf.split(x,[2,2,1,2,1,1],1), name='split_input')(input_s)
    #object_thetaz = input_s[4:5]
    #object_thetaz = layers.Lambda( lambda x: tf.slice(x, (4), (1)))(input_s)
    #object_thetaz = tf.keras.Input(shape=(1,), name='object_thetaz_input')

    #previous_action_id = tf.keras.Input(shape = (num_action_classes,), name='previous_action_id_input')
    #gripper_finger_joint_angles = input_s[5:7]
    #gripper_finger_joint_angles = layers.Lambda( lambda x: K.slice(x, (5), (2)))(input_s)
    #gripper_finger_joint_angles = tf.keras.Input(shape = (2,), name='gripper_finger_joint_angle_input')

    #action_id = input_s[7:8]
    #action_id = layers.Lambda( lambda x: tf.slice(x, (7), (1)))(input_s)
    #action_id = tf.keras.Input(shape = (1,), name='action_id_input')
    if add_action:
        mapped_action_without_flatten = layers.Lambda(map_actions, name = 'map_action', arguments={'num_classes': num_action_classes})(action_id)
        mapped_action = layers.Flatten()(mapped_action_without_flatten)

    #object_id = input_s[8:9]
    #object_id = layers.Lambda( lambda x: tf.slice(x, (8), (1)))(input_s)
    #object_id = tf.keras.Input(shape = (1,), name='object_id_input')
    one_hot_object_id_without_flatten = layers.Lambda(get_one_hot, name = 'map_object_id', arguments={'num_classes': num_object_classes})(object_id)
    one_hot_object_id = layers.Flatten()(one_hot_object_id_without_flatten)

    relative_2D_pose = layers.Lambda(get_relative_pos, name = 'rel_object_pos')([gripper_2D_pos, object_2D_pos])
    #transition_model = tf.keras.Model(inputs= input_s, outputs=relative_2D_pose)
    #transition_model.summary()
    if add_action:
        final_s = layers.Concatenate()([gripper_2D_pos, gripper_finger_joint_angles,
        relative_2D_pose, object_thetaz, one_hot_object_id, mapped_action])
    else:
        final_s = layers.Concatenate()([gripper_2D_pos, gripper_finger_joint_angles,
        relative_2D_pose, object_thetaz, one_hot_object_id])
    return final_s

def get_image_input_layer(image_input, output_dimension):

    conv_layer_dim = 32
    conv_layer_dim_2 = 16
    conv_layer_dim_3 = 8
    intermediate_dim = 32
    dense_layer_dimension = output_dimension

    x = layers.Conv2D(conv_layer_dim, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(conv_layer_dim_2, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(conv_layer_dim_3, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    #conv_1 = layers.Conv2D(conv_layer_dim, kernel_size=(3,3), activation='relu')(image_input)
    #conv_2 = layers.Conv2D(conv_layer_dim_2, kernel_size=(3,3), activation='relu')(conv_1)
    #max_pool_layer = layers.MaxPooling2D(pool_size=(2,2))(conv_2)
    flatten_layer = layers.Flatten()(encoded)
    final_dense_layer_1 = layers.Dense(intermediate_dim, activation='relu')(flatten_layer)
    final_dense_layer = layers.Dense(dense_layer_dimension)(final_dense_layer_1)
    #image_conv_model = tf.keras.Model(image_input, final_dense_layer, name = 'image_conv_model')
    #return image_conv_model
    return final_dense_layer
def get_image_conv_decoder(encoded):
    conv_layer_dim = 32
    conv_layer_dim_2 = 16
    conv_layer_dim_3 = 8
    #x = layer
    x = layers.Conv2D(conv_layer_dim_3, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(conv_layer_dim_2, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(conv_layer_dim, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded

def construct_encoder(input_s, final_s, expected_output_processed, expected_output, latent_dim, intermediate_dim):

    final_s_encoder = layers.Concatenate()([final_s, expected_output_processed])
    final_s_1 = layers.Dense(intermediate_dim, activation='relu')(final_s_encoder)
    z_mean = layers.Dense(latent_dim, name='z_mean')(final_s_1)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(final_s_1)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = tf.keras.Model([input_s, expected_output], [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return encoder,kl_loss

def construct_decoder(input_s, final_s, output_dim, intermediate_dim, latent_dim):
    # build decoder model
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    cond_decoder_inputs = layers.Concatenate()([latent_inputs, final_s])
    x = layers.Dense(intermediate_dim, activation='relu')(cond_decoder_inputs)
    outputs = layers.Dense(output_dim)(x)

    # instantiate decoder model
    decoder = tf.keras.Model([latent_inputs, input_s], outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    return decoder
def keras_function_observation_model_test(latent_dim):
    decoder_intermediate_dimension = 32
    image_h = 184
    image_w = 208
    image_h_scaled_down = image_h/8
    image_w_scaled_down = image_w/8
    output_dim = 8
    input_dim = 9
    input_s = tf.keras.Input(shape=(input_dim,), name = 'input_state')
    input_s1 = tf.keras.Input(shape=(input_dim,), name = 'input_state_1')
    final_s = prepare_state_input_layer(latent_dim,input_s)
    final_s1 = prepare_state_input_layer(latent_dim, input_s1,add_action = False)
    image_input = tf.keras.Input(shape=(image_h,image_w,1,), name = 'input_image')
    image_input_flatten = layers.Flatten()(image_input)
    #image_conv_model = get_image_input_layer(image_input, output_dim)
    #encoded_image = image_conv_model(image_input)
    encoded_image =  get_image_input_layer(image_input, output_dim)
    encoder, kl_loss = construct_encoder(input_s, final_s,encoded_image,image_input , latent_dim, decoder_intermediate_dimension)
    #encoder,kl_loss = construct_encoder(input_s, final_s, expected_output, latent_dim, decoder_intermediate_dimension)
    decoder = construct_decoder(input_s, final_s,output_dim, decoder_intermediate_dimension, latent_dim)
    # instantiate VAE model
    encoder_output = encoder([input_s, image_input])[2]
    vae_outputs = decoder([encoder_output,input_s])

    final_concatenate_layer = layers.Concatenate(name='concatenate_decoder_output')([final_s1, vae_outputs])
    dense_layer_1 = layers.Dense(decoder_intermediate_dimension, activation='relu', name='dense_layer_1')(final_concatenate_layer)
    prob_output = layers.Dense(1, name='final_dese_layer')(dense_layer_1)


    prob_input = tf.keras.Input(shape=(1,), name = 'prob_input')







    vae = tf.keras.Model([input_s, image_input, input_s1, prob_input], [vae_outputs,prob_output], name='vae_mlp')
    reconstruction_loss = mse(encoded_image, vae_outputs)
    reconstruction_loss *= output_dim

    prob_loss = mse(prob_input, prob_output)
    vae_loss = K.mean(reconstruction_loss + kl_loss + prob_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_observation_model.png',
               show_shapes=True)
    return vae,encoder, decoder
def keras_function_observation_model(latent_dim):
    decoder_intermediate_dimension = 32
    image_h = 184
    image_w = 208
    image_h_scaled_down = image_h/8
    image_w_scaled_down = image_w/8
    output_dim = 8
    input_dim = 9
    input_s = tf.keras.Input(shape=(input_dim,), name = 'input_state')
    input_s1 = tf.keras.Input(shape=(input_dim,), name = 'input_state_1')
    final_s = prepare_state_input_layer(latent_dim,input_s)
    final_s1 = prepare_state_input_layer(latent_dim, input_s1,add_action = False)
    image_input = tf.keras.Input(shape=(image_h,image_w,1,), name = 'input_image')
    image_input_flatten = layers.Flatten()(image_input)
    #image_conv_model = get_image_input_layer(image_input, output_dim)
    #encoded_image = image_conv_model(image_input)
    encoded_image =  get_image_input_layer(image_input, output_dim)
    encoder, kl_loss = construct_encoder(input_s, final_s,encoded_image,image_input , latent_dim, decoder_intermediate_dimension)
    #encoder,kl_loss = construct_encoder(input_s, final_s, expected_output, latent_dim, decoder_intermediate_dimension)
    decoder = construct_decoder(input_s, final_s,output_dim, decoder_intermediate_dimension, latent_dim)
    # instantiate VAE model
    encoder_output = encoder([input_s, image_input])[2]
    vae_outputs = decoder([encoder_output,input_s])

    encoded_conv_image_size = image_h_scaled_down*image_w_scaled_down

    encoded_conv_image_flat = layers.Dense(encoded_conv_image_size, activation='relu', name='dense_image')(vae_outputs)
    encoded_conv_image = layers.Reshape((image_h_scaled_down,image_w_scaled_down,1), name='reshape_image')(encoded_conv_image_flat)
    image_output = get_image_conv_decoder(encoded_conv_image)
    image_output_flatten = layers.Flatten()(image_output)
    #print image_output.shape
    #image_conv_model_output = image_conv_model(image_input)

    final_concatenate_layer = layers.Concatenate(name='concatenate_decoder_output')([final_s1, vae_outputs])
    dense_layer_1 = layers.Dense(decoder_intermediate_dimension, activation='relu', name='dense_layer_1')(final_concatenate_layer)
    prob_output = layers.Dense(1, name='final_dese_layer')(dense_layer_1)
    vae_without_loss = tf.keras.Model([input_s, image_input, input_s1], [vae_outputs, image_output, prob_output], name='vae_mlp')
    vae_without_loss.summary()
    plot_model(vae_without_loss, to_file='vae_without_loss_observation_model.png', show_shapes=True)

    prob_input = tf.keras.Input(shape=(1,), name = 'prob_input')
    vae_multi_outputs = vae_without_loss([input_s,image_input,input_s1])
    vae = tf.keras.Model([input_s, image_input, input_s1, prob_input], vae_multi_outputs, name='vae_mlp_loss')
    #vae = tf.keras.Model([input_s, image_input, input_s1, prob_input], [vae_outputs, image_output, prob_output], name='vae_mlp')

    #reconstruction_loss = mse(encoded_image, vae_multi_outputs[0])
    reconstruction_loss = mse(encoded_image, vae_outputs)
    reconstruction_loss *= output_dim
    print reconstruction_loss.shape
    #reconstruction_loss_image = mse(image_input, vae_multi_outputs[1])

    reconstruction_loss_image = mse(image_input_flatten, image_output_flatten)
    reconstruction_loss_image *=(image_h*image_w)
    print reconstruction_loss_image.shape
    #print prob_input.shape
    #print vae_multi_outputs[2].shape
    prob_loss = mse(prob_input, vae_multi_outputs[2])
    #prob_loss = mse(prob_input, prob_output)
    print prob_loss.shape
    print kl_loss.shape
    #vae_loss = K.mean(reconstruction_loss + reconstruction_loss_image + kl_loss )
    vae_loss = K.mean(reconstruction_loss + reconstruction_loss_image + kl_loss + prob_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_observation_model.png',
               show_shapes=True)
    return vae, encoder, decoder

def keras_functional_transition_model(latent_dim):
    input_dim = 9 #gripper_2D_pos, object_2D_pose, theta z, joint_angles, object_id, action id
    input_s = tf.keras.Input(shape=(input_dim,), name = 'input_state')
    final_s = prepare_state_input_layer(latent_dim, input_s)
    output_dim = 9 #delta gripper_2D_pos, delta object_2D_pose, delta theta z, delta joint_angles, touch values
    intermediate_dim = 32

    expected_output = tf.keras.Input(shape=(output_dim,), name='next_state_output')


    #final_s_encoder = layers.Concatenate()([final_s, expected_output])
    encoder,kl_loss = construct_encoder(input_s, final_s, expected_output, expected_output, latent_dim, intermediate_dim)


    decoder = construct_decoder(input_s, final_s, output_dim, intermediate_dim, latent_dim)
    # build decoder model
    #latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    #cond_decoder_inputs = layers.Concatenate()([latent_inputs, final_s])
    #x = layers.Dense(intermediate_dim, activation='relu')(cond_decoder_inputs)
    #outputs = layers.Dense(output_dim)(x)

    # instantiate decoder model
    #decoder = tf.keras.Model([latent_inputs, input_s], outputs, name='decoder')
    #decoder.summary()
    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)


    # instantiate VAE model
    encoder_output = encoder([input_s, expected_output])[2]
    vae_outputs = decoder([encoder_output,input_s])
    vae = tf.keras.Model([input_s, expected_output], vae_outputs, name='vae_mlp')

    reconstruction_loss = mse(expected_output, vae_outputs)


    reconstruction_loss *= input_dim

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_transition_model.png',
               show_shapes=True)
    #tranition_model_encoder = tf.keras.Model()

    #transition_model = tf.keras.Model(inputs= [gripper_2D_pos, object_2D_pos], outputs=relative_2D_pose)
    return vae, encoder, decoder

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

def train_transition_model(object_name_list, data_dir, object_id_mapping_file):
    batch_size = 128
    epochs = 50
    input_s, expected_outcome, image_input = dataProcessor.get_training_data(object_name_list, data_dir, object_id_mapping_file)
    latent_dim = 2
    sess = K.get_session()

    vae, encoder, decoder = keras_functional_transition_model(latent_dim)
    vae.fit([expected_outcome, input_s],
                epochs=epochs,
                batch_size=batch_size)
    print expected_outcome.shape
    print input_s.shape
    vae.save_weights('vae_transition_model.h5')
    #Y = vae.predict([expected_outcome, input_s])
    #print Y
    #model.summary()
    #X = np.array([[0,1,2,3,4,5,6,7,8,9,10]]).transpose()
    #X = np.random.random((10, 2))
    #X1 = np.random.random((10, 2))
    #print X.shape
    #Y = model.predict([X,X1])
    #print Y
    #plot_model(model,
    #           to_file='vae_mlp.png',
    #           show_shapes=True)


def train_observation_model(object_name_list, data_dir, object_id_mapping_file):
    #from tensorflow.python import debug as tf_debug
    batch_size = 128
    epochs = 50
    input_s_gen, image_input_gen, input_s_existing, prob = dataProcessor.get_training_data_for_observation_model(object_name_list, data_dir, object_id_mapping_file, '../')
    latent_dim = 2
    sess = K.get_session()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #K.set_session(sess)
    vae, encoder, decoder = keras_function_observation_model(latent_dim)
    print input_s_gen.shape
    print image_input_gen.shape
    print input_s_existing.shape
    print prob.shape
    vae.fit([input_s_gen, image_input_gen, input_s_existing, prob],
            epochs=epochs,
            batch_size=batch_size)
    vae.save_weights('vae_observation_model.h5')

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
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    #save_model()
    object_id_mapping_file = '../ObjectNameToIdMapping.yaml'
    data_dir = '../data_low_friction_table_exp_wider_object_workspace_ver8/data_for_regression'
    object_id_mapping = yaml.load(file(object_id_mapping_file, 'r'))
    object_name_list = object_id_mapping.keys()
    #dataProcessor.get_training_data(object_name_list, data_dir, object_id_mapping_file)
    train_observation_model(object_name_list, data_dir, object_id_mapping_file)
    #train_model(object_name_list, data_dir, object_id_mapping_file)
    #keras_functional_transition_model(2)
    #print "Creating observation model"
    #keras_function_observation_model(2)
