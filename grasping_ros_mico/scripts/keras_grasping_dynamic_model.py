
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


#Copied from : https://www.dlology.com/blog/how-to-convert-trained-keras-model-to-tensorflow-and-make-prediction/
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



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

#wrong lambda function. not being used
def get_close_called(x,action_id):
    z = tf.cast(x,tf.uint8)
    if(K.equal(action_id,8)):
        if(K.equal(z,0)):
            K.update(z,1)
    if(K.equal(action_id,9)):
        if(K.equal(z,1)):
            K.update(z,0)
    return z
def get_close_called_close_action(x):
    #z = tf.cast(x,tf.uint8)
    #y = tf.where(z < 1, tf.ones_like(z), z)
    y = tf.ones_like(x)
    return tf.cast(y,tf.float32)

def get_close_called_open_action(x):
    #z = tf.cast(x,tf.uint8)
    #y = tf.where(z > 0, tf.zeros_like(z), z)
    y = tf.zeros_like(x)
    return tf.cast(y,tf.float32)

def clip_gripper_values(x):
    gripper_x,gripper_y,rem = tf.split(x,[1,1,5],1)
    gripper_x_1 = tf.where(gripper_x < 0.3379, 0.3379*tf.ones_like(gripper_x), gripper_x)
    gripper_x_2 = tf.where(gripper_x_1 > 0.5279, 0.5279*tf.ones_like(gripper_x), gripper_x_1)

    gripper_y_1 = tf.where(gripper_y < 0.0816, 0.0816*tf.ones_like(gripper_y), gripper_y)
    gripper_y_2 = tf.where(gripper_y_1 > 0.2316, 0.2316*tf.ones_like(gripper_y), gripper_y_1)
    return tf.concat([gripper_x_2,gripper_y_2,rem],1)

def get_reward_move_actions(args):
    close_called_input = args[0]
    left_touch,right_touch =  tf.split(args[1],[1,1],1)
    y = tf.where(left_touch >= 0.35, -0.5*tf.ones_like(left_touch), -1*tf.ones_like(left_touch))
    y1 = tf.where(right_touch >= 0.35, -0.5*tf.ones_like(left_touch), y)
    y2 = tf.where(close_called_input > 0.5, -1000*tf.ones_like(left_touch), y1)
    return y2
def get_reward_close_action(args):
    close_called_input = args[0]
    left_touch,right_touch =  tf.split(args[1],[1,1],1)
    #y = tf.where(left_touch >= 0.35, -0.5*tf.ones_like(left_touch), -1*tf.ones_like(left_touch))
    #y1 = tf.where(right_touch >= 0.35, -0.5*tf.ones_like(left_touch), y)
    y = tf.where(left_touch >= 0.35, -1*tf.ones_like(left_touch), -1*tf.ones_like(left_touch))
    y1 = tf.where(right_touch >= 0.35, y, y)
    y2 = tf.where(close_called_input > 0.5, -1000*tf.ones_like(left_touch), y1)
    return y2
def get_reward_open_action(args):
    close_called_input = args[0]
    left_touch,right_touch =  tf.split(args[1],[1,1],1)
    #y = tf.where(left_touch >= 0.35, -0.5*tf.ones_like(left_touch), -1*tf.ones_like(left_touch))
    #y1 = tf.where(right_touch >= 0.35, -0.5*tf.ones_like(left_touch), y)
    y = tf.where(left_touch >= 0.35, -1*tf.ones_like(left_touch), -1*tf.ones_like(left_touch))
    y1 = tf.where(right_touch >= 0.35, y, y)
    y2 = tf.where(close_called_input < 0.5, -1000*tf.ones_like(left_touch), y1)
    return y2
def get_reward_pick_action(args):
    random_number_input = args[0]
    zero_prob,one_prob =  tf.split(args[1],[1,1],1)
    #close_called_input = tf.cast(args[2], tf.uint8)
    close_called_input = args[2]
    y = tf.where(random_number_input < zero_prob, -10*tf.ones_like(random_number_input), 100*tf.ones_like(random_number_input))
    y1 = tf.where(close_called_input < 0.5, -1000*tf.ones_like(random_number_input), y)
    return y1
def get_terminal_value_non_pick(x):
    return tf.zeros_like(x)
def get_terminal_value_pick(x):
    return tf.ones_like(x)

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




def prepare_state_input_layer(latent_dim, input_s,add_action = False):
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

    #gripper_2D_pos, object_2D_pos, object_thetaz, gripper_finger_joint_angles,object_id, action_id= layers.Lambda(lambda x: tf.split(x,[2,2,1,2,1,1],1), name='split_input')(input_s)
    gripper_2D_pos, object_2D_pos, object_thetaz, gripper_finger_joint_angles,object_id= layers.Lambda(lambda x: tf.split(x,[2,2,1,2,1],1), name='split_input')(input_s)
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

    #if add_action:
    #    mapped_action_without_flatten = layers.Lambda(map_actions, name = 'map_action', arguments={'num_classes': num_action_classes})(action_id)
    #    mapped_action = layers.Flatten()(mapped_action_without_flatten)

    #object_id = input_s[8:9]
    #object_id = layers.Lambda( lambda x: tf.slice(x, (8), (1)))(input_s)
    #object_id = tf.keras.Input(shape = (1,), name='object_id_input')
    one_hot_object_id_without_flatten = layers.Lambda(get_one_hot, name = 'map_object_id', arguments={'num_classes': num_object_classes})(object_id)
    one_hot_object_id = layers.Flatten()(one_hot_object_id_without_flatten)

    relative_2D_pose = layers.Lambda(get_relative_pos, name = 'rel_object_pos')([gripper_2D_pos, object_2D_pos])
    #transition_model = tf.keras.Model(inputs= input_s, outputs=relative_2D_pose)
    #transition_model.summary()

    #if add_action:
    #    final_s = layers.Concatenate()([gripper_2D_pos, gripper_finger_joint_angles,
    #    relative_2D_pose, object_thetaz, one_hot_object_id, mapped_action])
    #else:
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
def construct_image_decoder(output_dim, encoded_conv_image_size,image_h_scaled_down, image_w_scaled_down):
    vae_outputs = tf.keras.Input(shape=(output_dim,), name='vae_inputs')
    encoded_conv_image_flat = layers.Dense(encoded_conv_image_size, activation='relu', name='dense_image')(vae_outputs)
    encoded_conv_image = layers.Reshape((image_h_scaled_down,image_w_scaled_down,1), name='reshape_image')(encoded_conv_image_flat)
    image_output = get_image_conv_decoder(encoded_conv_image)
    #image_output_flatten = layers.Flatten()(image_output)
    image_decoder = tf.keras.Model([vae_outputs], [image_output])
    image_decoder.summary()
    plot_model(image_decoder, to_file='vae_mlp_image_decoder.png', show_shapes=True)
    return image_decoder

def construct_prob_decoder(output_dim, decoder_intermediate_dimension, input_s1,final_s1):
    vae_outputs = tf.keras.Input(shape=(output_dim,), name='vae_inputs')
    final_concatenate_layer = layers.Concatenate(name='concatenate_decoder_output')([final_s1, vae_outputs])
    dense_layer_1 = layers.Dense(decoder_intermediate_dimension, activation='relu', name='dense_layer_1')(final_concatenate_layer)
    prob_output = layers.Dense(1, name='final_dese_layer')(dense_layer_1)
    prob_decoder = tf.keras.Model([input_s1, vae_outputs], [prob_output])
    prob_decoder.summary()
    plot_model(prob_decoder, to_file='vae_mlp_prob_decoder.png', show_shapes=True)
    return prob_decoder

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

def keras_function_observation_model_without_image_prob_decoder(latent_dim):
    decoder_intermediate_dimension = 32
    image_h = 184
    image_w = 208
    image_h_scaled_down = image_h/8
    image_w_scaled_down = image_w/8
    output_dim = 8
    input_dim = 8
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
    #image_decoder = construct_image_decoder(output_dim, encoded_conv_image_size,image_h_scaled_down, image_w_scaled_down)
    encoded_conv_image_flat = layers.Dense(encoded_conv_image_size, activation='relu', name='dense_image')(vae_outputs)
    encoded_conv_image = layers.Reshape((image_h_scaled_down,image_w_scaled_down,1), name='reshape_image')(encoded_conv_image_flat)
    image_output = get_image_conv_decoder(encoded_conv_image)
    #image_output = image_decoder([vae_outputs])
    image_output_flatten = layers.Flatten()(image_output)

    #print image_output.shape
    #image_conv_model_output = image_conv_model(image_input)

    #prob_decoder = construct_prob_decoder(output_dim, decoder_intermediate_dimension, input_s1,final_s1)
    #prob_output = prob_decoder([input_s1,vae_outputs])
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
    #latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    #final_decoder = tf.Keras.Model([latent_inputs,input_s, input_s1], [prob_output], name = 'final_decoder_prob_output_only')
    #final_decoder_image = tf.Keras.Model([latent_inputs,input_s, input_s1], [vae_outputs, image_output,prob_output], name = 'final_decoder')

    return vae, encoder, decoder, vae_without_loss #, image_decoder, prob_decoder #, final_decoder, final_decoder_image
def keras_function_observation_model(latent_dim):
    decoder_intermediate_dimension = 32
    image_h = 184
    image_w = 208
    image_h_scaled_down = image_h/8
    image_w_scaled_down = image_w/8
    output_dim = 8
    input_dim = 8
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
    image_decoder = construct_image_decoder(output_dim, encoded_conv_image_size,image_h_scaled_down, image_w_scaled_down)
    #encoded_conv_image_flat = layers.Dense(encoded_conv_image_size, activation='relu', name='dense_image')(vae_outputs)
    #encoded_conv_image = layers.Reshape((image_h_scaled_down,image_w_scaled_down,1), name='reshape_image')(encoded_conv_image_flat)
    #image_output = get_image_conv_decoder(encoded_conv_image)
    image_output = image_decoder([vae_outputs])
    image_output_flatten = layers.Flatten()(image_output)

    #print image_output.shape
    #image_conv_model_output = image_conv_model(image_input)

    prob_decoder = construct_prob_decoder(output_dim, decoder_intermediate_dimension, input_s1,final_s1)
    prob_output = prob_decoder([input_s1,vae_outputs])
    #final_concatenate_layer = layers.Concatenate(name='concatenate_decoder_output')([final_s1, vae_outputs])
    #dense_layer_1 = layers.Dense(decoder_intermediate_dimension, activation='relu', name='dense_layer_1')(final_concatenate_layer)
    #prob_output = layers.Dense(1, name='final_dese_layer')(dense_layer_1)

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
    #latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    #final_decoder = tf.Keras.Model([latent_inputs,input_s, input_s1], [prob_output], name = 'final_decoder_prob_output_only')
    #final_decoder_image = tf.Keras.Model([latent_inputs,input_s, input_s1], [vae_outputs, image_output,prob_output], name = 'final_decoder')

    return vae, encoder, decoder, image_decoder, prob_decoder, vae_without_loss #, final_decoder, final_decoder_image

def keras_functional_transition_model(latent_dim):
    input_dim = 8 #gripper_2D_pos, object_2D_pose, theta z, joint_angles, object_id, action id, removing action_id
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
def keras_functional_transition_model_pick_action():
    input_dim = 8 #gripper_2D_pos, object_2D_pose, theta z, joint_angles, object_id, action id, removing action_id
    #input_dim = 8

    input_s = tf.keras.Input(shape=(input_dim,), name = 'input_state')
    final_s = prepare_state_input_layer(2, input_s)
    intermediate_s = layers.Dense(8, activation='relu', name='dense_layer_1')(final_s)
    pick_output = layers.Dense(2, activation='softmax', name='dense_layer_2')(intermediate_s)

    pick_model = tf.keras.Model([input_s], pick_output, name='pick_model')
    pick_model.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

    random_number_input = tf.keras.Input(shape=(1,), name = 'random_number')
    input_dim_full = 8 + 1
    input_s_full = tf.keras.Input(shape=(input_dim_full,), name = 'input_state') #input_s, close_called_input, action id needed to update close_called_input
    input_s_breakup,close_called_input = layers.Lambda(lambda x: tf.split(x,[input_dim,1],1), name='split_full_input')(input_s_full)
    pick_output_ = pick_model(input_s_breakup)
    pick_reward = layers.Lambda(get_reward_pick_action, name = 'pick_action_reward')([random_number_input,pick_output_,close_called_input])
    terminal_state_output = layers.Lambda(get_terminal_value_pick, name = 'get_terminal_value_pick')(close_called_input)
    pick_model_with_reward = tf.keras.Model([input_s_full, random_number_input], [terminal_state_output,pick_reward], name='pick_reward_model')
    return pick_model,pick_model_with_reward


def keras_functional_transiton_model_with_reward_and_terminal_state(latent_dim,action_id):
    vae, encoder, decoder = keras_functional_transition_model(latent_dim)
    #create reward and next state outputs
    input_dim = 8
    input_dim_full = 8 + 1
    input_s_full = tf.keras.Input(shape=(input_dim_full,), name = 'input_state') #input_s, close_called_input, action id needed to update close_called_input
    input_s,close_called_input = layers.Lambda(lambda x: tf.split(x,[input_dim,1],1), name='split_full_input')(input_s_full)
    final_s = prepare_state_input_layer(latent_dim, input_s)
    output_dim = 9 #delta gripper_2D_pos, delta object_2D_pose, delta theta z, delta joint_angles, touch values
    intermediate_dim = 32
    #decoder = construct_decoder(input_s, final_s, output_dim, intermediate_dim, latent_dim)
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    decoder_outputs = decoder([latent_inputs,input_s])
    object_state,object_id= layers.Lambda(lambda x: tf.split(x,[7,1],1), name='split_state')(input_s)
    object_state_change,touch_values = layers.Lambda(lambda x: tf.split(x,[7,2],1), name='split_output')(decoder_outputs)
    next_state = layers.Add()([object_state,object_state_change])
    next_state = layers.Lambda(clip_gripper_values, name = 'clip_gripper_values')(next_state)
    if(action_id==8):
        #close_called_input = layers.Lambda(get_close_called, name = 'update_close_called', arguments={'action_id': action_id})(close_called_input)

        reward = layers.Lambda(get_reward_close_action, name = 'close_action_reward')([close_called_input,touch_values])
        close_called_input = layers.Lambda(get_close_called_close_action, name = 'update_close_called_close_action')(close_called_input)
    elif(action_id ==9):

        reward = layers.Lambda(get_reward_open_action, name = 'open_action_reward')([close_called_input,touch_values])
        close_called_input = layers.Lambda(get_close_called_open_action, name = 'update_close_called_open_action')(close_called_input)
    else:
        reward = layers.Lambda(get_reward_move_actions, name = 'move_action_reward')([close_called_input,touch_values])
    next_state_full = layers.Concatenate(name='concatenate_next_state_output')([next_state, object_id,close_called_input])
    terminal_state_output = layers.Lambda(get_terminal_value_non_pick, name = 'get_terminal_value_non_pick')(close_called_input)
    full_model = tf.keras.Model([input_s_full,latent_inputs], [next_state_full,terminal_state_output,reward], name = 'full_model')
    return decoder,full_model
def keras_functional_transiton_model_with_reward_and_terminal_state_old(latent_dim,action_id):
    #create reward and next state outputs
    input_dim = 8
    input_dim_full = 8 + 1
    input_s_full = tf.keras.Input(shape=(input_dim_full,), name = 'input_state') #input_s, close_called_input, action id needed to update close_called_input
    input_s,close_called_input = layers.Lambda(lambda x: tf.split(x,[input_dim,1],1), name='split_full_input')(input_s_full)
    final_s = prepare_state_input_layer(latent_dim, input_s)
    output_dim = 9 #delta gripper_2D_pos, delta object_2D_pose, delta theta z, delta joint_angles, touch values
    intermediate_dim = 32
    decoder = construct_decoder(input_s, final_s, output_dim, intermediate_dim, latent_dim)
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    decoder_outputs = decoder([latent_inputs,input_s])
    object_state,object_id= layers.Lambda(lambda x: tf.split(x,[7,1],1), name='split_state')(input_s)
    object_state_change,touch_values = layers.Lambda(lambda x: tf.split(x,[7,2],1), name='split_output')(decoder_outputs)
    next_state = layers.Add()([object_state,object_state_change])
    next_state = layers.Lambda(clip_gripper_values, name = 'clip_gripper_values')(next_state)
    if(action_id==8):
        #close_called_input = layers.Lambda(get_close_called, name = 'update_close_called', arguments={'action_id': action_id})(close_called_input)
        reward = layers.Lambda(get_reward_close_action, name = 'close_action_reward')([close_called_input,touch_values])
        close_called_input = layers.Lambda(get_close_called_close_action, name = 'update_close_called_close_action')(close_called_input)
    elif(action_id ==9):

        reward = layers.Lambda(get_reward_open_action, name = 'open_action_reward')([close_called_input,touch_values])
        close_called_input = layers.Lambda(get_close_called_open_action, name = 'update_close_called_open_action')(close_called_input)
    else:
        reward = layers.Lambda(get_reward_move_actions, name = 'move_action_reward')([close_called_input,touch_values])
    next_state_full = layers.Concatenate(name='concatenate_next_state_output')([next_state, object_id,close_called_input])
    terminal_state_output = layers.Lambda(get_terminal_value_non_pick, name = 'get_terminal_value_non_pick')(close_called_input)
    full_model = tf.keras.Model([input_s_full,latent_inputs], [next_state_full,terminal_state_output,reward], name = 'full_model')
    return decoder,full_model

def get_final_observation_model(latent_dim, v1=False):
    vae, encoder, decoder,image_decoder,prob_decoder, vae_without_loss = keras_function_observation_model(latent_dim)
    input_dim = 8
    input_dim_full = 8 + 1
    input_s_full1 = tf.keras.Input(shape=(input_dim_full,), name = 'input_state1') #input_s, close_called_input, action id needed to update close_called_input
    input_s1,close_called_input1 = layers.Lambda(lambda x: tf.split(x,[input_dim,1],1), name='split_full_input1')(input_s_full1)
    if v1:
        decoder_output_dim = 8
        decoder_outputs = tf.keras.Input(shape=(decoder_output_dim,), name = 'obs_input')
    else:
        input_s_full = tf.keras.Input(shape=(input_dim_full,), name = 'input_state') #input_s, close_called_input, action id needed to update close_called_input
        input_s,close_called_input = layers.Lambda(lambda x: tf.split(x,[input_dim,1],1), name='split_full_input')(input_s_full)
        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
        decoder_outputs = decoder([latent_inputs,input_s])


    prob_output = prob_decoder([input_s1,decoder_outputs])
    if v1:
        full_model = tf.keras.Model([input_s_full1,decoder_outputs], prob_output, name = 'full_prob_model')
    else:
        full_model = tf.keras.Model([input_s_full,input_s_full1,latent_inputs], prob_output, name = 'full_prob_model')
    return vae, encoder, decoder,image_decoder,prob_decoder, vae_without_loss, full_model

def keras_functional_transiton_model_with_reward_and_terminal_state_v1(latent_dim,action_id):
    vae, encoder, decoder,image_decoder,prob_decoder, vae_without_loss = keras_function_observation_model(latent_dim)
    decoder1,full_model = keras_functional_transiton_model_with_reward_and_terminal_state(latent_dim,action_id)
    input_dim = 8
    input_dim_full = 8 + 1
    input_s_full = tf.keras.Input(shape=(input_dim_full,), name = 'input_state') #input_s, close_called_input, action id needed to update close_called_input
    input_s,close_called_input = layers.Lambda(lambda x: tf.split(x,[input_dim,1],1), name='split_full_input')(input_s_full)
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
    [next_state_full,terminal_state_output,reward] = full_model([input_s_full,latent_inputs])
    obs_output = decoder([latent_inputs,input_s])
    full_model_v1 = tf.keras.Model([input_s_full,latent_inputs], [next_state_full,terminal_state_output,reward,obs_output], name = 'full_model_v1')
    return decoder, decoder1, full_model_v1



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

def train_transition_model(action_,object_name_list, data_dir, object_id_mapping_file, gpuID):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    batch_size = 128
    epochs = 50
    sess = K.get_session()
    data_loader = dataProcessor.LoadTransitionData()
    for action in range(0,10):
        if(action % 2 == 0 or action > 8):
            if(action==action_ or action_ < 0):
                input_s, expected_outcome, image_input = data_loader.get_training_data(action, object_name_list, data_dir, object_id_mapping_file)
                latent_dim = 2
                vae, encoder, decoder = keras_functional_transition_model(latent_dim)
                num_samples = input_s.shape[0]
                arr = np.arange(num_samples)
                np.random.shuffle(arr)
                input_s_shuffled = input_s[arr[0:num_samples]]
                expected_outcome_shuffled = expected_outcome[arr[0:num_samples]]

                vae.fit([ input_s_shuffled, expected_outcome_shuffled],
                    epochs=epochs,
                    batch_size=batch_size, validation_split = 0.05)
                print expected_outcome.shape
                print input_s.shape
                vae.save('transition_model/vae_transition_model_' + repr(action) + '.h5')
    if action_ == 10: #pick_action
        input_s,y = data_loader.get_training_data(action_, object_name_list, data_dir, object_id_mapping_file)
        y_categorical = tf.keras.utils.to_categorical(y, 2)
        pick_model, pick_model_with_reward = keras_functional_transition_model_pick_action()

        num_samples = input_s.shape[0]
        arr = np.arange(num_samples)
        np.random.shuffle(arr)
        input_s_shuffled = input_s[arr[0:num_samples]]
        y_shuffled = y_categorical[arr[0:num_samples]]


        pick_model.fit(input_s_shuffled,y_shuffled,
            epochs=epochs,
                    batch_size=batch_size, validation_split = 0.05)
        pick_model.save('transition_model/pick_transition_model_' + repr(action_) + '.h5')
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


def train_observation_model(action_, object_name_list, data_dir, object_id_mapping_file, gpuID):
    #from tensorflow.python import debug as tf_debug
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    batch_size = 128
    epochs = 50
    latent_dim = 2
    sess = K.get_session()
    data_loader = dataProcessor.LoadTransitionData()
    action_samples = {}
    #action_samples[0] = [14136047, 744002]
    action_samples[0] = [1413647, 74402]
    action_samples[2] = [1412849, 74360]
    action_samples[4] = [1412369, 74335 ]
    action_samples[6] = [1412849, 74360 ]
    action_samples[8] = [1425000, 75000 ]
    action_samples[9] = [1413064, 74372]

    for action in range(0,10):
        if(action % 2 == 0 or action > 8):
            if(action==action_ or action_ < 0):
                    #input_s_gen, image_input_gen, input_s_existing, prob = data_loader.get_training_data_for_observation_model(action, object_name_list, data_dir, object_id_mapping_file, '../')
                    training_data_file_name = 'observation_model/data_cluster_size_10/' + repr(action) + '_train.data'
                    validation_data_file_name = 'observation_model/data_cluster_size_10/' + repr(action) + '_val.data'
                    num_training_samples = action_samples[action][0]
                    num_val_samples = action_samples[action][1]
                    training_generator = data_loader.observation_data_generator(training_data_file_name, '../', batch_size)
                    validation_generator = data_loader.observation_data_generator(validation_data_file_name, '../', batch_size)

                    train_steps_per_epoch = math.ceil(num_training_samples*1.0/batch_size)
                    val_steps_per_epoch = math.ceil(num_val_samples*1.0/batch_size)
                    #num_samples = input_s_gen.shape[0]
                    #arr = np.arange(num_samples)
                    #np.random.shuffle(arr)


                    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                    #K.set_session(sess)
                    #vae, encoder, decoder, vae_without_loss = keras_function_observation_model_without_image_prob_decoder(latent_dim)
                    vae, encoder, decoder,image_decoder,prob_decoder,vae_without_loss = keras_function_observation_model(latent_dim)
                    #print input_s_gen.shape
                    #print image_input_gen.shape
                    #print input_s_existing.shape
                    #print prob.shape

                    #input_s_gen_shuffled = input_s_gen[arr[0:num_samples]]
                    #image_input_gen_shuffled = image_input_gen[arr[0:num_samples]]
                    #input_s_existing_shuffled = input_s_existing[arr[0:num_samples]]
                    #prob_shuffled = prob[arr[0:num_samples]]
                    #vae.fit([input_s_gen_shuffled, image_input_gen_shuffled, input_s_existing_shuffled, prob_shuffled],
                    #epochs=epochs,
                    #batch_size=batch_size, validation_split = 0.05)
                    vae.fit_generator(training_generator, steps_per_epoch = train_steps_per_epoch, epochs = epochs,
                    validation_data = validation_generator, validation_steps = val_steps_per_epoch)
                    vae.save('observation_model/vae_observation_model_' + repr(action) +'.h5')
                    if action_ < 0 :
                        break;

def load_observation_model(action,gpuID):
    K.set_learning_phase(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    latent_dim = 2
    observation_model_name = 'observation_model/vae_observation_model_' + repr(action) +'.h5'
    vae, encoder, decoder, vae_without_loss = keras_function_observation_model_without_image_prob_decoder(latent_dim)
    vae1, encoder1, decoder1,image_decoder1,prob_decoder1, vae_without_loss1 = keras_function_observation_model(latent_dim)
    #vae1 = tf.keras.models.load_model(observation_model_name)
    #vae.set_weights(vae1.get_weights())
    vae.load_weights(observation_model_name)
    #vae1.set_weights(vae.get_weights())
    encoder1.set_weights(encoder.get_weights())
    decoder1.set_weights(decoder.get_weights())

    all_layers = image_decoder1.layers + prob_decoder1.layers
    for layer in all_layers:
        weights_set = False
        for layer1 in vae_without_loss.layers:
            if layer.name == layer1.name:
                print "Setting weights:" + layer.name
                layer.set_weights(layer1.get_weights())
                weights_set = True
                break
        if not weights_set:
            print "Weight not set:" + layer.name
    for i in range(10,14):
        layer_name = 'conv2d_' + repr(i)
        layer_name1 = 'conv2d_' + repr(i-7)
        image_decoder1.get_layer(layer_name).set_weights(vae_without_loss.get_layer(layer_name1).get_weights())
    for layer in vae_without_loss.layers:
        print layer.name
        #print layer.get_weights()
    print "Printing new model layers aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    for layer in vae_without_loss1.layers:
        print layer.name
    print "Printing new model layers aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    for layer in image_decoder1.layers:
        print layer.name
    print "Printing new model layers aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    for layer in prob_decoder1.layers:
        print layer.name

    #X = np.array([[0.387915, 0.091614, 0.601582, 0.0546314, -1.04977531092, -0.00016737, -0.000296831, 1.0]])
    X = np.array([[0.437916, 0.151595, 0.561615, 0.102654, 1.93312538491, -0.000924826, -0.000560999, 2.0]])
    X1 = np.array([[0.377922, 0.201654, 0.557641, 0.09265, 2.7918653026, 0.000204325, 2.19345e-05, 0.0],
                    [0.437916, 0.151595, 0.561615, 0.102654, 1.93312538491, -0.000924826, -0.000560999, 2.0],
                    [0.427921, 0.0916133, 0.555786, 0.0946642, 0.872971909477, 0.000200272, -0.000123262, 0.0],
                    [0.367916, 0.211634, 0.561941, 0.0943089, -1.57313648144, -0.000118017, -0.000352859, 1.0]])
    #R = np.array([[0.9,0.5]])#K.random_normal(shape=(1, latent_dim))#np.array([[0.9,0.5]])
    R = np.array([np.random.normal(0,1,latent_dim)])
    Y = decoder.predict([R,X])
    print Y
    print decoder.input
    print vae_without_loss.get_layer('conv2d_6').output
    print decoder.output
    out_image = image_decoder1.predict(Y)
    print out_image.shape
    depth_im = perception.DepthImage(out_image[0])
    depth_im.save('test.png')

    Y1 = prob_decoder1.predict([np.array([X1[0]]),Y])
    print Y1
    #get_3rd_layer_output = K.function([decoder.output],
    #                              [vae_without_loss.get_layer('conv2d_6').output])
    #layer_output = get_3rd_layer_output(np.array(Y))
    #print layer_output.shape
    #intermediate_layer_model = tf.keras.Model(inputs=decoder.input,
    #                             outputs=vae_without_loss.get_layer('conv2d_6').output, name = 'inter_model')
    #intermediate_output = intermediate_layer_model.predict([R,X])
    #print intermediate_output.shape
    #print vae.get_weights()
    #print "Printing decoder weights aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    #print decoder.get_weights()


def save_final_observation_model(action,gpuID, v1=False):
    K.set_learning_phase(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    latent_dim = 2
    observation_model_name = 'observation_model/vae_observation_model_' + repr(action) +'.h5'
    if(action==8):
        vae1, encoder1, decoder1,image_decoder1,prob_decoder1, vae_without_loss1, full_model = get_final_observation_model(latent_dim,v1)

        vae1.load_weights(observation_model_name)
    else:
        vae, encoder, decoder, vae_without_loss = keras_function_observation_model_without_image_prob_decoder(latent_dim)
        vae1, encoder1, decoder1,image_decoder1,prob_decoder1, vae_without_loss1, full_model = get_final_observation_model(latent_dim,v1)
        vae.load_weights(observation_model_name)

        encoder1.set_weights(encoder.get_weights())
        decoder1.set_weights(decoder.get_weights())
        all_layers = image_decoder1.layers + prob_decoder1.layers
        for layer in all_layers:
            weights_set = False
            for layer1 in vae_without_loss.layers:
                if layer.name == layer1.name:
                    print "Setting weights:" + layer.name
                    layer.set_weights(layer1.get_weights())
                    weights_set = True
                    break
            if not weights_set:
                print "Weight not set:" + layer.name
        for i in range(10,14):
            layer_name = 'conv2d_' + repr(i)
            layer_name1 = 'conv2d_' + repr(i-7)
            image_decoder1.get_layer(layer_name).set_weights(vae_without_loss.get_layer(layer_name1).get_weights())
        for layer in vae_without_loss.layers:
            print layer.name
            #print layer.get_weights()
        print "Printing new model layers aaaaaaaaaaaaaaaaaaaaaaaaaaa"
        for layer in vae_without_loss1.layers:
            print layer.name
        print "Printing new model layers aaaaaaaaaaaaaaaaaaaaaaaaaaa"
        for layer in image_decoder1.layers:
            print layer.name
        print "Printing new model layers aaaaaaaaaaaaaaaaaaaaaaaaaaa"
        for layer in prob_decoder1.layers:
            print layer.name

    if(v1):
        print full_model.outputs
        print full_model.inputs
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in full_model.outputs])
        tf.train.write_graph(frozen_graph, ".", 'observation_model/full_observation_model_v1_' + repr(action) +'.pb', as_text=False)
    else:
        print full_model.outputs
        print prob_decoder1.inputs
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in full_model.outputs])
        tf.train.write_graph(frozen_graph, ".", 'observation_model/full_observation_model_' + repr(action) +'.pb', as_text=False)

def save_transiton_model_with_reward_graph_v1(action,gpuID):
    K.set_learning_phase(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    latent_dim = 2
    if(action < 10):
        transition_model_name = 'transition_model/vae_transition_model_' + repr(action) +'.h5'
        observation_model_name = 'observation_model/vae_observation_model_' + repr(action) +'.h5'
        vae1, encoder1, decoder1 = keras_functional_transition_model(latent_dim)
        vae1.load_weights(transition_model_name)
        if(action==8):
            vae, encoder, decoder,image_decoder,prob_decoder, vae_without_loss, full_model = get_final_observation_model(latent_dim)
            vae.load_weights(observation_model_name)
        else:
            vae, encoder, decoder, vae_without_loss = keras_function_observation_model_without_image_prob_decoder(latent_dim)
            vae.load_weights(observation_model_name)
        #vae1 = tf.keras.models.load_model(transition_model_name)
        #vae.set_weights(vae1.get_weights())

        decoder_, decoder1_,full_model = keras_functional_transiton_model_with_reward_and_terminal_state_v1(latent_dim,action)
        decoder1_.set_weights(decoder1.get_weights())
        decoder_.set_weights(decoder.get_weights())

        print full_model.outputs
        print full_model.inputs
        #X = np.array([[0.387915, 0.091614, 0.601582, 0.0546314, -1.04977531092, -0.00016737, -0.000296831, 1.0, 1.0]])
        X = np.array([[0.33738 ,0.15156 ,0.55902 ,0.11335, -0.031684, 1.0613, 1.061, 0 ,1]])
        X1 = np.array([[0.33738 ,0.15156 ,0.55902 ,0.11335, -0.031684, 1.0613, 1.061, 0]])
        #X = np.array([[0.33738 ,0.15156 ,0.55902 ,0.11335, -0.031684, 0.00001, 0.0002, 0 ,0]])
        #X1 = np.array([[0.33738 ,0.15156 ,0.55902 ,0.11335, -0.031684, 0.00001, 0.0002, 0]])
        R = np.array([np.random.normal(0,1,latent_dim)])
        Y = full_model.predict([X,R])
        Y1 = decoder1_.predict([R,X1])
        print X
        print Y1
        print Y
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in full_model.outputs])
        tf.train.write_graph(frozen_graph, ".", 'transition_model/full_transition_model_v1_' + repr(action) +'.pb', as_text=False)

    if action == 10:
        transition_model_name = 'transition_model/pick_transition_model_' + repr(action) +'.h5'
        pick_model, full_model = keras_functional_transition_model_pick_action()
        pick_model.load_weights(transition_model_name)
        print full_model.outputs
        print full_model.inputs
        #X = np.array([[0.387915, 0.091614, 0.601582, 0.0546314, -1.04977531092, -0.00016737, -0.000296831, 1.0, 1.0]])
        X = np.array([[0.5279, 0.1516, 0.52742, 0.109544, 0, -0.00016737, -0.000296831, 0.0, 1.0]])
        R = np.array([0.5])
        Y = full_model.predict([X,R])
        #X1 = np.array([[0.387915, 0.091614, 0.601582, 0.0546314, -1.04977531092, -0.00016737, -0.000296831, 1.0]])
        X1 = np.array([[0.5279, 0.1516, 0.52742, 0.109544, 0, -0.00016737, -0.000296831, 0.0]])

        Y1 = pick_model.predict(X1)
        print Y
        print Y1
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in full_model.outputs])
        tf.train.write_graph(frozen_graph, ".", 'transition_model/full_transition_model_v1_' + repr(action) +'.pb', as_text=False)
def save_transiton_model_with_reward_graph(action,gpuID):
    K.set_learning_phase(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    latent_dim = 2
    if(action < 10):
        transition_model_name = 'transition_model/vae_transition_model_' + repr(action) +'.h5'
        vae, encoder, decoder = keras_functional_transition_model(latent_dim)
        #vae1 = tf.keras.models.load_model(transition_model_name)
        #vae.set_weights(vae1.get_weights())
        vae.load_weights(transition_model_name)
        decoder1,full_model = keras_functional_transiton_model_with_reward_and_terminal_state(latent_dim,action)
        decoder1.set_weights(decoder.get_weights())
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in full_model.outputs])
        tf.train.write_graph(frozen_graph, ".", 'transition_model/full_transition_model_' + repr(action) +'.pb', as_text=False)
    if action == 10:
        transition_model_name = 'transition_model/pick_transition_model_' + repr(action) +'.h5'
        pick_model, full_model = keras_functional_transition_model_pick_action()
        pick_model.load_weights(transition_model_name)
        frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in full_model.outputs])
        tf.train.write_graph(frozen_graph, ".", 'transition_model/full_transition_model_' + repr(action) +'.pb', as_text=False)


def load_transition_model(action,gpuID):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    latent_dim = 2
    transition_model_name = 'transition_model/vae_transition_model_' + repr(action) +'.h5'
    vae, encoder, decoder = keras_functional_transition_model(latent_dim)
    #vae1 = tf.keras.models.load_model(transition_model_name)
    #vae.set_weights(vae1.get_weights())
    vae.load_weights(transition_model_name)
    #print vae.get_weights();
    #print "Printing decoder weights aaaaaaaaaaaaaaaaaaaaaaaaaaa"
    #print decoder.get_weights();
    #X = np.array([[0.33,0.08,0.47,0.08,0.01, 1.1,1.5,0]])
    X = np.array([[0.387915, 0.091614, 0.601582, 0.0546314, -1.04977531092, -0.00016737, -0.000296831, 1.0]])
    #R = np.array([[0.9,0.5]])#K.random_normal(shape=(1, latent_dim))#np.array([[0.9,0.5]])
    R = np.array([np.random.normal(0,1,latent_dim)])
    Y = decoder.predict([R,X])
    print Y

def save_keras_model_transition_graph(graph_file):
    K.set_learning_phase(0)
    #vae, encoder, decoder = keras_functional_transition_model(2)
    #json_string = decoder.to_json()

    #vae.load_weights(graph_file)
    decoder = tf.keras.models.load_model('decoder_transition_model.h5')
    print(decoder.outputs)
    print(decoder.inputs)
    frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in decoder.outputs])
    tf.train.write_graph(frozen_graph, ".", "decoder_transition_model.pb", as_text=False)
    #decoder.save('decoder_transition_model.h5')
    #X = np.array([[0.33,0.08,0.47,0.08,0.01, 1.1,1.5,0,0]])
    X = np.array([[0.387915, 0.091614, 0.561582, 0.0946314, -1.04977531092, -0.00016737, -0.000296831, 1.0]])
    R = np.array([[0.9,0.5]])
    Y = decoder.predict([R,X])
    print Y


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
    help_ = "Train transition or observation"
    parser.add_argument("-t",
                        "--train",
                        help=help_)
    help_ = "Action name"
    parser.add_argument("-a",
                        "--action",
                        help=help_)
    help_ = "GPU ID"
    parser.add_argument("-g",
                        "--gpuID",
                        help=help_)
    args = parser.parse_args()
    #save_model()
    object_id_mapping_file = '../ObjectNameToIdMapping.yaml'
    data_dir = '../data_low_friction_table_exp_wider_object_workspace_ver8/data_for_regression'
    object_id_mapping = yaml.load(file(object_id_mapping_file, 'r'))
    object_name_list = object_id_mapping.keys()
    #dataProcessor.get_training_data(object_name_list, data_dir, object_id_mapping_file)
    if(args.train == 'o'):
        train_observation_model(int(args.action), object_name_list, data_dir, object_id_mapping_file, args.gpuID)
    if(args.train == 't'):
        train_transition_model(int(args.action), object_name_list, data_dir, object_id_mapping_file, args.gpuID)

    if(args.train == 'lo'): #load_observation_model
        load_observation_model(int(args.action),args.gpuID)
    if(args.train == 'lt'):
        load_transition_model(int(args.action),args.gpuID)
    if(args.train == 'so'): #save observation model
        save_final_observation_model(int(args.action),args.gpuID)
    if(args.train == 'st'): #save transition model
        save_transiton_model_with_reward_graph(int(args.action),args.gpuID)
    if(args.train == 'sov1'): #save observation model
        save_final_observation_model(int(args.action),args.gpuID,True)
    if(args.train == 'stv1'): #save transition model
        save_transiton_model_with_reward_graph_v1(int(args.action),args.gpuID)
    #keras_functional_transition_model(2)
    #print "Creating observation model"
    #keras_function_observation_model(2)
    #save_keras_model_transition_graph('vae_transition_model.h5')
