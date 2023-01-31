###############################
# WIP - NO TRAININIG / TEST YET
###############################

import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

os.environ["TFDS_DATA_DIR"] = "./cache/"

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

################################################
#TODO - CHECK IF OFFICIAL PARAMETER VALUES EXIST
################################################
def WaveNetBlock_NonConditional(x, channel_size, name, kernel_size = 2, dilation_rate = 1):
    x_1 = x
    x_2 = x
    
    x_1_a = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=False, activation="tanh", name=name+"_conv_tanh")(x_1)
    x_1_b = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=False, activation="sigmoid", name=name+"_conv_sigmoid")(x_1)
    
    x_1 = tf.keras.layers.Multiply(name=name+"_mult")([ x_1_a, x_1_b ])
    x_1 = tf.keras.layers.Conv1D(channel_size, 1, strides=1, padding="same", use_bias=False, name=name+"_conv_1x1")(x_1)
    
    #residual connection
    return tf.keras.layers.Add(name=name+"_residual")([x_1, x_2]), x_1
    
def WaveNet(input_length = 2400, channels = 1, channel_size = 16, num_layers = 16, dilation_limit=128):
    inputs = tf.keras.Input(shape=(input_length, channels), name="inputs")
    x = inputs
    
    list_skip = []
    dilation_rate = 1
    for idx in range(num_layers):
        x, x_skip = WaveNetBlock_NonConditional(x, channel_size, "WaveNetBlock_" + str(idx), dilation_rate = dilation_rate)
        
        dilation_rate = 1 if dilation_rate == dilation_limit else dilation_rate*2
        list_skip.append(x_skip)
    
    x = tf.keras.layers.Add(name = "SkipConnections")(list_skip)
    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_1")(x)
    x = tf.keras.layers.Conv1D(1, 1, strides=1, padding="same", use_bias=False, name="SkipConnections_conv_1")(x)
    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_2")(x)
    x = tf.keras.layers.Conv1D(1, 1, strides=1, padding="same", use_bias=False, name="SkipConnections_conv_2")(x)
    x = tf.keras.layers.Flatten(name = "SkipConnections_flatten")(x)
    outputs = tf.keras.layers.Softmax(name= "outputs")(x)
    
    
    #x = tf.keras.layers.GlobalAveragePooling1D()(x)
    #outputs = tf.keras.layers.Dense(16, name='predictions', activation='relu')(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='WaveNet')

WaveNet().summary()
