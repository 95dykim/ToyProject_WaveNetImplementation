import os
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

os.environ["TFDS_DATA_DIR"] = "./cache/"

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

import shutil
import librosa
import pickle
import glob

USE_BIAS = True
NUM_BLOCKS = 4
DILATION_LIMIT = 10

QUANT_B = 128

OUT_SIZE = 16
GLOBAL_INPUT_LENGTH = 2**DILATION_LIMIT * NUM_BLOCKS + OUT_SIZE + 1 #add 1 due to initial conv

"""
spoken_digit (https://www.tensorflow.org/datasets/catalog/spoken_digit)
A free audio dataset of spoken digits. Think MNIST for audio.
A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz. The recordings are trimmed so that they have near minimal silence at the beginnings and ends.
"""
def load_dataset():
    if os.path.exists("dataset/gtzan/dataset_8000hz_xyf.pickle"):
        with open("dataset/gtzan/dataset_8000hz_xyf.pickle", "rb") as f:
            list_aud, list_label, list_fname = pickle.load(f)
            
        return list_aud, list_label, list_fname
    else:
        list_aud = []
        list_label = []
        list_fname = []
        
        for fpath_genre in glob.glob("dataset/gtzan/genres_original/*"):
            label = fpath_genre.split("/")[-1]
            
            for fpath in glob.glob(fpath_genre + "/*.wav") :
                try:
                    fname = fpath.split("/")[-1]
                    aud = librosa.load(fpath, sr=22050)[0]
                    
                    list_aud.append( librosa.resample(aud, orig_sr=22050, target_sr=8000) )
                    list_label.append(label)
                    list_fname.append(fname)
                except:
                    print("FAIL - {}".format(fname))
                
        with open("dataset/gtzan/dataset_8000hz_xyf.pickle", "wb") as f:
            pickle.dump([list_aud, list_label, list_fname], f)
        
        return list_aud, list_label, list_fname
            
################################################
#TODO - CHECK IF OFFICIAL PARAMETER VALUES EXIST
################################################
def WaveNetBlock_NonConditional(x, channel_size, skip_channel, name, kernel_size = 2, dilation_rate = 1):
    x_1a = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, activation="tanh", dilation_rate = dilation_rate, name=name+"_conv_tanh")(x)
    x_1b = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, activation="sigmoid", dilation_rate = dilation_rate, name=name+"_conv_sigmoid")(x)
    
    x_1 = tf.keras.layers.Multiply(name=name+"_mult")([x_1a, x_1b])
    x_1 = tf.keras.layers.Conv1D(channel_size, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_1x1")(x_1)
    
    x_1_skip = tf.keras.layers.Conv1D(skip_channel, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_skip")(x_1)

    return tf.keras.layers.Add(name=name+"_residual")([x_1, x]), x_1_skip

def WaveNet(input_length = None, channel_size = 32, num_blocks = NUM_BLOCKS, dilation_limit=DILATION_LIMIT, skip_channel =64, max_n=QUANT_B, out_size = OUT_SIZE):
    inputs = tf.keras.Input(shape=(input_length, 1), name="inputs")
    x = inputs
    
    #Initial Conv
    x = tf.keras.layers.Conv1D(channel_size, 2, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = 2, name="conv_initial")(x)
    
    #WaveNet ResidualBlocks
    list_skip = []
    for idx_i in range(num_blocks):
        for idx_j in range(dilation_limit):
            x, x_skip = WaveNetBlock_NonConditional(x, channel_size, skip_channel, "WaveNetBlock_{}_{}".format(idx_i, idx_j), dilation_rate = 2**(idx_j))

            list_skip.append(x_skip)
    
    #Output layers
    x = tf.keras.layers.Add(name = "SkipConnections")(list_skip)

    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_1")(x)
    x = tf.keras.layers.Conv1D(skip_channel, 1, strides=1, padding="same", use_bias=True, name="SkipConnections_conv_1")(x)

    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_2")(x)
    x = tf.keras.layers.Conv1D(max_n, 1, strides=1, padding="same", use_bias=True, name="SkipConnections_conv_2")(x)

    #outputs = x[:,-out_size:]
    
    outputs = tf.keras.layers.Softmax(name= "outputs")(x[:,-out_size:])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='WaveNet')

########################################################################################################

#A function to quantize an audio to a list of labels
def quantize_aud(x, max_n = QUANT_B):
    quantized = np.sign( x ) * np.log( 1 + ( max_n - 1 ) * abs(x) ) / np.log( max_n )
    quantized = ( quantized + 1 ) / 2
    quantized = np.digitize( quantized, np.arange(max_n) / (max_n-1) ) - 1
    return quantized

#A function to dequantize a list of predictions
def dequantize_aud(x, max_n = QUANT_B):
    dequantized = x / (max_n - 1)
    dequantized = dequantized * 2 - 1
    dequantized = (np.exp( dequantized * np.log(max_n) / np.sign(dequantized) ) - 1) / (max_n - 1) * np.sign(dequantized)
    return dequantized

########################################################################################################

class DataSeq(tf.keras.utils.Sequence):
    def __init__(self, list_aud, input_length = GLOBAL_INPUT_LENGTH, out_size = OUT_SIZE):
        self.x = [ ]
        for aud in list_aud:
            aud_q = quantize_aud(aud).astype(int)
            
            if len(aud_q) >= (input_length + out_size):
                self.x.append(aud_q)
            else:
                aud_pad = np.zeros(input_length + out_size)
                aud_pad[:len(aud_q)] = aud_q
                self.x.append(aud_pad)
        
        self.input_length = input_length
        self.out_size = out_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        window_start = np.random.randint( max( len(self.x[idx]) - self.input_length - self.out_size, 0 ) + 1 )

        return_x = dequantize_aud( self.x[idx][window_start:window_start+self.input_length] )
        return_y = self.x[idx][window_start+self.input_length:window_start+self.input_length+self.out_size]
        
        return return_x, tf.one_hot(return_y, QUANT_B)
        
def DataSet(list_aud, input_length = GLOBAL_INPUT_LENGTH, out_size = OUT_SIZE):
    dataseq = DataSeq(list_aud, input_length)
    ds = tf.data.Dataset.from_generator( lambda: (x for x in dataseq),
        output_signature=(
            tf.TensorSpec(shape=(input_length, ), dtype=tf.float32),
            tf.TensorSpec(shape=(out_size, QUANT_B), dtype=tf.int32),
        )
    )
    ds = ds
    return ds
