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

GLOBAL_INPUT_LENGTH = 4410
USE_BIAS = True

"""
spoken_digit (https://www.tensorflow.org/datasets/catalog/spoken_digit)
A free audio dataset of spoken digits. Think MNIST for audio.
A simple audio/speech dataset consisting of recordings of spoken digits in wav files at 8kHz. The recordings are trimmed so that they have near minimal silence at the beginnings and ends.
"""
def load_dataset():
    if os.path.exists("dataset/gtzan/dataset_xyf.pickle"):
        with open("dataset/gtzan/dataset_xyf.pickle", "rb") as f:
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
                    
                    list_aud.append(aud)
                    list_label.append(label)
                    list_fname.append(fname)
                except:
                    print("FAIL - {}".format(fname))
                
        with open("dataset/gtzan/dataset_xyf.pickle", "wb") as f:
            pickle.dump([list_aud, list_label, list_fname], f)
        
        return list_aud, list_label, list_fname
            
################################################
#TODO - CHECK IF OFFICIAL PARAMETER VALUES EXIST
################################################
def WaveNetBlock_NonConditional(x, channel_size, name, kernel_size = 2, dilation_rate = 1):
    x_1a = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, activation="tanh", dilation_rate = dilation_rate, name=name+"_conv_tanh")(x)
    x_1b = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, activation="sigmoid", dilation_rate = dilation_rate, name=name+"_conv_sigmoid")(x)
    
    x_1 = tf.keras.layers.Multiply(name=name+"_mult")([x_1a, x_1b])
    x_1 = tf.keras.layers.Conv1D(channel_size, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_1x1")(x_1)
    
    #residual connection
    #ReLU applied after skip-connections, according to the paper
    return tf.keras.layers.Add(name=name+"_residual")([x_1, x]), x_1
    
def WaveNet(input_length = None, channel_size = 32, num_layers = 32, dilation_limit=128, max_n=256):
    inputs = tf.keras.Input(shape=(input_length, 1), name="inputs")
    x = inputs
    
    #Initial Conv
    x = tf.keras.layers.Conv1D(channel_size, 2, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = 2, name="conv_initial")(x)
    #NOTE - Activation?
    
    #WaveNet ResidualBlocks
    list_skip = []
    dilation_rate = 1
    for idx in range(num_layers):
        x, x_skip = WaveNetBlock_NonConditional(x, channel_size, "WaveNetBlock_" + str(idx), dilation_rate = dilation_rate)
        
        dilation_rate = 1 if dilation_rate >= dilation_limit else dilation_rate*2
        list_skip.append(x_skip)
    
    #Output layers
    x = tf.keras.layers.Add(name = "SkipConnections")(list_skip)
    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_1")(x)
    x = tf.keras.layers.Conv1D(channel_size, 1, strides=1, padding="same", use_bias=USE_BIAS, name="SkipConnections_conv_1")(x)
    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_2")(x)
    x = tf.keras.layers.Conv1D(max_n, 1, strides=1, padding="same", use_bias=USE_BIAS, name="SkipConnections_conv_2")(x)

    #x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    
    outputs = tf.keras.layers.Softmax(name= "outputs")(x[:,-1])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='WaveNet')

########################################################################################################

#A function to quantize an audio to a list of labels
def quantize_aud(x, max_n = 256):
    quantized = np.sign( x ) * np.log( 1 + ( max_n - 1 ) * abs(x) ) / np.log( max_n )
    quantized = ( quantized + 1 ) / 2
    quantized = np.digitize( quantized, np.arange(max_n) / (max_n-1) ) - 1
    return quantized

#A function to dequantize a list of predictions
def dequantize_aud(x, max_n = 256):
    dequantized = x / (max_n - 1)
    dequantized = dequantized * 2 - 1
    dequantized = (np.exp( dequantized * np.log(max_n) / np.sign(dequantized) ) - 1) / (max_n - 1) * np.sign(dequantized)
    return dequantized

########################################################################################################

# A function to quantize a list of audios to dataset_x, dataset_y
def convert_to_dataset( list_aud, input_length = GLOBAL_INPUT_LENGTH, win_stride=int(GLOBAL_INPUT_LENGTH/2) ):
    dataset_x = []
    dataset_y = []
    
    for aud in list_aud:
        for div in range( 1 + max(0, int( np.ceil( ( len(aud) - input_length + 1 ) / win_stride ) )) ):
            aud_div = aud[ div*win_stride : div*win_stride + input_length + 1 ]
            aud_pad = np.zeros(input_length+1)
            aud_pad[:len(aud_div)] = aud_div

            dataset_x.append( aud_pad[:-1] )
            dataset_y.append( quantize_aud(aud_pad[-1:]) )

    return np.stack(dataset_x), np.stack(dataset_y)

########################################################################################################

class DataSeq_Train(tf.keras.utils.Sequence):
    def __init__(self, list_aud, input_length = GLOBAL_INPUT_LENGTH):
        self.x = [ ]
        for aud in list_aud:
            if len(aud) >= (input_length + 1):
                self.x.append(aud)
            else:
                aud_pad = np.zeros(input_length + 1)
                aud_pad[:len(aud)] = aud
                self.x.append(aud_pad)
        
        self.y = [ quantize_aud(x[1:]) for x in self.x ]
        self.x = [ x[:-1] for x in self.x ]
        
        self.input_length = input_length

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        window_start = np.random.randint( max( len(self.x[idx]) - self.input_length, 0 ) + 1 )
        #return self.x[idx][window_start:window_start+self.input_length], self.y[idx][window_start:window_start+self.input_length]
        return self.x[idx][window_start:window_start+self.input_length], self.y[idx][window_start:window_start+self.input_length][-1:]
        
def DataSet_Train(list_aud, input_length = GLOBAL_INPUT_LENGTH):
    dataseq = DataSeq_Train(list_aud, input_length)
    ds = tf.data.Dataset.from_generator( lambda: (x for x in dataseq),
        output_signature=(
            tf.TensorSpec(shape=(input_length, ), dtype=tf.float32),
            tf.TensorSpec(shape=(1, ), dtype=tf.float32),
        )
    )
    ds = ds
    return ds

########################################################################################################

class DataSeq_Valid(tf.keras.utils.Sequence):
    def __init__(self, list_aud, input_length = GLOBAL_INPUT_LENGTH):
        self.x, self.y = convert_to_dataset(list_aud, input_length = input_length, win_stride = 500)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
def DataSet_Valid(list_aud, input_length = GLOBAL_INPUT_LENGTH):
    dataseq = DataSeq_Valid(list_aud, input_length)
    ds = tf.data.Dataset.from_generator( lambda: (x for x in dataseq),
        output_signature=(
            tf.TensorSpec(shape=(input_length, ), dtype=tf.float32),
            tf.TensorSpec(shape=(1, ), dtype=tf.float32),
        )
    )
    ds = ds
    return ds
