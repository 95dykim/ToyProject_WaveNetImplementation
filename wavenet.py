import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

os.environ["TFDS_DATA_DIR"] = "./cache/"

import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd

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

def load_dataset(split="train", hz=16000):
    if split.lower() not in ["train", "validation", "test"]:
        print("INVALID SPLIT")
        return
    
    os.makedirs("dataset/groove/", exist_ok=True)
    
    # Original Dataset
    if os.path.exists("dataset/groove/dataset_" + split + "_16000hz_xy.pickle"):
        with open("dataset/groove/dataset_" + split + "_16000hz_xy.pickle", "rb") as f:
            list_aud, list_label = pickle.load(f)
    else:
        ds_tf = tfds.load('groove/full-16000hz', split=split)
        
        list_aud = []
        list_label = []
        
        for elem in ds_tf:
            list_aud.append( elem['audio'].numpy() )
            list_label.append( elem['style']['primary'].numpy() )
                
        with open("dataset/groove/dataset_" + split + "_16000hz_xy.pickle", "wb") as f:
            pickle.dump([list_aud, list_label], f)
        
    # check if resampling is required
    if hz == 16000:
        return list_aud, list_label
    else:
        if os.path.exists("dataset/groove/dataset_" + split + "_" + str(hz) + "hz_xy.pickle"):
            with open("dataset/groove/dataset_" + split + "_" + str(hz) + "hz_xy.pickle", "rb") as f:
                list_aud, list_label = pickle.load(f)
            return list_aud, list_label
        else:
            list_aud = [ librosa.resample(aud, orig_sr=16000, target_sr=hz) for aud in list_aud ]
            with open("dataset/groove/dataset_" + split + "_" + str(hz) + "hz_xy.pickle", "wb") as f:
                pickle.dump([list_aud, list_label], f)
            return list_aud, list_label
            
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

list_aud_train, list_label_train = load_dataset(split="train", hz=16000)
list_aud_valid, list_label_valid = load_dataset(split="validation", hz=16000)

def remove_short( list_aud ):
    list_return = []
    for aud in list_aud:
        if len(aud) < GLOBAL_INPUT_LENGTH + OUT_SIZE:
            continue
        list_return.append( aud )
        
    return list_return

list_aud_train = remove_short(list_aud_train)
list_aud_valid = remove_short(list_aud_valid)

train_ds = DataSet(list_aud_train).repeat(10).shuffle(512).batch(128)
valid_ds = DataSet(list_aud_valid).batch(128)

model = WaveNet()
loss = tf.keras.losses.CategoricalCrossentropy()
optim = tf.keras.optimizers.Adam()
model.compile(loss=loss, optimizer=optim)

model.summary()

checkpoint_filepath = './checkpoints/g0_checkpoint-{epoch}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_filepath, save_weights_only=True, save_best_only=True)

history = model.fit(train_ds, epochs=10000, validation_data=valid_ds, callbacks=[model_checkpoint_callback])#, LR_Scheduler] )

pd.DataFrame( history.history ).to_csv("g0.csv", index=False)
