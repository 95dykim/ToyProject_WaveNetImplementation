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

NORMALIZE_AUD = False
USE_BIAS = True
NUM_BLOCKS = 4
DILATION_LIMIT = 10

QUANT_B = 64

OUT_SIZE = 16
GLOBAL_INPUT_LENGTH = 2**DILATION_LIMIT * NUM_BLOCKS + OUT_SIZE + 1 #add 1 due to initial conv

LABEL_DEPTH = 0

SAVENAME = "WaveNet_rosenstock_{}_{}_{}_{}".format(NUM_BLOCKS, DILATION_LIMIT, QUANT_B, OUT_SIZE)

########################################################################################################
# TODO
# Add comments...
########################################################################################################

def load_dataset_rosenstock(hz=22050):
    # Original Dataset
    if os.path.exists("./dataset/rosenstock/dataset_22050hz_xy.pickle"):
        with open("./dataset/rosenstock/dataset_22050hz_xy.pickle", "rb") as f:
            list_aud, list_label = pickle.load(f)
    else:
        list_aud = []
        list_label = []
        
        label = -1
        for fpath_parent in glob.glob("./dataset/rosenstock/*"):
            label = label + 1
            
            print(fpath_parent)
            for fpath in glob.glob(fpath_parent + "/*.mp3"):
                try:
                    aud = librosa.load(fpath)[0]
                    
                    list_aud.append(aud)
                    list_label.append(label)
                except:
                    pass
                
        with open("./dataset/rosenstock/dataset_22050hz_xy.pickle", "wb") as f:
            pickle.dump([list_aud, list_label], f)
        
    # check if resampling is required
    if hz == 22050:
        return list_aud, list_label
    else:
        if os.path.exists("./dataset/rosenstock/dataset_" + str(hz) + "hz_xy.pickle"):
            with open("./dataset/rosenstock/dataset_" + str(hz) + "hz_xy.pickle", "rb") as f:
                list_aud, list_label = pickle.load(f)
            return list_aud, list_label
        else:
            list_aud = [ librosa.resample(aud, orig_sr=22050, target_sr=hz) for aud in list_aud ]
            with open("./dataset/rosenstock/dataset_" + str(hz) + "hz_xy.pickle", "wb") as f:
                pickle.dump([list_aud, list_label], f)
            return list_aud, list_label

def load_dataset_groove(split="train", hz=16000):
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
    x_1a = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = dilation_rate, name=name+"_A_conv")(x)
    x_1a = tf.keras.layers.Activation("tanh", name = name+"_A_tanh")(x_1a)
    x_1b = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = dilation_rate, name=name+"_B_conv")(x)
    x_1b = tf.keras.layers.Activation("sigmoid", name = name+"_B_sigmoid")(x_1b)
    
    x_1 = tf.keras.layers.Multiply(name=name+"_mult")([x_1a, x_1b])
    x_1 = tf.keras.layers.Conv1D(channel_size, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_out")(x_1)
    
    x_1_skip = tf.keras.layers.Conv1D(skip_channel, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_skip")(x_1)    
    x = tf.keras.layers.Add(name=name+"_residual")([x_1, x])
    
    return x, x_1_skip

def WaveNetBlock_GlobalConditional(x, gc, input_length, channel_size, skip_channel, name, kernel_size = 2, dilation_rate = 1):
    x_1a = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = dilation_rate, name=name+"_A_conv")(x)
    x_1a_gc = tf.keras.layers.Dense(input_length, use_bias=False, name=name+"_A_gc")(gc)
    x_1a_gc = tf.keras.layers.Reshape((-1,1), name = name+"_A_gc_reshape")(x_1a_gc)
    x_1a = tf.keras.layers.Add(name = name+"_A_add")([x_1a, x_1a_gc])
    x_1a = tf.keras.layers.Activation("tanh", name = name+"_A_tanh")(x_1a)
    
    x_1b = tf.keras.layers.Conv1D(channel_size, kernel_size, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = dilation_rate, name=name+"_B_conv")(x)
    x_1b_gc = tf.keras.layers.Dense(input_length, use_bias=False, name=name+"_B_gc")(gc)
    x_1b_gc = tf.keras.layers.Reshape((-1,1), name = name+"_B_gc_reshape")(x_1b_gc)
    x_1b = tf.keras.layers.Add(name = name+"_B_add")([x_1b, x_1b_gc])
    x_1b = tf.keras.layers.Activation("sigmoid", name = name+"_B_sigmoid")(x_1b)
    
    x_1 = tf.keras.layers.Multiply(name=name+"_mult")([x_1a, x_1b])
    x_1 = tf.keras.layers.Conv1D(channel_size, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_out")(x_1)
    
    x_1_skip = tf.keras.layers.Conv1D(skip_channel, 1, strides=1, padding="same", use_bias=USE_BIAS, name=name+"_conv_skip")(x_1)    
    x = tf.keras.layers.Add(name=name+"_residual")([x_1, x])
    
    return x, x_1_skip

def WaveNet(input_length=GLOBAL_INPUT_LENGTH, channel_size=32, num_blocks=NUM_BLOCKS, dilation_limit=DILATION_LIMIT, skip_channel=16, max_n=QUANT_B, out_size=OUT_SIZE, include_softmax=True, global_condition=0):
    inputs = tf.keras.Input(shape=(input_length, 1), name="inputs")
    
    if global_condition:
        gc = tf.keras.Input(shape=(global_condition,), name="input_gc")

    x = inputs
    
    #Initial Conv
    x = tf.keras.layers.Conv1D(channel_size, 2, strides=1, padding="causal", use_bias=USE_BIAS, dilation_rate = 2, name="initial_conv")(x)
    
    #WaveNet ResidualBlocks
    list_skip = []
    for idx_i in range(num_blocks):
        for idx_j in range(dilation_limit):
            #x, x_skip = WaveNetBlock_NonConditional(x, channel_size, skip_channel, "WaveNetBlock_{}_{}".format(idx_i, idx_j), dilation_rate = 2**(idx_j))
            #"""
            if global_condition:
                x, x_skip = WaveNetBlock_GlobalConditional(x, gc, input_length, channel_size, skip_channel, "WaveNetBlock_{}_{}".format(idx_i, idx_j), dilation_rate = 2**(idx_j))
            else:
                x, x_skip = WaveNetBlock_NonConditional(x, channel_size, skip_channel, "WaveNetBlock_{}_{}".format(idx_i, idx_j), dilation_rate = 2**(idx_j))
            #"""

            list_skip.append(x_skip)
    
    #Output layers
    x = tf.keras.layers.Add(name = "SkipConnections")(list_skip)

    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_1")(x)
    x = tf.keras.layers.Conv1D(skip_channel, 1, strides=1, padding="same", use_bias=True, name="out_conv_1")(x)

    x = tf.keras.layers.ReLU(name= "SkipConnections_ReLU_2")(x)
    x = tf.keras.layers.Conv1D(max_n, 1, strides=1, padding="same", use_bias=True, name="out_conv_2")(x)

    if include_softmax:
        outputs = tf.keras.layers.Softmax(name= "outputs")(x[:,-out_size:])
    else:
        outputs = x[:,-out_size:]

    if global_condition:
        return tf.keras.Model(inputs=[inputs, gc], outputs=outputs, name='WaveNet')
    else:
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
    def __init__(self, list_aud, list_gc = None, input_length = GLOBAL_INPUT_LENGTH, out_size = OUT_SIZE):
        self.x = [ ]
        
        if type(list_gc) != type(None):
            max_gc = max(list_gc)+1
            self.gc = [ tf.one_hot(gc, max_gc, dtype=tf.float32) for gc in list_gc ]
        else:
            self.gc = None
        
        for aud in list_aud:
            aud_q = quantize_aud( np.concatenate( (np.zeros(input_length), aud) ) ).astype(int)

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
        
        if type(self.gc) != type(None):
            return (return_x, self.gc[idx]), tf.one_hot(return_y, QUANT_B)
        else:
            return return_x, tf.one_hot(return_y, QUANT_B)
        
def DataSet_Unconditional(list_aud, input_length = GLOBAL_INPUT_LENGTH, out_size = OUT_SIZE):
    dataseq = DataSeq(list_aud, input_length = input_length)
    ds = tf.data.Dataset.from_generator( lambda: (x for x in dataseq),
        output_signature=(
            tf.TensorSpec(shape=(input_length, ), dtype=tf.float32),
            tf.TensorSpec(shape=(out_size, QUANT_B), dtype=tf.int32),
        )
    )
    ds = ds
    return ds

def DataSet_GlobalConditional(list_aud, list_gc, input_length = GLOBAL_INPUT_LENGTH, out_size = OUT_SIZE):
    dataseq = DataSeq(list_aud, list_gc, input_length = input_length)
    ds = tf.data.Dataset.from_generator( lambda: (x for x in dataseq),
        output_signature=(
            ( tf.TensorSpec(shape=(input_length, ), dtype=tf.float32), tf.TensorSpec(shape=(max(list_gc)+1, ), dtype=tf.float32) ),
            tf.TensorSpec(shape=(out_size, QUANT_B), dtype=tf.int32),
        )
    )
    ds = ds
    return ds

list_aud_train, list_label_train = load_dataset_rosenstock(hz=10000)

def remove_short( list_aud, list_label ):
    list_return_aud = []
    list_return_label = []
    for aud, label in zip(list_aud, list_label):
        if len(aud) < GLOBAL_INPUT_LENGTH + OUT_SIZE:
            continue
        list_return_aud.append( (librosa.util.normalize(aud) if NORMALIZE_AUD else aud) )
        list_return_label.append( label )
        
    return list_return_aud, list_return_label
list_aud_train, list_label_train = remove_short(list_aud_train, list_label_train)

list_aud_train = np.asarray(list_aud_train, dtype=object)
list_label_train = np.asarray(list_label_train, dtype=object)

list_idx = np.arange(len(list_aud_train))
np.random.shuffle(list_idx)

train_ds = DataSet_Unconditional(list_aud_train[list_idx[1:]]).repeat(4).shuffle(1024).batch( len(list_idx)-1 )
valid_ds = DataSet_Unconditional(list_aud_train[list_idx[:1]]).batch(1)
        
########################################################################################################

model = WaveNet(global_condition=LABEL_DEPTH)
loss = tf.keras.losses.CategoricalCrossentropy()
optim = tf.keras.optimizers.Adam()
model.compile(loss=loss, optimizer=optim)

class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (epoch+1)%100 == 0:
            self.model.save("./checkpoints/"+SAVENAME+"_checkpoint-{}.h5".format(epoch+1))
customsaver = CustomSaver()

history = model.fit(train_ds, epochs=50000, validation_data=valid_ds, callbacks=[customsaver])#, LR_Scheduler] )

pd.DataFrame( history.history ).to_csv(SAVENAME+".csv", index=False)
