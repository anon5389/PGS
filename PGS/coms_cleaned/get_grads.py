import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np
import os


def get_data(dataset_name, discrete=False):
    if discrete:
        with open('discrete_data/'+ dataset_name +'.pickle', 'rb') as f:
            dataset = pickle.load(f)
    else:
        with open('continuous_data/'+ dataset_name +'.pickle', 'rb') as f:
            dataset = pickle.load(f)    
        
    
    return dataset["x"], dataset["y"]

def normalize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_std = np.where(x_std == 0.0, 1.0, x_std)
    # print("mean: ", x_mean)
    # print("std: ", x_std)
    return (x - x_mean) / x_std

if __name__ == '__main__':   
    discrete = False
    if discrete:
        dataset_names = ['TFBind8', 'GFP', 'UTR']
    else:
        dataset_names = ["AntMorphology", "DKittyMorphology", "HopperController", "Superconductor"]
   
    dataset_ = dataset_names[1]
    if discrete:
        x, y = get_data(dataset_ + "_32", discrete=discrete)
    else:
        x, y = get_data(dataset_, discrete=discrete)
        x = normalize(x)
        y = normalize(y) 
    model = keras.models.load_model('ant_model')
    x_in = tf.convert_to_tensor(x[:3].reshape(3,60), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x_in)
        y = model(x_in)
        y = tf.reduce_sum(y)
        
    print(y)
    dy_dx = tape.gradient(y, x_in)
    print(dy_dx.numpy())
    print(dy_dx.numpy().shape)