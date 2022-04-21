"""
Created by: Lucas Da Silva
date: 4/21/2022
Description: This file is for creating and mapping the training into tensor objects, that are to be used in the model.
Code: Data Parser Class taken but modded from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import numpy as np
import keras
from scipy.io import wavfile
from keras.utils import to_categorical

class DataParser(keras.utils.Sequence):
  'Generates data for Keras'
  def __init__(self, wav_files, labels, batch_size=32,
                n_classes=10, shuffle=True, sampling_rate=16000, delta_time=1, n_channels=1):
    'Initialization'
    self.batch_size = batch_size
    self.labels = labels
    self.wav_files = wav_files
    self.delta_time = delta_time
    self.sampling_rate = sampling_rate
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.wav_files) / self.batch_size))

  def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of wav file paths
    wav_files_temp = [self.wav_files[k] for k in indexes]
    labels = [self.labels[k] for k in indexes]
    
    # Generate data
    X, y = self.__data_generation(wav_files_temp, labels)

    return X, y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.wav_files))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, wav_files, labels):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, self.n_channels, int(self.delta_time*self.sampling_rate)), dtype=np.int16)
    y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

    # Generate data
    for i, (file, label) in enumerate(zip(wav_files,labels)):
      # Store sample
      rate, wav = wavfile.read(file)
      X[i,] = wav.reshape(1, -1)
      
      # Store class
      y[i] = to_categorical(label, num_classes=self.n_channels)

    return X, y