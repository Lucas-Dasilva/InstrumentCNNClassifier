"""
Lucas Da Silva & Courtney Cadenhead
date: 4/21/2022
Description: This file is for creating and mapping the training into tensor objects, that are to be used in the model.
Code: Data Parser Class taken but modded from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import numpy as np
import keras
from scipy.io import wavfile
from keras.utils.np_utils import to_categorical
from glob import glob
from sklearn.preprocessing import LabelEncoder
import os

def get_features():
  """
  Get all the features or wav_paths from the clean directory
  """
  wav_paths = glob('{}/**'.format("clean"), recursive=True)
  wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
  return wav_paths
  
def get_labels():
  """
  Get all the labels from the clean directory
  """
  label_encoder = LabelEncoder()
  wav_paths = glob('{}/**'.format("clean"), recursive=True)
  wav_paths = [x.replace(os.sep, '/') for x in wav_paths if '.wav' in x]
  classes = sorted(os.listdir("clean"))
  label_encoder.fit(classes)
  labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
  labels = label_encoder.transform(labels)
  return labels


class DataParser(keras.utils.all_utils.Sequence):
  'Generates data for Keras'
  def __init__(self, wav_files, labels, sampling_rate, delta_time, n_classes,
                batch_size, shuffle=True, n_channels=1):
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
    X = np.empty((self.batch_size, int(self.delta_time*self.sampling_rate), self.n_channels), dtype=np.int16)
    y = np.empty((self.batch_size, self.n_classes), dtype=np.float32)

    # Generate data
    for i, (file, label) in enumerate(zip(wav_files,labels)):
      # Store sample
      rate, wav = wavfile.read(file)
      X[i,] = wav.reshape(-1, 1)
      
      # Store class
      y[i] = to_categorical(label, num_classes=self.n_classes)

    return X, y