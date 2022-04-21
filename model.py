"""
Created by: Lucas Da Silva
Date: 4/21/2022
Description: Create a class to store the model parameters and setup the model layers
"""
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizer_v2.adam import Adam
from keras.metrics import accuracy
from kapre.composed import get_melspectrogram_layer
from keras.layers import TimeDistributed, LayerNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.regularizers import L2


class Conv2D(Model):
  """A Convolutional 2D model"""
  def __init__(self, n_classes, sampling_rate, delta_time):
    super(Conv2D, self).__init__()
    self.input_shape = (int(sampling_rate*delta_time), 1)
    self.loss_object = categorical_crossentropy()
    self.optimizer = Adam()
    self.metric = accuracy()
    self.mel = get_melspectrogram_layer(input_shape=self.input_shape, n_mels=128,
                                  input_data_format="channels_last", output_data_format="channels_last",
                                  pad_end=True, n_fft=512, win_length=400,
                                  hop_length=160, sample_rate=sampling_rate,
                                  return_decibel=True)
    self.norm = LayerNormalization(axis=2)
    self.max_pool = MaxPooling2D(pool_size=(2,2), padding='same')
    self.conv_tahn = Conv2D(activation='tanh', padding='same')
    self.conv_relu = Conv2D(activation='relu', padding='same')
    self.flat = Flatten()
    self.drop = Dropout()
    self.dense = Dense(activation='relu', activity_regularizer=L2(0.001))
    self.soft = Dense(n_classes, activation='softmax')
  def call(self, x):
    x = self.mel(x)
    x = self.norm(x)
    x = self.conv_tahn(x)
    x = self.max_pool(x)
    x = self.conv_relu(x)
    x = self.max_pool(x)
    x = self.conv_relu(x)
    x = self.max_pool(x)
    x = self.conv_relu(x)
    x = self.max_pool(x)
    x = self.conv_relu(x)
    x = self.flat(x)
    x = self.drop(x)
    x = self.dense(x)
    return self.soft(x)

   

