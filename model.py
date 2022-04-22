"""
Created by: Lucas Da Silva
Date: 4/21/2022
Description: Create a class to store the model parameters and setup the model layers
"""
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from kapre.composed import get_melspectrogram_layer
from keras.layers import TimeDistributed, LayerNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.regularizers import L2


class MelConv2D(Model):
  """A Convolutional 2D model"""
  def __init__(self, n_classes, sampling_rate, delta_time):
    super(MelConv2D, self).__init__()
    self.mel = get_melspectrogram_layer(input_shape=(int(sampling_rate*delta_time), 1), n_mels=128,
                                  input_data_format="channels_last", output_data_format="channels_last",
                                  pad_end=True, n_fft=512, win_length=400,
                                  hop_length=160, sample_rate=sampling_rate,
                                  return_decibel=True)
    self.norm = LayerNormalization(axis=2)
    self.max_pool = MaxPooling2D(pool_size=(2,2), padding='same')
    self.conv_tahn = Conv2D(8, 7, activation='tanh', padding='same')
    self.conv_relu1 = Conv2D(16, 5, activation='relu', padding='same')
    # self.conv_relu2 = Conv2D(16, 3, activation='relu', padding='same')
    # self.conv_relu3 = Conv2D(32, 3, activation='relu', padding='same')
    # self.conv_relu4 = Conv2D(32, 3, activation='relu', padding='same')
    self.flat = Flatten()
    self.drop = Dropout(rate=0.2)
    self.dense = Dense(64, activation='relu', activity_regularizer=L2(0.001))
    self.soft = Dense(n_classes, activation='softmax')
  def call(self, x):
    x = self.mel(x)
    x = self.norm(x)
    x = self.conv_tahn(x)
    x = self.max_pool(x)
    x = self.conv_relu1(x)
    x = self.max_pool(x)
    # x = self.conv_relu2(x)
    x = self.max_pool(x)
    # x = self.conv_relu3(x)
    x = self.max_pool(x)
    # x = self.conv_relu4(x)
    x = self.flat(x)
    x = self.drop(x)
    x = self.dense(x)
    return self.soft(x)

   

