"""
Created by: Lucas Da Silva
date: 4/21/2022
Description: Training methods for training our model
Code Credit: Main ideas taken from seth814 github page
"""
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.model_selection import train_test_split
from setup import get_features, get_labels, DataParser
from model import MelConv2D
from keras.losses import categorical_crossentropy
from keras.metrics import accuracy

import os

def train(sampling_rate, delta_time, batch_size, n_classes):
  X = get_features()
  y = get_labels()

  # Generate training and testing samples
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
  dp = DataParser(X_train, y_train, sampling_rate, delta_time, n_classes, batch_size)
  model = MelConv2D(n_classes, sampling_rate, delta_time)
  csv_logger = CSVLogger(csv_path, append=False)
  model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
  model.fit(dp, epochs=30, verbose=1,
            callbacks=[csv_logger])

if __name__ == '__main__':
  os.makedirs("logs", exist_ok=True)
  os.makedirs("models", exist_ok=True)
  csv_path = os.path.join('logs', '{}_history.csv'.format("MelConv2D"))
  train(16000, 1, 32, 10)


