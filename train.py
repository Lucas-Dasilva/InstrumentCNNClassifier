"""
Lucas Da Silva & Courtney Cadenhead
date: 4/21/2022
Description: Training methods for training our model
Code Credit: Main ideas taken from seth814 github page
"""
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from setup import get_features, get_labels, DataParser
from model import Mel2Conv, Mel5Conv, Mel10Conv
from predict import make_prediction
import matplotlib.pyplot as plt
import numpy as np
import os

def get_model_summary(n_classes, sampling_rate, delta_time, model):
  X = get_features()
  y = get_labels()
  if model == 5:
    model = Mel5Conv(n_classes, sampling_rate, delta_time)
  elif model == 2:
    model = Mel2Conv(n_classes, sampling_rate, delta_time)
  else:
    model = Mel10Conv(n_classes, sampling_rate, delta_time)
  model.build((1,16000))
  model.summary()
  
def ConvModelFit(n_classes, sampling_rate, delta_time, model, total_epoch):
  X = get_features()
  y = get_labels()
  if model == 5:
    model = Mel5Conv(n_classes, sampling_rate, delta_time)
    csv_path = os.path.join('logs', '{}_history.csv'.format("Mel5Conv"))
  elif model == 2:
    csv_path = os.path.join('logs', '{}_history.csv'.format("Mel2Conv"))
    model = Mel2Conv(n_classes, sampling_rate, delta_time)
  else:
    csv_path = os.path.join('logs', '{}_history.csv'.format("Mel10Conv"))
    model = Mel10Conv(n_classes, sampling_rate, delta_time)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

  train_data = DataParser(X_train, y_train, sampling_rate, delta_time, n_classes, batch_size)
  val_data = DataParser(X_val, y_val, sampling_rate, delta_time, n_classes, batch_size)
  test_data = DataParser(X_test, y_test, sampling_rate, delta_time, n_classes, batch_size)

  csv_logger = CSVLogger(csv_path, append=False)
  model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
  history = model.fit(train_data, epochs=total_epoch, verbose=1, validation_data=val_data,
            callbacks=[csv_logger])
  
  return test_data, model, history

def train(sampling_rate, delta_time, batch_size, n_classes, epochs, model):
  test_data, model, model_history = ConvModelFit(n_classes, sampling_rate, delta_time, model, epochs)

  # Run on Testing data
  results = model.evaluate(test_data) 
  print(f"Melspectrogram Model On test Data:\nLoss:{results[0]} Accuracy:{results[1]}")

  # Plot validation/training data loss/accuracy
  loss_train = model_history.history['accuracy']
  loss_test = model_history.history['val_accuracy']
  epochs = range(1,epochs+1)
  plt.plot(epochs, loss_train, 'b', label='Training Accuracy')
  plt.plot(epochs, loss_test, 'r', label='Validation Accuracy')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.show()
  return model
  

if __name__ == '__main__':
  os.makedirs("logs", exist_ok=True)
  sampling_rate = 16000
  delta_time = 1
  batch_size = 32
  n_classes = 10
  epochs = 30
  model = 5 # Options == 2, 5, 10
  # sub = Mel2Conv(n_classes, sampling_rate, delta_time)
  # sub.summary()
  # sub = Mel5Conv(n_classes, sampling_rate, delta_time)
  # sub.summary()
  # sub = Mel10Conv(n_classes, sampling_rate, delta_time)
  # sub.summary()

  model = train(sampling_rate, delta_time, batch_size, n_classes, epochs, model)
  make_prediction(model, sampling_rate, delta_time)

