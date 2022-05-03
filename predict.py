from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
from enveloping import envelope, downsample_mono
import os


def make_prediction(model, sampling_rate, delta_time):
  """
  Makes prediction on all the 300 files, based on the model weights from train.py
  """
  wav_paths = glob('{}/**'.format("Resources"), recursive=True)
  wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
  classes = sorted(os.listdir("Resources"))
  labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
  le = LabelEncoder()
  y_true = le.fit_transform(labels)
  results = []
  total=len(wav_paths)
  for z, wav_fn in enumerate(wav_paths):
    rate, wav = downsample_mono(wav_fn, sampling_rate)
    mask, env = envelope(wav, rate, threshold=20)
    clean_wav = wav[mask]
    step = int(sampling_rate * delta_time)
    batch = []

    for i in range(0, clean_wav.shape[0], step):
      sample = clean_wav[i:i+step]
      sample = sample.reshape(-1, 1)
      if sample.shape[0] < step:
        tmp = np.zeros(shape=(step, 1), dtype=np.float32)
        tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
        sample = tmp
      batch.append(sample)
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = model.predict(X_batch)
    y_mean = np.mean(y_pred, axis=0)
    y_pred = np.argmax(y_mean)
    real_class = os.path.dirname(wav_fn).split('/')[-1]
    print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
          

  # np.save(os.path.join('logs', args.pred_fn), np.array(results))