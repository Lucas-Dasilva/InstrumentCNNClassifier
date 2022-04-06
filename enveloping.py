import wavfile
import argparse
import os
from glob import glob
import numpy as np
import pandas as pd
from librosa.core import resample, to_mono
from tqdm import tqdm
import ~


def Envelope(signal, rate, threshold):
    """
    Converts audio data in the form of a numpy array to pandas series,
    finds the rolling average and applies the abs value to the signal
    at all points.

    Then we will filter out data below the threshold and fill the array 'mask'. 

    """

    mask = []
    signal = pd.Series(signal).apply(np.abs)
    signal_mean = signal.rolling(window = int(rate/10), min_periods = 1, center = True).mean()

    #Filter out data below the threshold.
    
    for mean in signal_mean:

        if mean > threshold:
            mask.append(True)
            
        else:
            mask.append(False)

    return mask, signal_mean

def FFT(signal,rate):
    """
    Convert the signal from the time domain to the frequency domain

    """
    signal_fft = np.fft.fft(signal)
    spectrum = np.abs(signal_fft)
    frequency = np.linspace(0, rate, len(spectrum))

    left_spec = spectrum[:int(len(spectrum)/2)]
    left_freq = frequency[:int(len(spectrum)/2)]
    
    return left_freq
