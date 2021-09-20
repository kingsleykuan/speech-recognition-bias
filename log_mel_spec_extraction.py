# -*- coding: utf-8 -*-
"""log_mel_spec_extraction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wJtTxFb_tzsFgQowGTfAukZwBddsoSDl
"""

import pandas as pd
import numpy as np
import itertools
from scipy.io import wavfile
import librosa
import librosa.display
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


input_path = 'Data/AudioWAV/'
output_path = 'Data/Mel_Spec/'
file_names = os.listdir(input_path)
print(f'{len(file_names)} audio files')

def get_log_melspectrogram(y, sr, n_fft, hop_length):
  spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
  mel_spect = librosa.power_to_db(spect, ref=np.max)
  return mel_spect

def plot_log_melspectrogram(mel_spect):
  librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
  plt.title('Log Mel Spectrogram');
  plt.colorbar(format='%+2.0f dB');

for index, audio_filename in enumerate(file_names):
  print(f'audio index: {index} and file: {audio_filename}')
  y, sr = librosa.load(input_path + audio_filename)

  # extract log mel spectrogram
  mel_spect = get_log_melspectrogram(y, sr, n_fft=2048, hop_length=512)
  
  print(f'saving log-mel spectrogram with {len(mel_spect)} frames and {len(mel_spect[0])} mel frequency bins...')
  
  # save log mel spectrogram to pickle 
  pickle.dump(mel_spect, open( output_path + audio_filename[:-3] + 'pickle', 'wb'))



