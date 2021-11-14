import os
import pickle
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

input_path = Path('Data/AudioWAVSplit/')
output_path = Path('Data/MelSpecSplit/')
file_names = list(input_path.glob('**/*.wav'))
print(f'{len(file_names)} audio files')


def get_log_melspectrogram(y, sr, n_fft, hop_length):
    spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_spect = librosa.power_to_db(spect, ref=np.max)
    return mel_spect


def plot_log_melspectrogram(mel_spect):
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
    plt.title('Log Mel Spectrogram');
    plt.colorbar(format='%+2.0f dB');


for index, audio_path in enumerate(file_names):
    print(f'audio index: {index} and file: {audio_path}')
    y, sr = librosa.load(str(audio_path), duration=3)
    print(f'sampling rate: {sr} and file length: {np.size(y)}')

    fixed_length = 66150  # Add the fix length you want
    y = librosa.util.fix_length(y, fixed_length)
    print(f'sampling rate: {sr} and file length: {np.size(y)}')

    # extract log mel spectrogram
    mel_spect = get_log_melspectrogram(y, sr, n_fft=2048, hop_length=512)
    print(f'saving log-mel spectrogram with {len(mel_spect)} mel frequency bins and {len(mel_spect[0])} frames...')
    print(" ")

    audio_path = output_path / audio_path.parent.name / f'{audio_path.stem}.pickle'
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # save log mel spectrogram to pickle
    pickle.dump(mel_spect, open(audio_path, 'wb'))
