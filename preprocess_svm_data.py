import csv
import itertools
import os
import pickle
from glob import glob

import numpy as np

from AudioLibrary import AudioFeatures, AudioSignal

csv_path = "/content/gdrive/MyDrive/IS4152/Data/processedResults/summaryTable.csv"
name = open(csv_path , 'r')
file = csv.DictReader(name)

filename_voicevote = []

for col in file:
    filename = col['FileName'] + '.wav'
    voicevote = col['VoiceVote']
    filename_voicevote.append([filename, voicevote])

file_path = '/content/gdrive/MyDrive/IS4152/Data/AudioWAVSplit/train/'
files = set(os.listdir(file_path))

filename_voicevote = [(filename, voicevote) for (filename, voicevote) in filename_voicevote if filename in files]

print(filename_voicevote)

# Start feature extraction
print("Import Data: START")

# Sample rate (44.1 kHz)
sample_rate = 44100

# Initialize signal and labels list
signal = []
labels = []

# Compute global statistics features for all audio file
for audio_index, (audio_file, voicevote) in enumerate(filename_voicevote):
    # Read audio file
    signal.append(AudioSignal(sample_rate, filename=file_path + audio_file))
        
    # Set label
    labels.append(voicevote)

    # Print running...
    if (audio_index % 100 == 0):
        print("Import Data: RUNNING ... {} files".format(audio_index))

# Cast labels to array
labels = np.asarray(labels).ravel()

# Stop feature extraction
print("Import Data: END \n")
print("Number of audio files imported: {}".format(labels.shape[0]))

# Audio features extraction function
def global_feature_statistics(y, win_size=0.025, win_step=0.01, nb_mfcc=12, mel_filter=40,
                             stats = ['mean', 'std', 'med', 'kurt', 'skew', 'q1', 'q99', 'min', 'max', 'range'],
                             features_list =  ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy', 'spectral_flux', 'sprectral_rolloff', 'mfcc']):
    
    # Extract features
    audio_features = AudioFeatures(y, win_size, win_step)
    features, features_names = audio_features.global_feature_extraction(stats=stats, features_list=features_list)
    return features
    
# Features extraction parameters
sample_rate = 16000 # Sample rate (16.0 kHz)
win_size = 0.025    # Short term window size (25 msec)
win_step = 0.01     # Short term window step (10 msec)
nb_mfcc = 12        # Number of MFCCs coefficients (12)
nb_filter = 40      # Number of filter banks (40)
stats = ['mean', 'std', 'med', 'kurt', 'skew', 'q1', 'q99', 'min', 'max', 'range'] # Global statistics
features_list =  ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', # Audio features
                      'spectral_entropy', 'spectral_flux', 'sprectral_rolloff', 'mfcc']

print(global_feature_statistics(signal[0]))

# Start feature extraction
print("Feature extraction: START")

# Compute global feature statistics for all audio file
features = np.asarray(list(map(global_feature_statistics, signal)))

# Stop feature extraction
print("Feature extraction: END!")

# Anger (ANG) 1
# Disgust (DIS) 2
# Fear (FEA) 3
# Happy/Joy (HAP) 4
# Neutral (NEU) 5 
# Sad (SAD) 6

# Happy 4
# Neutral 5

# label = 'H:N'
# labels = [0, 0, 0, 1, 1, 0]

print(labels)

emotions = {
    
    'A': 0,
    'D': 1,
    'F': 2,
    'H': 3,
    'N': 4,
    'S': 5,

}

label_vectors = []
for label in labels:
    label_ids = np.zeros(len(emotions))
    for l in label.split(':'):
        label_id = emotions[l]
        label_ids[label_id] = 1
    label_vectors.append(label_ids)

label_vectors = np.asarray(label_vectors)
print(label_vectors)

# Save DataFrame to pickle
pickle.dump([features, label_vectors], open("/content/gdrive/My Drive/IS4152/Data/SVM_features/observe_preprocessing_train.p", 'wb'))
