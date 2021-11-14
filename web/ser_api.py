import sys
sys.path.append('../')
import torch
import librosa
import json
import io
import os
import copy
from ser_model.cnn_lstm_model import CNNLSTM2DModel
import audio_handler as ah
import numpy as np
import soundfile as sf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMOTIONS = (
    'Anger',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
)
GENDERS = (
    'Male',
    'Female',
)
RACES = (
    'Caucasian',
    'Non-Caucasian',
)

FIXED_AUDIO_LENGTH = 3
N_FFT = 2048
HOP_LENGTH = 512

## Load models
class SerModel:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.model = CNNLSTM2DModel.load(path, map_location=device)
        self.model = self.model.to(device)
        self.model = self.model.eval()
    
    def preprocess_audio(self,
        audio,
        sampling_rate,
        fixed_audio_length,
        n_fft,
        hop_length):
        audio = librosa.util.fix_length(audio, fixed_audio_length * sampling_rate)
        melspectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length)
        log_melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
        return log_melspectrogram


    def predict_emotions(self,
        audio,
        sampling_rate,
        fixed_audio_length=3,
        n_fft=2048,
        hop_length=512):
        log_melspectrogram = self.preprocess_audio(
            audio, sampling_rate, fixed_audio_length, n_fft, hop_length)

        log_melspectrogram = torch.from_numpy(log_melspectrogram).float()
        log_melspectrogram = log_melspectrogram.to(device)
        log_melspectrogram = torch.unsqueeze(log_melspectrogram, dim=0)

        with torch.no_grad():
            outputs = self.model(log_melspectrogram)
            logits = torch.squeeze(outputs['logits'], dim=0)
            predictions = torch.sigmoid(logits)
            predictions = predictions.cpu().numpy().tolist()

        labels = list(EMOTIONS)
        if len(predictions) > len(labels):
            predictions.insert(-2, 1 - predictions[-2])
            predictions.insert(-1, 1 - predictions[-1])
            labels.extend(GENDERS)
            labels.extend(RACES)

        predictions = {
            label: prediction for label, prediction in zip(labels, predictions)}
        return predictions

# Entrance

config = {}
with open(os.getcwd() + '/config.json') as f:
    config = json.load(f)

path_list = config['model_path']
model_list = []

for obj in path_list:
    for name, path in obj.items():
        print("Loading model: ", name," Path:" ,path)
        sermodel = SerModel(name, path)
        model_list.append(sermodel)


def predict_emotions(audio, ar):
    result = {}
    # Convert audio to wav
    audio = ah.get_numpy_array_from_ogg(io.BytesIO(audio), ar)
    audio = np.array(audio).astype(float)
    for model in model_list:
        try:
            result[model.name] = model.predict_emotions(copy.deepcopy(audio), ar)
        except:
            pass
    return result
