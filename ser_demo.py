import argparse

import librosa
import numpy as np
import torch

from cnn_lstm_model import CNNLSTM2DModel

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


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Classify emotions using speech emotion recognition.")

    parser.add_argument(
        '--model_path', default='models/intended/cnn_lstm', type=str,
        help='Path to saved model.')

    parser.add_argument(
        '--audio_path', type=str, required=True,
        help='Path to audio file.')

    return parser


def load_model(model_path):
    model = CNNLSTM2DModel.load(model_path, map_location=device)
    model = model.to(device)
    model = model.eval()
    return model


def load_audio(audio_path, max_length=None):
    audio, sampling_rate = librosa.load(audio_path, duration=max_length)
    return audio, sampling_rate


def preprocess_audio(
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


def predict_emotions(
        model,
        audio,
        sampling_rate,
        fixed_audio_length=3,
        n_fft=2048,
        hop_length=512):
    log_melspectrogram = preprocess_audio(
        audio, sampling_rate, fixed_audio_length, n_fft, hop_length)

    log_melspectrogram = torch.from_numpy(log_melspectrogram).float()
    log_melspectrogram = log_melspectrogram.to(device)
    log_melspectrogram = torch.unsqueeze(log_melspectrogram, dim=0)

    with torch.no_grad():
        outputs = model(log_melspectrogram)
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


def main(
        model_path,
        audio_path,
        fixed_audio_length=3,
        n_fft=2048,
        hop_length=512):
    model = load_model(model_path)

    audio, sampling_rate = load_audio(
        audio_path, max_length=fixed_audio_length)

    predictions = predict_emotions(
        model,
        audio,
        sampling_rate,
        fixed_audio_length=fixed_audio_length,
        n_fft=n_fft,
        hop_length=hop_length)
    print(predictions)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(
        args.model_path,
        args.audio_path,
        fixed_audio_length=FIXED_AUDIO_LENGTH,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH)
