import argparse
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from AudioLibrary.AudioFeatures import AudioFeatures
from AudioLibrary.AudioSignal import AudioSignal
from ser_data.crema_metadata import read_metadata, read_ratings

# Sample rate (44.1 kHz)
SAMPLE_RATE = 44100

EMOTIONS = {
    'A': 0,
    'D': 1,
    'F': 2,
    'H': 3,
    'N': 4,
    'S': 5,
}


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess SVM data.")

    parser.add_argument(
        '--data_path', default='Data/AudioWAVSplit/test', type=str,
        help='Path to audio wav files.')

    parser.add_argument(
        '--ratings_path', default='Data/processedResults/summaryTable.csv',
        type=str, help='Path ratings CSV file.')

    parser.add_argument(
        '--output_path', default='Data/svm_features/test.p', type=str,
        help='Path to output SVM features.')

    return parser


# Audio features extraction function
def global_feature_statistics(
        y,
        win_size=0.025,
        win_step=0.01,
        nb_mfcc=12,
        mel_filter=40,
        stats=[
            'mean',
            'std',
            'med',
            'kurt',
            'skew',
            'q1',
            'q99',
            'min',
            'max',
            'range'
        ],
        features_list=[
            'zcr',
            'energy',
            'energy_entropy',
            'spectral_centroid',
            'spectral_spread',
            'spectral_entropy',
            'spectral_flux',
            'sprectral_rolloff',
            'mfcc'
        ]):
    # Extract features
    audio_features = AudioFeatures(y, win_size, win_step)
    features, features_names = audio_features.global_feature_extraction(
        stats=stats,
        features_list=features_list,
        nb_mfcc=nb_mfcc,
        nb_filter=mel_filter)
    return features


def main(data_path, ratings_path, output_path):
    print("Import Data: START")

    metadata = read_metadata(data_path)
    ratings = read_ratings(ratings_path)

    metadata = metadata.filter(items=['path', 'filename', 'emotion'])
    ratings = ratings.filter(items=['filename', 'voice_vote'])

    metadata = metadata.merge(ratings, how='inner', on='filename', sort=True)

    signals = []
    for path in tqdm(metadata['path']):
        # Read audio file
        signals.append(AudioSignal(SAMPLE_RATE, filename=path))

    print("Import Data: END")
    print("Number of audio files imported: {}".format(len(signals)))

    print("Feature extraction: START")
    features = [global_feature_statistics(signal) for signal in tqdm(signals)]
    print("Feature extraction: END!")

    # Create label vectors
    intended_label_vectors = np.zeros((len(metadata), len(EMOTIONS)))
    for i, label in enumerate(metadata['emotion']):
        label_id = EMOTIONS[label]
        intended_label_vectors[i, label_id] = 1

    observed_label_vectors = np.zeros((len(metadata), len(EMOTIONS)))
    for i, observed_label in enumerate(metadata['voice_vote']):
        for label in observed_label.split(':'):
            label_id = EMOTIONS[label]
            observed_label_vectors[i, label_id] = 1

    # Save DataFrame to pickle
    filenames = metadata['filename'].tolist()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as file:
        pickle.dump(
            {
                'filenames': filenames,
                'features': features,
                'intended_label_vectors': intended_label_vectors,
                'observed_label_vectors': observed_label_vectors
            },
            file)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    main(args.data_path, args.ratings_path, args.output_path)
