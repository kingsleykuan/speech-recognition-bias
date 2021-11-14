import pickle

import numpy as np
from sklearn.utils import resample
from torch.utils.data import Dataset

from ser_data.crema_metadata import (
    read_actor_demographics, read_metadata, read_ratings)

EMOTIONS = (
    'Anger',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
)

GENDERS = {
    'Male',
    'Female',
}

RACES = {
    'Caucasian',
    'African American',
    'Asian',
    'Unknown',
}


class CremaAudioDataset(Dataset):
    def __init__(
            self,
            data_path,
            demographics_csv_path,
            ratings_csv_path,
            bootstrap_sampling=False,
            out_of_bootstrap=False,
            random_seed=0):
        metadata = read_metadata(data_path)
        actor_demographics = read_actor_demographics(demographics_csv_path)
        ratings = read_ratings(ratings_csv_path)

        metadata = metadata.merge(
            actor_demographics,
            how='inner',
            on='actor_id',
            validate='many_to_one')
        metadata = metadata.merge(
            ratings,
            how='inner',
            on='filename',
            validate='one_to_one')

        metadata.sort_values('filename', inplace=True, kind='mergesort')

        if bootstrap_sampling:
            bootstrap_metadata = resample(
                metadata,
                replace=True,
                n_samples=len(metadata),
                random_state=random_seed)

            if out_of_bootstrap:
                metadata = metadata[
                    ~metadata['filename'].isin(bootstrap_metadata['filename'])]
            else:
                metadata = bootstrap_metadata

        data = []
        for path in metadata['path']:
            with open(path, 'rb') as file:
                data.append(pickle.load(file))

        self.data = data
        self.metadata = metadata

        self.emotions = EMOTIONS
        self.genders = GENDERS
        self.races = RACES

        self.emotion_to_id = {
            emotion[0]: id for id, emotion in enumerate(self.emotions)}
        self.gender_to_id = {
            gender: id for id, gender in enumerate(self.genders)
        }
        self.race_to_id = {
            race: 1 for id, race in enumerate(self.races)
        }
        self.race_to_id['Caucasian'] = 0

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        log_mel_spec = self.data[idx]
        metadata = self.metadata.iloc[idx]

        filename = metadata['filename']

        emotion = metadata['emotion']
        emotion_label = np.zeros(len(self.emotion_to_id), dtype=np.float32)
        emotion_label[self.emotion_to_id[emotion]] = 1

        emotion_ratings = metadata['voice_vote']
        emotion_ratings = emotion_ratings.split(':')
        emotion_rating_labels = np.zeros(
            len(self.emotion_to_id), dtype=np.float32)
        for emotion_rating in emotion_ratings:
            emotion_rating_labels[self.emotion_to_id[emotion_rating]] = 1

        gender = metadata['sex']
        gender_label = np.asarray(self.gender_to_id[gender], dtype=np.float32)

        race = metadata['race']
        race_label = np.asarray(self.race_to_id[race], dtype=np.float32)

        data = {
            'filename': filename,
            'log_mel_spec': log_mel_spec,
            'emotion_label': emotion_label,
            'emotion_rating_labels': emotion_rating_labels,
            'gender_label': gender_label,
            'race_label': race_label,
        }

        return data
