from pathlib import Path

import pandas as pd


def read_metadata(data_path):
    data_path = Path(data_path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    filenames = [path.stem for path in paths]
    metadata = [filename.split('_') for filename in filenames]

    metadata = [
        [path] + [filename] + attributes
        for path, filename, attributes in zip(paths, filenames, metadata)]
    metadata = pd.DataFrame(
        metadata,
        columns=(
            'path',
            'filename',
            'actor_id',
            'sentence',
            'emotion',
            'emotion_level',
        ))
    metadata['actor_id'] = pd.to_numeric(metadata['actor_id'])
    metadata['emotion'] = metadata['emotion'].apply(lambda x: x[0])

    return metadata


def read_actor_demographics(demographics_csv_path):
    actor_demographics = pd.read_csv(
        demographics_csv_path,
        sep=',',
        header=0,
        usecols=(
            'ActorID',
            'Age',
            'Sex',
            'Race',
            'Ethnicity',
        ))
    actor_demographics = actor_demographics.rename(
        columns={
            'ActorID': 'actor_id',
            'Age': 'age',
            'Sex': 'sex',
            'Race': 'race',
            'Ethnicity': 'ethnicity',
        })
    return actor_demographics


def read_ratings(ratings_csv_path):
    ratings = pd.read_csv(
        ratings_csv_path,
        sep=',',
        header=0,
        usecols=(
            'FileName',
            'VoiceVote',
            'VoiceLevel',
            'FaceVote',
            'FaceLevel',
            'MultiModalVote',
            'MultiModalLevel',
        ))
    ratings = ratings.rename(
        columns={
            'FileName': 'filename',
            'VoiceVote': 'voice_vote',
            'VoiceLevel': 'voice_level',
            'FaceVote': 'face_vote',
            'FaceLevel': 'face_level',
            'MultiModalVote': 'multimodal_vote',
            'MultiModalLevel': 'multimodal_level',
        })
    return ratings
