#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from shutil import copy2

from sklearn.model_selection import train_test_split

from crema_metadata import read_actor_demographics, read_metadata


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Split CREMA-D dataset into train/val/test sets.")

    parser.add_argument(
        '--data_path', default='Data/AudioWAV', type=str,
        help='Path to audio wav files.')

    parser.add_argument(
        '--demographics_csv_path', default='Data/VideoDemographics.csv',
        type=str, help='Path actor demographics CSV.')

    parser.add_argument(
        '--train_size', default=0.8,
        type=float, help='Ratio of data to use for training.')

    parser.add_argument(
        '--val_size', default=0.1,
        type=float, help='Ratio of data to use for validation.')

    parser.add_argument(
        '--train_val_test_split_path', default='train_val_test_split.json',
        type=str, help="""Path to train val test split JSON file.
                          Will write to file if it does not exist.""")

    return parser


def split_dataset(
        data_path,
        demographics_csv_path,
        train_size,
        val_size,
        train_val_test_split_path):
    # Read actor demographics from CSV file
    actor_demographics = read_actor_demographics(demographics_csv_path)
    actor_demographics['sex_race'] = \
        actor_demographics['sex'] + ' ' + actor_demographics['race']

    # Read data metadata from data directory
    data_path = Path(data_path)
    dataset = read_metadata(data_path)
    dataset = dataset.merge(
        actor_demographics, how='inner', on='actor_id', validate='many_to_one')

    # Read train val test split from file if it exists
    # Otherwise perform stratified split and write split to file
    train_val_test_split_path = Path(train_val_test_split_path)
    if train_val_test_split_path.exists():
        print(f"Read train val test split: {train_val_test_split_path}\n")
        with open(train_val_test_split_path) as file:
            train_val_test_split = json.load(file)

        train_filenames = train_val_test_split['train']
        val_filenames = train_val_test_split['val']
        test_filenames = train_val_test_split['test']

        train = dataset[dataset['filename'].isin(set(train_filenames))]
        val = dataset[dataset['filename'].isin(set(val_filenames))]
        test = dataset[dataset['filename'].isin(set(test_filenames))]
    else:
        train, val_test = train_test_split(
            dataset,
            train_size=train_size,
            random_state=0,
            shuffle=True,
            stratify=dataset['sex_race'])

        val, test = train_test_split(
            val_test,
            train_size=val_size / (1.0 - train_size),
            random_state=0,
            shuffle=True,
            stratify=val_test['sex_race'])

        train_filenames = sorted(train['filename'].tolist())
        val_filenames = sorted(val['filename'].tolist())
        test_filenames = sorted(test['filename'].tolist())

        train_val_test_split = {
            'train': train_filenames,
            'val': val_filenames,
            'test': test_filenames,
        }

        with open(train_val_test_split_path, 'w') as file:
            json.dump(train_val_test_split, file, indent=4)
        print(f"Wrote train val test split: {train_val_test_split_path}\n")

    # Print dataset statistics
    print("Dataset Demographics Statistics\n{}\n{}\n".format(
        len(dataset), dataset['sex_race'].value_counts()))

    print("Train Demographics Statistics\n{}\n{}\n".format(
        len(train), train['sex_race'].value_counts()))

    print("Validation Demographics Statistics\n{}\n{}\n".format(
        len(val), val['sex_race'].value_counts()))

    print("Test Demographics Statistics\n{}\n{}\n".format(
        len(test), test['sex_race'].value_counts()))

    # Copy data files according to split
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    test_path = data_path / 'test'

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    for path in train['path']:
        copy2(path, train_path)
    for path in val['path']:
        copy2(path, val_path)
    for path in test['path']:
        copy2(path, test_path)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    split_dataset(
        args.data_path,
        args.demographics_csv_path,
        args.train_size,
        args.val_size,
        args.train_val_test_split_path)
