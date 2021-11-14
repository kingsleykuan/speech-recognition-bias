#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnn_lstm_model import CNNLSTM2DModel
from crema_data import CremaAudioDataset
from utils import recursive_to_device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Classify emotions using speech emotion recognition.")

    parser.add_argument(
        '--data_path', default='Data/MelSpecSplit/test', type=str,
        help='Path to data files.')

    parser.add_argument(
        '--demographics_csv_path', default='Data/VideoDemographics.csv',
        type=str, help='Path to demographics CSV file.')

    parser.add_argument(
        '--ratings_csv_path', default='Data/processedResults/summaryTable.csv',
        type=str, help='Path to observer rating CSV files.')

    parser.add_argument(
        '--model_path', default='models/intended/cnn_lstm', type=str,
        help='Path to saved model.')

    parser.add_argument(
        '--output_path', default='predictions/intended/cnn_lstm.csv', type=str,
        help='Path to output predictions.')

    parser.add_argument(
        '--predict_gender', action='store_true', default=False,
        help='Whether model predicts gender.')

    parser.add_argument(
        '--predict_race', action='store_true', default=False,
        help='Whether model predicts race.')

    parser.add_argument(
        '--batch_size', default=64, type=int,
        help='Batch size.')

    parser.add_argument(
        '--num_workers', default=4, type=int,
        help='Number of data loader workers.')

    parser.add_argument(
        '--bootstrap_sampling', action='store_true', default=False,
        help='Whether to use bootstrap sampling for validation.')

    parser.add_argument(
        '--num_bootstrap_samples', default=100, type=int,
        help='Number of bootstrap samples.')

    parser.add_argument(
        '--random_seed', default=0, type=int,
        help='Random seed.')

    return parser


def load_data(
        data_path,
        demographics_csv_path,
        ratings_csv_path,
        bootstrap_sampling=False,
        out_of_bootstrap=True,
        batch_size=64,
        num_workers=4,
        random_seed=0):
    dataset = CremaAudioDataset(
        data_path,
        demographics_csv_path,
        ratings_csv_path,
        bootstrap_sampling=bootstrap_sampling,
        out_of_bootstrap=out_of_bootstrap,
        random_seed=random_seed)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)
    return data_loader


def load_model(model_path):
    model = CNNLSTM2DModel.load(model_path, map_location=device)
    return model


def predict_emotions(data_loader, model):
    with torch.no_grad():
        filenames_list = []
        predictions_list = []
        for data in tqdm(data_loader):
            data = recursive_to_device(data, device, non_blocking=True)

            filenames = data['filename']
            filenames_list.extend(filenames)

            log_mel_spec = data['log_mel_spec']
            outputs = model(log_mel_spec)
            predictions = torch.sigmoid(outputs['logits'])
            predictions_list.append(predictions)

        filenames = filenames_list
        predictions = torch.cat(predictions_list, dim=0).cpu().numpy()

    return filenames, predictions


def classify_emotions(
        data_path,
        demographics_csv_path,
        ratings_csv_path,
        model_path,
        output_path,
        predict_gender=False,
        predict_race=False,
        batch_size=64,
        num_workers=4,
        bootstrap_sampling=False,
        random_seed=0,
        **kwargs):
    data_loader = load_data(
        data_path,
        demographics_csv_path,
        ratings_csv_path,
        bootstrap_sampling=bootstrap_sampling,
        out_of_bootstrap=True,
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=random_seed)

    model = load_model(model_path)
    model = model.to(device)
    model = model.eval()

    filenames, predictions = predict_emotions(data_loader, model)

    labels = list(data_loader.dataset.emotions)

    if predict_gender:
        labels.append('Female')
    if predict_race:
        labels.append('Non-Caucasian')

    data = pd.DataFrame(predictions, columns=labels)
    data.insert(0, 'Filename', filenames)

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, float_format='%.5f', header=True, index=False)

    print(data)


def classify_emotions_bootstrap(num_bootstrap_samples, **config):
    model_path = Path(config['model_path'])
    output_path = Path(config['output_path'])

    for i in tqdm(range(num_bootstrap_samples)):
        config['bootstrap_sampling'] = True
        config['model_path'] = model_path.parent / (model_path.name + f'_{i}')
        config['output_path'] = output_path.parent \
            / (output_path.stem + f'_{i}' + output_path.suffix)
        config['random_seed'] = i

        classify_emotions(**config)


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    if args.bootstrap_sampling:
        classify_emotions_bootstrap(**vars(args))
    else:
        classify_emotions(**vars(args))
