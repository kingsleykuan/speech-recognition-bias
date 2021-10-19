import optuna
import torch
from torch.utils.data import DataLoader

from crema_data import CremaAudioDataset
from ser_trainer import SpeechEmotionRecognitionTrainer
from simple_model import SimpleModel

CONFIG = {
    'data_path_train': 'Data/MelSpecSplit/train',
    'data_path_val': 'Data/MelSpecSplit/val',
    'demographics_csv_path': 'Data/VideoDemographics.csv',
    'ratings_csv_path': 'Data/processedResults/summaryTable.csv',

    'save_path': 'models/simple',
    'log_dir': 'runs/simple',

    'use_ratings': False,
    'num_epochs': 100,
    'steps_per_log': 50,
    'epochs_per_eval': 5,
    'batch_size': 64,
    'num_workers': 4,
    'learning_rate': 5e-5,
    'weight_decay': 1e-5,

    'hidden_size': 2048,
    'dropout_rate': 0.7,
    'label_smoothing': 0,

    'random_seed': 0,
}


def load_data(
        data_path,
        demographics_csv_path,
        ratings_csv_path,
        batch_size=32,
        num_workers=4,
        shuffle=True):
    dataset = CremaAudioDataset(
        data_path, demographics_csv_path, ratings_csv_path)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)
    return data_loader


def load_model(hidden_size=2048, dropout_rate=0.8, label_smoothing=0.0):
    model = SimpleModel(
        128 * 130,
        hidden_size,
        6,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing)
    return model


def train(
        data_path_train,
        data_path_val,
        demographics_csv_path,
        ratings_csv_path,
        save_path,
        log_dir=None,
        use_ratings=False,
        num_epochs=20,
        steps_per_log=100,
        epochs_per_eval=5,
        batch_size=32,
        num_workers=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        hidden_size=2048,
        dropout_rate=0.8,
        label_smoothing=0.0,
        random_seed=0):
    torch.manual_seed(random_seed)

    data_loader_train = load_data(
        data_path_train,
        demographics_csv_path,
        ratings_csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)

    data_loader_eval = load_data(
        data_path_val,
        demographics_csv_path,
        ratings_csv_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    model = load_model(
        hidden_size=hidden_size,
        dropout_rate=dropout_rate,
        label_smoothing=label_smoothing)

    trainer = SpeechEmotionRecognitionTrainer(
        data_loader_train,
        data_loader_eval,
        model,
        num_epochs=num_epochs,
        steps_per_log=steps_per_log,
        epochs_per_eval=epochs_per_eval,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        log_dir=log_dir,
        save_path=save_path,
        use_ratings=use_ratings)

    trainer.train()
    return trainer.best_f1_score


def tune_hyperparameters(**config):
    def objective(trial):
        config['num_epochs'] = 100
        config['epochs_per_eval'] = 10

        config['learning_rate'] = trial.suggest_float(
            'learning_rate', 1e-5, 1e-2, log=True)
        config['weight_decay'] = trial.suggest_float(
            'weight_decay', 1e-5, 1e-2, log=True)
        config['hidden_size'] = trial.suggest_int(
            'hidden_size', 32, 4096, log=True)
        config['dropout_rate'] = trial.suggest_float(
            'dropout_rate', 0.0, 1.0)
        config['label_smoothing'] = trial.suggest_float(
            'label_smoothing', 0.0, 0.5)

        return train(**config)

    def print_status(study, trial):
        print(f"Trial Params:\n {trial.params}")
        print(f"Trial Value:\n {trial.value}")
        print(f"Best Params:\n {study.best_params}")
        print(f"Best Value:\n {study.best_value}")

    study = optuna.create_study(study_name='simple', direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


if __name__ == '__main__':
    # tune_hyperparameters(**CONFIG)
    train(**CONFIG)
