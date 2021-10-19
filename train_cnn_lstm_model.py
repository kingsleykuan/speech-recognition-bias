import optuna
import torch
from torch.utils.data import DataLoader

from crema_data import CremaAudioDataset
from ser_trainer import SpeechEmotionRecognitionTrainer
from cnn_lstm_model import CNNLSTM2DModel

CONFIG = {
    'data_path_train': 'Data/MelSpecSplit/train',
    'data_path_val': 'Data/MelSpecSplit/val',
    'demographics_csv_path': 'Data/VideoDemographics.csv',
    'ratings_csv_path': 'Data/processedResults/summaryTable.csv',

    'save_path': 'models/cnn_lstm_attention',
    'log_dir': 'runs/cnn_lstm_attention',

    'use_ratings': False,
    'use_gender_label': False,
    'use_race_label': False,

    'num_epochs': 100,
    'steps_per_log': 50,
    'epochs_per_eval': 5,
    'batch_size': 64,
    'num_workers': 4,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,

    'lstm_hidden_size': 64,
    'use_self_attention': True,
    'self_attention_size': 128,
    'num_self_attention_heads': 8,
    'dropout_rate': 0.1,
    'label_smoothing': 0.1,

    'random_seed': 0
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


def load_model(
        lstm_hidden_size=128,
        use_self_attention=False,
        self_attention_size=128,
        num_self_attention_heads=8,
        use_gender_label=False,
        use_race_label=False,
        dropout_rate=0.1,
        label_smoothing=0.1):
    output_size = 6
    if use_gender_label:
        output_size += 1
    if use_race_label:
        output_size += 1

    model = CNNLSTM2DModel(
        output_size=output_size,
        lstm_hidden_size=lstm_hidden_size,
        use_self_attention=use_self_attention,
        self_attention_size=self_attention_size,
        num_self_attention_heads=num_self_attention_heads,
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
        use_gender_label=False,
        use_race_label=False,
        num_epochs=20,
        steps_per_log=100,
        epochs_per_eval=5,
        batch_size=32,
        num_workers=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        lstm_hidden_size=128,
        use_self_attention=False,
        self_attention_size=128,
        num_self_attention_heads=8,
        dropout_rate=0.1,
        label_smoothing=0.1,
        random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True

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
        lstm_hidden_size=lstm_hidden_size,
        use_self_attention=use_self_attention,
        self_attention_size=self_attention_size,
        num_self_attention_heads=num_self_attention_heads,
        use_gender_label=use_gender_label,
        use_race_label=use_race_label,
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
        use_ratings=use_ratings,
        use_gender_label=use_gender_label,
        use_race_label=use_race_label)

    trainer.train()
    return trainer.best_f1_score


def tune_hyperparameters(**config):
    def objective(trial):
        config['num_epochs'] = 100
        config['epochs_per_eval'] = 10

        # config['learning_rate'] = trial.suggest_float(
        #     'learning_rate', 1e-5, 1e-2, log=True)
        # config['weight_decay'] = trial.suggest_float(
        #     'weight_decay', 1e-5, 1e-2, log=True)
        config['lstm_hidden_size'] = trial.suggest_int(
            'lstm_hidden_size', 32, 512, log=True)
        config['self_attention_size'] = trial.suggest_int(
            'self_attention_size', 32, 512, log=True)
        # config['dropout_rate'] = trial.suggest_float(
        #     'dropout_rate', 0.0, 0.5)
        # config['label_smoothing'] = trial.suggest_float(
        #     'label_smoothing', 0.0, 0.3)

        return train(**config)

    def print_status(study, trial):
        print(f"Trial Params:\n {trial.params}")
        print(f"Trial Value:\n {trial.value}")
        print(f"Best Params:\n {study.best_params}")
        print(f"Best Value:\n {study.best_value}")

    study = optuna.create_study(study_name='cnn_lstm', direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[print_status])
    print(study.best_params)
    print(study.best_value)


if __name__ == '__main__':
    # tune_hyperparameters(**CONFIG)
    train(**CONFIG)
