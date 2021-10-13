import torch
from torch.utils.data import DataLoader

from crema_data import CremaAudioDataset
from ser_trainer import SpeechEmotionRecognitionTrainer
from simple_model import SimpleModel

data_path_train = 'Data/MelSpecSplit/train'
data_path_val = 'Data/MelSpecSplit/val'
demographics_csv_path = 'Data/VideoDemographics.csv'
ratings_csv_path = 'Data/processedResults/summaryTable.csv'
save_path = 'models/simple'
log_dir = 'runs/simple'
use_ratings = False
num_epochs = 50
steps_per_log = 100
epochs_per_eval = 5
batch_size = 32
num_workers = 4
learning_rate = 1e-3
weight_decay = 1e-4
random_seed = 0


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


def load_model():
    model = SimpleModel(
        128 * 130, 2048, 6, dropout_rate=0.8, label_smoothing=0.0)
    return model


def main(
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

    model = load_model()

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


if __name__ == '__main__':
    main(
        data_path_train,
        data_path_val,
        demographics_csv_path,
        ratings_csv_path,
        save_path,
        log_dir=log_dir,
        use_ratings=use_ratings,
        num_epochs=num_epochs,
        steps_per_log=steps_per_log,
        epochs_per_eval=epochs_per_eval,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        random_seed=random_seed)
