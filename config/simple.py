CONFIG = {
    'data_path_train': 'Data/MelSpecSplit/train',
    'data_path_val': 'Data/MelSpecSplit/val',
    'demographics_csv_path': 'Data/VideoDemographics.csv',
    'ratings_csv_path': 'Data/processedResults/summaryTable.csv',

    'save_path': 'models/simple',
    'log_dir': 'runs/simple',

    'bootstrap_sampling': False,
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
