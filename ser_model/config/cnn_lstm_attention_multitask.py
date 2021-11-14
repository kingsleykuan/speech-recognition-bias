CONFIG = {
    'data_path_train': 'Data/MelSpecSplit/train',
    'data_path_val': 'Data/MelSpecSplit/val',
    'demographics_csv_path': 'Data/VideoDemographics.csv',
    'ratings_csv_path': 'Data/processedResults/summaryTable.csv',

    'save_path': 'models/cnn_lstm_attention_multitask',
    'log_dir': 'runs/cnn_lstm_attention_multitask',

    'bootstrap_sampling': False,
    'use_ratings': False,
    'use_gender_label': True,
    'use_race_label': True,

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
