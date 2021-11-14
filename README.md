# Do Machines Discriminate? Examining the Hidden Gender and Racial Bias in Speech Emotion Recognition AI

A IS4152 / IS5452: Affective Computing Project

## Setup
Environment can be setup using Conda:
```
conda env create -f environment.yml
conda activate ser-bias
```

## Dataset
Dataset is available at: https://github.com/CheyneyComputerScience/CREMA-D

Please download and place in the `Data` folder.

## Code Demo
Pre-trained model files are located in `models.zip`

Predictions for models and commercial models are located in `predictions.zip`

Sample data from CREMA-D dataset is located in `sample_data`

Demo of speech emotion recognition can be run with:
```
python -m ser_evaluation.ser_demo \
--model_path 'models/observed/cnn_lstm_attention_multitask' \
--audio_path 'sample_data/1001_DFA_ANG_XX.wav'
```

## Preprocess Dataset
Split Dataset
```
python -m ser_preprocess.split_dataset \
--data_path 'Data/AudioWAV' \
--demographics_csv_path 'Data/VideoDemographics.csv' \
--train_size 0.8 \
--val_size 0.1 \
--train_val_test_split_path 'ser_data/train_val_test_split.json'
```

Preprocess Log Mel Spectrograms
```
python -m ser_preprocess.log_mel_spec_extraction
```

Preprocess SVM Features
```
python -m ser_preprocess.preprocess_svm_data \
--data_path 'Data/AudioWAVSplit/train' \
--ratings_path 'Data/processedResults/summaryTable.csv' \
--output_path 'Data/svm_features/train.p'

python -m ser_preprocess.preprocess_svm_data \
--data_path 'Data/AudioWAVSplit/val' \
--ratings_path 'Data/processedResults/summaryTable.csv' \
--output_path 'Data/svm_features/val.p'

python -m ser_preprocess.preprocess_svm_data \
--data_path 'Data/AudioWAVSplit/test' \
--ratings_path 'Data/processedResults/summaryTable.csv' \
--output_path 'Data/svm_features/test.p'
```

## Train 2D CNN LSTM Model
Sample configuration for different models can be found in `ser_model/config/`

Please change config in `ser_train/train_cnn_lstm_model.py`. Important options include:
- `bootstrap_sampling`
- `use_ratings`
- `use_gender_label`
- `use_race_label`
```
python -m ser_train.train_cnn_lstm_model
```

## Train SVM
python -m ser_train.train_svm

## Speech Emotion Recognition Demo
This assumes that models trained on intended labels are saved in `models/intended/`
```
python -m ser_evaluation.ser_demo \
--model_path 'models/intended/cnn_lstm_attention_multitask' \
--audio_path 'Data/AudioWAV/1001_DFA_ANG_XX.wav'
```

## Classify Emotions
This assumes that models trained on intended labels are saved in `models/intended/`
```
python -m ser_evaluation.classify_emotions \
--model_path 'models/intended/cnn_lstm' \
--output_path 'predictions/intended/cnn_lstm.csv'
```

```
python -m ser_evaluation.classify_emotions \
--model_path 'models/intended/cnn_lstm_attention' \
--output_path 'predictions/intended/cnn_lstm_attention.csv'
```

```
python -m ser_evaluation.classify_emotions \
--model_path 'models/intended/cnn_lstm_attention_multitask' \
--output_path 'predictions/intended/cnn_lstm_attention_multitask.csv' \
--predict_gender \
--predict_race
```

## Classify Emotions with Bootstrap Sampling
This assumes that bootstrap sampling was used and models trained on intended labels are saved in `models_bootstrap/intended/`
```
python -m ser_evaluation.classify_emotions \
--data_path 'Data/MelSpecSplit/train' \
--model_path 'models_bootstrap/intended/cnn_lstm/cnn_lstm' \
--output_path 'predictions_bootstrap/intended/cnn_lstm/cnn_lstm.csv' \
--bootstrap_sampling \
--num_bootstrap_samples 100
```

```
python -m ser_evaluation.classify_emotions \
--data_path 'Data/MelSpecSplit/train' \
--model_path 'models_bootstrap/intended/cnn_lstm_attention/cnn_lstm_attention' \
--output_path 'predictions_bootstrap/intended/cnn_lstm_attention/cnn_lstm_attention.csv' \
--bootstrap_sampling \
--num_bootstrap_samples 100
```

```
python -m ser_evaluation.classify_emotions \
--data_path 'Data/MelSpecSplit/train' \
--model_path 'models_bootstrap/intended/cnn_lstm_attention_multitask/cnn_lstm_attention_multitask' \
--output_path 'predictions_bootstrap/intended/cnn_lstm_attention_multitask/cnn_lstm_attention_multitask.csv' \
--predict_gender \
--predict_race \
--bootstrap_sampling \
--num_bootstrap_samples 100
```

## Evaluate Predictions
This assumes that all bootstrapped predictions for models, intended, and observed are located at: `predictions_bootstrap/intended` and `predictions_bootstrap/observed`. This means that the above commands must be run multiple times for all models, for both annotation types.

Commercial model predictions should be located in `predictions/commercial_results`
```
python -m ser_evaluation.evaluate_user
python -m ser_evaluation.evaluate_commercial
```

## Web Demo
Check WEB application here: [web](https://github.com/wwongwk/speech-recognition-bias/tree/main/web), Have fun!

## Authors

* **Kingsley Kuan** - [kingsleykuan](https://github.com/kingsleykuan)
* **Wong Wei Kang** - [wwongwk](https://github.com/wwongwk)
* **Jeremy Sim** - [jemerysim](https://github.com/jemerysim)
* **Shao Guoxin** - [gxshao](https://github.com/gxshao)
* **Saja Alamoudi** - [saja-alamoudi](https://github.com/saja-alamoudi)
