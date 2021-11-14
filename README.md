# Speech Emotion Recognition Bias

## Setup
Environment can be setup using Conda:
```
conda env create -f environment.yml
conda activate ser-bias
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

## Train Simple Model
```
python -m train_simple_model
```

## Train 2D CNN LSTM Model
```
python -m train_cnn_lstm_model
```

## Models Trained on Acted Emotion Labels (Validation Set)
| Model                        | Macro F1-Score |
|------------------------------|----------------|
| Dense                        | 0.50927        |
| CNN LSTM                     | 0.65964        |
| CNN LSTM Attention           | 0.66651        |
| CNN LSTM Attention Multitask | 0.68006        |

## Models Trained on Observed Emotion Labels (Validation Set)
| Model                        | Macro F1-Score |
|------------------------------|----------------|
| Dense                        | 0.45105        |
| CNN LSTM                     | 0.50542        |
| CNN LSTM Attention           | 0.51589        |
| CNN LSTM Attention Multitask | 0.51678        |

## Classify Emotions
```
python -m classify_emotions \
--model_path 'models/acted/cnn_lstm' \
--output_path 'predictions/acted/cnn_lstm.csv'
```

```
python -m classify_emotions \
--model_path 'models/acted/cnn_lstm_attention' \
--output_path 'predictions/acted/cnn_lstm_attention.csv'
```

```
python -m classify_emotions \
--model_path 'models/acted/cnn_lstm_attention_multitask' \
--output_path 'predictions/acted/cnn_lstm_attention_multitask.csv' \
--predict_gender \
--predict_race
```

## Speech Emotion Recognition Demo
```
python -m ser_demo \
--model_path 'models/acted/cnn_lstm_attention_multitask' \
--audio_path 'Data/AudioWAV/1001_DFA_ANG_XX.wav'
```

## Classify Emotions with Bootstrap Sampling
```
python -m classify_emotions \
--data_path 'Data/MelSpecSplit/train' \
--model_path 'models/intended/cnn_lstm' \
--output_path 'predictions/intended/cnn_lstm.csv' \
--bootstrap_sampling \
--num_bootstrap_samples 100
```

```
python -m classify_emotions \
--data_path 'Data/MelSpecSplit/train' \
--model_path 'models/intended/cnn_lstm_attention' \
--output_path 'predictions/intended/cnn_lstm_attention.csv' \
--bootstrap_sampling \
--num_bootstrap_samples 100
```

```
python -m classify_emotions \
--data_path 'Data/MelSpecSplit/train' \
--model_path 'models/intended/cnn_lstm_attention_multitask' \
--output_path 'predictions/intended/cnn_lstm_attention_multitask.csv' \
--predict_gender \
--predict_race \
--bootstrap_sampling \
--num_bootstrap_samples 100
```


## Authors

* **Kingsley Kuan** - [kingsleykuan](https://github.com/kingsleykuan)
* **Wong Wei Kang** - [wwongwk](https://github.com/wwongwk)
* **Jeremy Sim** - [jemerysim](https://github.com/jemerysim)
* **Shao Guoxin** - [gxshao](https://github.com/gxshao)
* **Saja Alamoudi** - [saja-alamoudi](https://github.com/saja-alamoudi)
