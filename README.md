# Speech Emotion Recognition Bias

## Setup
Environment can be setup using Conda:
```
conda env create -f environment.yml
conda activate ser-bias
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
