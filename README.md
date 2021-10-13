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
| Annotation Type  | Macro F1-Score |
|------------------|----------------|
| Acted Emotion    | 0.51045        |
| Observed Emotion | 0.44372        |

## Train 2D CNN LSTM Model
```
python -m train_cnn_lstm_model
```

| Annotation Type  | Macro F1-Score |
|------------------|----------------|
| Acted Emotion    | 0.64529        |
| Observed Emotion | 0.50023        |
