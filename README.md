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
Macro F1-Score (Acted Emotion): 0.51045
Macro F1-Score (Observed Emotion): 0.44372
