"""
Preprocess Commercial .json results
"""
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
import numpy as np
import os
import pandas as pd
from pathlib import Path
from crema_metadata import read_actor_demographics, read_ratings
import json
import pandas as pd

def preprocess_vokaturi(path):
    data_path = Path(path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    index = []
    anger = []
    happy = []
    neutral = []
    sad = []
    
    # Vokaturi
    for path in paths:
        data = json.load(open(path))
        ind = data['index']
        preds = data['vokaturi']
        print(preds)
        
        index.append(ind)
        if 'msg' in preds.keys():
            anger.append('null')
            happy.append('null')
            neutral.append('null')
            sad.append('null')
        else:
            anger.append(preds['Angry'])
            happy.append(preds['Happy'])
            neutral.append(preds['Neutral'])
            sad.append(preds['Sad'])
    
    df = pd.DataFrame(data=zip(index,anger,happy,neutral,sad), \
                      columns=['Filename','Anger','Happy', 'Neutral', 'Sad'])
    return df     

def preprocess_empath(path):
    pass
    
def preprocess_da(path):
    pass
