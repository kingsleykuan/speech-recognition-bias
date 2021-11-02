"""
Preprocess Commercial .json results
"""
from pathlib import Path
import json
import numpy as np
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
            anger.append(np.NaN)
            happy.append(np.NaN)
            neutral.append(np.NaN)
            sad.append(np.NaN)
        else:
            anger.append(preds['Angry'])
            happy.append(preds['Happy'])
            neutral.append(preds['Neutral'])
            sad.append(preds['Sad'])
    
    df = pd.DataFrame(data=zip(index,anger,happy,neutral,sad), \
                      columns=['Filename','Anger','Happy', 'Neutral', 'Sad']).dropna()
    return df     

def preprocess_empath(path):
    data_path = Path(path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    index = []
    anger = []
    happy = []
    neutral = []
    sad = []
    
    # empath
    for path in paths:
        data = json.load(open(path))
        ind = data['index']
        preds = data['empath']
        print(preds)
        
        index.append(ind)
        anger.append(preds['anger'] /50)
        happy.append(preds['joy'] /50)
        neutral.append(preds['calm'] /50)
        sad.append(preds['sorrow'] /50)
    
    df = pd.DataFrame(data=zip(index,anger,happy,neutral,sad), \
                      columns=['Filename','Anger','Happy', 'Neutral', 'Sad'])
    return df    

    
def preprocess_da(path):
    data_path = Path(path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    index = []
    anger = []
    happy = []
    neutral = []
    sad = []
    
    # deepaffect
    for path in paths:
        data = json.load(open(path))
        ind = data['index']
        preds_arr = data['deepaffect']
        print(preds_arr)
        
        index.append(ind)

        if len(preds_arr) > 1:
            timing = []
            emotion = []
            for preds in preds_arr:
                timing.append(preds['end'] - preds['start'])
                emotion.append(preds['emotion'])
            # timing= [3, 0.1]
            # emotion= ['excited', 'neutral']

            max_index =  timing.index(max(timing))
            emotion_pred = emotion[max_index]
        elif len(preds_arr) == 1:
            emotion_pred = preds_arr[0]['emotion']


        if emotion_pred == 'anger' or emotion_pred == 'frustration':
            anger.append(1)
            happy.append(0)
            neutral.append(0)
            sad.append(0)
        if emotion_pred == 'happy' or emotion_pred == 'excited':
            anger.append(0)
            happy.append(1)
            neutral.append(0)
            sad.append(0)
        if emotion_pred == 'neutral':
            anger.append(0)
            happy.append(0)
            neutral.append(1)
            sad.append(0)
        if emotion_pred == 'sad':
            anger.append(0)
            happy.append(0)
            neutral.append(0)
            sad.append(1)
        

    
    df = pd.DataFrame(data=zip(index,anger,happy,neutral,sad), \
                      columns=['Filename','Anger','Happy', 'Neutral', 'Sad'])
    return df


if __name__ == '__main__':
    output_dir = Path('predictions/commercial_results/results_csv')
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_da("predictions/commercial_results/results").to_csv('predictions/commercial_results/results_csv/deepaffect.csv', index = False)
    preprocess_empath("predictions/commercial_results/results").to_csv('predictions/commercial_results/results_csv/empath.csv', index = False)
    preprocess_vokaturi('predictions/commercial_results/results').to_csv('predictions/commercial_results/results_csv/vokaturi.csv', index = False)
