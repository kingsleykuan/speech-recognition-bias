"""
Preprocess Commercial .json results
"""
from pathlib import Path
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
        anger.append(preds['anger'])
        happy.append(preds['joy'])
        neutral.append(preds['calm'])
        sad.append(preds['sorrow'])
    
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
    preprocess_da("predictions/commercial_results/results").to_csv('predictions/commercial_results/deepaffect.csv', index = False)
    preprocess_empath("predictions/commercial_results/results").to_csv('predictions/commercial_results/empath.csv', index = False)
    preprocess_vokaturi('predictions/commercial_results/results').to_csv('predictions/commercial_results/vokaturi.csv', index = False)


