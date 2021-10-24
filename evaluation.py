from pathlib import Path
from sklearn.metrics import classification_report, f1_score
import numpy as np
import os
import pandas as pd
from pathlib import Path
from crema_metadata import read_actor_demographics, read_ratings

# Generate dataset based on emotion type
def preprocess(emotion_type):
    
    #Read data
    demo = read_actor_demographics('Data/VideoDemographics.csv')
    ratings = read_ratings('Data/processedResults/summaryTable.csv')
    
    #Get file names of test set
    test_file_name = []
    test_dir = os.listdir("Data/MelSpecSplit/test")
    for file in test_dir:
        test_file_name.append(file[:-7])
    
    #Generate observed and intended columns, and apply LabelEncorder()
    ratings['observed'] = ratings['voice_vote'] 
    
    ratings['observed_ANG'] = np.where(ratings['voice_vote'].str.contains("A"), 1, 0)
    ratings['observed_DIS'] = np.where(ratings['voice_vote'].str.contains("D"), 1, 0)
    ratings['observed_FEA'] = np.where(ratings['voice_vote'].str.contains("F"), 1, 0)
    ratings['observed_HAP'] = np.where(ratings['voice_vote'].str.contains("H"), 1, 0)
    ratings['observed_NEU'] = np.where(ratings['voice_vote'].str.contains("N"), 1, 0)
    ratings['observed_SAD'] = np.where(ratings['voice_vote'].str.contains("S"), 1, 0)
    
    ratings = ratings[ratings.filter(regex='filename|observed_').columns]
    
    ratings.loc[:, 'intended'] = np.vectorize(lambda x : x.split('_')[2])(ratings.loc[:,'filename'])
    ratings.loc[:, 'actor_id'] = np.vectorize(lambda x : int(x.split('_')[0]))(ratings.loc[:,'filename'])
    
    #Merge dataframes
    ratings_demo = pd.concat([
        ratings.merge(demo, on = 'actor_id'),
        pd.get_dummies(ratings['intended'], prefix = "intended_", prefix_sep = '')],
        axis = 1)
    
    ratings_demo = ratings_demo[ratings_demo['filename'].isin(test_file_name)]
    
    if emotion_type == 'observed':
        ratings_demo = ratings_demo.drop(list(ratings_demo.filter(regex = 'intended')), axis = 1)
    elif emotion_type == 'intended':
        ratings_demo = ratings_demo.drop(list(ratings_demo.filter(regex = 'observed')), axis = 1)
    else:
        raise Exception("Input 'observed' or 'intended'")
    
    return ratings_demo



def read_predictions(data_path):
    #data_path = Path('predictions')
    data_path = Path(data_path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
            
    reports = []
    for path in paths:
        if 'acted' in path.__str__():
            str_ = 'intended_'
            pred = pd.read_csv(
                path,
                names=(
                    'filename',
                    'intended_ANG_pred',
                    'intended_DIS_pred',
                    'intended_FEA_pred',
                    'intended_HAP_pred',
                    'intended_NEU_pred',
                    'intended_SAD_pred'),
                header=0 
                )
            ratings_demo = preprocess('intended')
            
        elif 'observed' in path.__str__():
            str_ = 'observed_'
            pred = pd.read_csv(
                path,
                names=(
                    'filename',
                    'observed_ANG_pred',
                    'observed_DIS_pred',
                    'observed_FEA_pred',
                    'observed_HAP_pred',
                    'observed_NEU_pred',
                    'observed_SAD_pred'),
                header=0 
                )
            ratings_demo = preprocess('observed') 
        for col in pred.drop(['filename'], axis = 1).columns:
            pred[col] = np.where(pred[col] > 0.5, 1, 0)
        
        ratings_demo_ls = [ 
            ratings_demo.loc[ratings_demo['sex'] == 'Male'],
            ratings_demo.loc[ratings_demo['sex'] == 'Female'],
            ratings_demo.loc[ratings_demo['race'] == 'Caucasian'],
            ratings_demo.loc[ratings_demo['race'] != 'Caucasian']
            ]

        for class_ in ratings_demo_ls:
            ratings_demo_pred = class_.merge(pred, on = 'filename')
            y_true = ratings_demo_pred[ratings_demo_pred.filter(regex=str_).columns] \
                     .drop(list(ratings_demo_pred.filter(regex = 'pred')), axis = 1).to_numpy()
            y_pred = ratings_demo_pred[ratings_demo_pred.filter(regex='pred').columns].to_numpy()
            reports.append(
                classification_report(y_true, y_pred, 
                                      target_names = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD'],
                                      zero_division = 0))
        
    keys = []
    emotion_types = ['intended', 'observed']
    models = ['cnn_lstm', 'cnn_lstm_attention', 'cnn_lstm_attention_multitask']
    demos = ['male', 'female', 'caucasian', 'non-caucasian']
    for emotion in emotion_types:
        for model in models:
            for demo in demos:
                keys.append("{}_{}_{}".format(emotion, model, demo))
                
    return dict(zip(keys, reports))
  
if __name__ == '__main__': 
    for k,v in read_predictions('predictions').items():
        print(k + ':')
        print(v)
        print('-------------------------------------------------------------')
        

    

