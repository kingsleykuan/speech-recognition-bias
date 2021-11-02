from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import scipy.stats as st
from crema_metadata import read_actor_demographics, read_ratings

#Get the filenames of the test set.
def get_test_set_filenames(path):
    testFileNames = pd.read_csv(path)['Filename'].to_numpy()
    return testFileNames

#Preprocess VideoDemographics.csv and summaryTable.csv
def preprocess_crema(ratings_path, demographics_path, test_file_names):
    demo = read_actor_demographics(demographics_path)
    ratings = read_ratings(ratings_path)
    
    #Generate observed and intended columns, and apply OHE
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
        axis = 1).drop(columns = ['intended'])
    
    #Filter according to test_file_names
    ratings_demo = ratings_demo[ratings_demo['filename'].isin(test_file_names)]
    
    observed_ratings_demo = ratings_demo.drop(list(ratings_demo.filter(regex = 'intended')), axis = 1)
    intended_ratings_demo = ratings_demo.drop(list(ratings_demo.filter(regex = 'observed')), axis = 1)
    
    #Return dictionary of observed and intended labels for the 4 categories we are comparing
    d = {'observed_male' : observed_ratings_demo.loc[observed_ratings_demo['sex'] == 'Male'],
         'observed_female' : observed_ratings_demo.loc[observed_ratings_demo['sex'] == 'Female'],
         'observed_cauc' : observed_ratings_demo.loc[observed_ratings_demo['race'] == 'Caucasian'],
         'observed_non-cauc' :  observed_ratings_demo.loc[observed_ratings_demo['race'] != 'Caucasian'],
         'intended_male' : intended_ratings_demo.loc[intended_ratings_demo['sex'] == 'Male'],
         'intended_female' : intended_ratings_demo.loc[intended_ratings_demo['sex'] == 'Female'],
         'intended_cauc' : intended_ratings_demo.loc[intended_ratings_demo['race'] == 'Caucasian'],
         'intended_non-cauc' :  intended_ratings_demo.loc[intended_ratings_demo['race'] != 'Caucasian']            
        }
    
    return d

#Macro Avg F1-score computation
def macro_avg_f1_score(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict = True, zero_division = 0)['macro avg']['f1-score']


def get_confidence_interval(df, alpha):
    target_intended_observed = []
    model_intended_observed = []
    model = []
    subset = []
    f1_score_mean = []
    f1_score_std = []
    f1_score_ci = []
    
    #Subset Df
    target_intended_observed_unique = ['intended', 'observed']
    model_intended_observed_unique = ['intended', 'observed']
    model_unique = ['cnn_lstm', 'cnn_lstm_attention', 'cnn_lstm_attention_multitask']
    subset_unique = ['male', 'female', 'cauc', 'non-cauc']
    
    for i in target_intended_observed_unique:   
        for j in model_intended_observed_unique:
            for k in model_unique:
                for l in subset_unique:
                    f1_scores = df.loc[(df['target_intended_observed'] == i) &
                                 (df['model_intended_observed'] == j) &
                                 (df['model'] == k) &
                                 (df['subset'] == l)]['f1_score'].to_numpy()
                    
                    mean_ = np.mean(f1_scores)
                    std_ = np.std(f1_scores)                  
                    ci_ = st.t.interval(alpha=alpha, 
                                  df=len(f1_scores)-1, 
                                  loc=np.mean(f1_scores), 
                                  scale=st.sem(f1_scores))
                    
                    target_intended_observed.append(i)
                    model_intended_observed.append(j)
                    model.append(k)
                    subset.append(l)
                    f1_score_mean.append(mean_)
                    f1_score_std.append(std_)
                    f1_score_ci.append(ci_)
                    
    ci_df = pd.DataFrame(data=zip(target_intended_observed,
                                      model_intended_observed,
                                      model,
                                      subset,
                                      f1_score_mean,
                                      f1_score_std,
                                      f1_score_ci), 
                             columns=['target_intended_observed',
                                      'model_intended_observed',
                                      'model', 
                                      'subset',
                                      'f1_score_mean',
                                      'f1_score_std',
                                      'f1_score_ci'])
    return ci_df