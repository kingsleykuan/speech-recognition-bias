from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from crema_metadata import read_actor_demographics, parse_metadata, read_ratings
import scipy.stats as st

#Get the filenames of the test set.
def get_test_set_filenames(path):
    testFileNames = pd.read_csv(path)['Filename'].to_numpy()
    return testFileNames

#Preprocess VideoDemographics.csv and summaryTable.csv
def preprocess_crema(ratings_path, demographics_path, test_file_names):
    metadata = parse_metadata(test_file_names)
    demo = read_actor_demographics(demographics_path)
    ratings = read_ratings(ratings_path)
    
    #Generate observed and intended columns, and apply OHE
    metadata = metadata.filter(items=['filename', 'actor_id', 'emotion'])
    metadata = pd.get_dummies(metadata, prefix='intended', prefix_sep='_', columns=['emotion'])

    ratings['observed_ANG'] = np.where(ratings['voice_vote'].str.contains("A"), 1, 0)
    ratings['observed_DIS'] = np.where(ratings['voice_vote'].str.contains("D"), 1, 0)
    ratings['observed_FEA'] = np.where(ratings['voice_vote'].str.contains("F"), 1, 0)
    ratings['observed_HAP'] = np.where(ratings['voice_vote'].str.contains("H"), 1, 0)
    ratings['observed_NEU'] = np.where(ratings['voice_vote'].str.contains("N"), 1, 0)
    ratings['observed_SAD'] = np.where(ratings['voice_vote'].str.contains("S"), 1, 0)
    ratings = ratings.filter(regex='filename|observed_')

    ratings = pd.merge(metadata, ratings, how='inner', on='filename')
    ratings_demo = pd.merge(ratings, demo, how='inner', on='actor_id')

    observed_ratings_demo = ratings_demo.filter(regex='filename|observed_|sex|race')
    intended_ratings_demo = ratings_demo.filter(regex='filename|intended_|sex|race')

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

def get_confidence_interval(df, alpha):
    target_intended_observed = []
    model_intended_observed = []
    model = []
    subset = []
    f1_score_mean = []
    f1_score_std = []
    f1_score_ci = []
    precision_mean = []
    precision_std = []
    precision_ci = []
    recall_mean = []
    recall_std = []
    recall_ci = []
    
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
                    precisions = df.loc[(df['target_intended_observed'] == i) &
                                 (df['model_intended_observed'] == j) &
                                 (df['model'] == k) &
                                 (df['subset'] == l)]['precision'].to_numpy()
                    recalls = df.loc[(df['target_intended_observed'] == i) &
                                 (df['model_intended_observed'] == j) &
                                 (df['model'] == k) &
                                 (df['subset'] == l)]['recall'].to_numpy()
                    
                    f1_score_mean_ = np.mean(f1_scores)
                    f1_score_std_ = np.std(f1_scores)                  
                    f1_score_ci_ = st.t.interval(alpha=alpha, 
                                  df=len(f1_scores)-1, 
                                  loc=np.mean(f1_scores), 
                                  scale=st.sem(f1_scores))
                    
                    precision_mean_ = np.mean(precisions)
                    precision_std_ = np.std(precisions)                  
                    precision_ci_ = st.t.interval(alpha=alpha, 
                                  df=len(precisions)-1, 
                                  loc=np.mean(precisions), 
                                  scale=st.sem(precisions))
                    
                    recall_mean_ = np.mean(recalls)
                    recall_std_ = np.std(recalls)                  
                    recall_ci_ = st.t.interval(alpha=alpha, 
                                  df=len(recalls)-1, 
                                  loc=np.mean(recalls), 
                                  scale=st.sem(recalls))
                    
                    target_intended_observed.append(i)
                    model_intended_observed.append(j)
                    model.append(k)
                    subset.append(l)
                    f1_score_mean.append(f1_score_mean_)
                    f1_score_std.append(f1_score_std_)
                    f1_score_ci.append(f1_score_ci_)
                    precision_mean.append(precision_mean_)
                    precision_std.append(precision_std_)
                    precision_ci.append(precision_ci_)
                    recall_mean.append(recall_mean_)
                    recall_std.append(recall_std_)
                    recall_ci.append(recall_ci_)
                    
    ci_df = pd.DataFrame(data=zip(target_intended_observed,
                                      model_intended_observed,
                                      model,
                                      subset,
                                      f1_score_mean,
                                      f1_score_std,
                                      f1_score_ci,
                                      precision_mean,
                                      precision_std,
                                      precision_ci,
                                      recall_mean,
                                      recall_std,
                                      recall_ci), 
                             columns=['target_intended_observed',
                                      'model_intended_observed',
                                      'model', 
                                      'subset',
                                      'f1_score_mean',
                                      'f1_score_std',
                                      'f1_score_ci',
                                      'precision_mean',
                                      'precision_std',
                                      'precision_ci',
                                      'recall_mean',
                                      'recall_std',
                                      'recall_ci'])
    return ci_df


#Macro Avg F1-score computation
def macro_avg_score(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict = True, zero_division = 0)['macro avg']
