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
    
    #Subset Df
    target_intended_observed_unique = ['intended', 'observed']
    model_intended_observed_unique = ['intended', 'observed']
    model_unique = ['svm', 'cnn_lstm', 'cnn_lstm_attention', 'cnn_lstm_attention_multitask']
    subset_unique = ['male', 'female', 'cauc', 'non-cauc']
    metrics = [
        'f1_score',
        'f1_score_anger',
        'f1_score_disgust',
        'f1_score_fear',
        'f1_score_happy',
        'f1_score_neutral',
        'f1_score_sad',
        'recall',
        'recall_anger',
        'recall_disgust',
        'recall_fear',
        'recall_happy',
        'recall_neutral',
        'recall_sad',
    ]

    # Lists for mean, std, and ci for each metric
    metrics_mean = [[] for _ in range(len(metrics))]
    metrics_std = [[] for _ in range(len(metrics))]
    metrics_ci_lower = [[] for _ in range(len(metrics))]
    metrics_ci_upper = [[] for _ in range(len(metrics))]

    for i in target_intended_observed_unique:   
        for j in model_intended_observed_unique:
            for k in model_unique:
                for l in subset_unique:
                    target_intended_observed.append(i)
                    model_intended_observed.append(j)
                    model.append(k)
                    subset.append(l)

                    for m, metric in enumerate(metrics):
                        score = df.loc[
                                    (df['target_intended_observed'] == i) &
                                    (df['model_intended_observed'] == j) &
                                    (df['model'] == k) &
                                    (df['subset'] == l)][metric].to_numpy()

                        score_mean_ = np.mean(score)
                        score_std_ = np.std(score)                  
                        score_ci_ = st.t.interval(alpha=alpha, 
                                    df=len(score)-1, 
                                    loc=np.mean(score), 
                                    scale=st.sem(score))

                        metrics_mean[m].append(score_mean_)
                        metrics_std[m].append(score_std_)
                        metrics_ci_lower[m].append(score_ci_[0])
                        metrics_ci_upper[m].append(score_ci_[1])

    # Flatten metrics into:
    # (metric_1_mean, metric_1_std, metric_1_ci_lower, metric_1_ci_upper, ...)
    metrics_flattened = []
    for i in range(len(metrics)):
        metrics_flattened.append(metrics_mean[i])
        metrics_flattened.append(metrics_std[i])
        metrics_flattened.append(metrics_ci_lower[i])
        metrics_flattened.append(metrics_ci_upper[i])

    # Concatenate metrics with other columns
    data = [
        target_intended_observed,
        model_intended_observed,
        model,
        subset,
    ]
    data = data + metrics_flattened

    # Concatenate metric labels with other column labels
    metrics_columns = []
    for metric in metrics:
        metrics_columns.append(f'{metric}_mean')
        metrics_columns.append(f'{metric}_std')
        metrics_columns.append(f'{metric}_ci_lower')
        metrics_columns.append(f'{metric}_ci_upper')
    columns = [
        'target_intended_observed',
        'model_intended_observed',
        'model',
        'subset'
    ]
    columns = columns + metrics_columns

    ci_df = pd.DataFrame(
        data=zip(*data),
        columns=columns)
    return ci_df


def scores(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict = True, zero_division = 0)
