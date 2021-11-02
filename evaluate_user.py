"""
Compare Our models with mean, std, CI over the 100 bootstrapped sets
"""
from evaluate import get_test_set_filenames, preprocess_crema, macro_avg_f1_score, get_confidence_interval
from pathlib import Path
import numpy as np
import pandas as pd

#Calculate F1-scores and return DataFrame
def get_f1_scores(bootstrap_path):
    #bootstrap_path = 'predictions_bootstrap'
    target_intended_observed_ls = []
    model_intended_observed_ls = []
    model_ls = []
    bootstrap_ls = []
    subset_ls = []
    f1_score = []
    
    data_path = Path(bootstrap_path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    
    for path in paths:
        model_intended_observed = path.__str__().split("\\")[2]
        model = path.__str__().split("\\")[3]
        bootstrap = path.__str__().split("\\")[4].split(".")[0].split("_")[-1]
        emotions = [
            'Filename',	
            'Anger',
            'Disgust',
            'Fear',
            'Happy',
            'Neutral',	
            'Sad']
        
        pred_df = pd.read_csv(
            path,
            usecols=(emotions)
            )
        
        col_names = {k: "{}_{}_pred".format(model_intended_observed, k[:3].upper()) if k != 'Filename'
                     else k.lower() for k in pred_df.columns}
        pred_df.rename(columns = col_names, inplace = True)
        
        #Convert probabilities to 0s and 1s
        for col in pred_df.drop(['filename'], axis = 1).columns:
            pred_df[col] = np.where(pred_df[col] >= 0.5, 1, 0)
        
        test_file_name = get_test_set_filenames(path)
        crema = preprocess_crema(demographics_path = 'Data/VideoDemographics.csv',
            ratings_path = 'Data/processedResults/summaryTable.csv',
            test_file_names= test_file_name)

        for key, cat in crema.items():
            
            subset = key.split("_")[-1]
            
            if 'observed' in key:
                target_intended_observed = 'observed'
            else:
                target_intended_observed = 'intended'
                            
            cat_pred_df = cat.merge(pred_df, on = 'filename')
            
            if model_intended_observed == target_intended_observed:
                y_true = cat_pred_df[cat_pred_df.filter(regex="{}_".format(target_intended_observed))
                                 .columns] \
                    .drop(list(cat_pred_df.filter(regex = 'pred')), axis = 1).to_numpy()
            else:
                y_true = cat_pred_df[cat_pred_df.filter(regex="{}_".format(target_intended_observed))
                                 .columns].to_numpy()
                    
            y_pred = cat_pred_df[cat_pred_df.filter(regex='pred').columns].to_numpy()
            
            target_intended_observed_ls.append(target_intended_observed)    
            model_intended_observed_ls.append(model_intended_observed)
            model_ls.append(model)
            subset_ls.append(subset)
            bootstrap_ls.append(bootstrap)
            f1_score.append(macro_avg_f1_score(y_true, y_pred))
        
        
    f1_df = pd.DataFrame(data=zip(target_intended_observed_ls,
                                      model_intended_observed_ls,
                                      model_ls,
                                      bootstrap_ls,
                                      subset_ls,
                                      f1_score), 
                             columns=['target_intended_observed',
                                      'model_intended_observed',
                                      'model', 
                                      'bootstrap', 
                                      'subset',
                                      'f1_score'])
    return f1_df

if __name__ == '__main__':    
    df = get_f1_scores('predictions_bootstrap')
    #df.to_csv('f1_results.csv', index = False)
    get_confidence_interval(df, 0.95).to_csv('f1_score_results/user/user_models.csv', index = False) 
