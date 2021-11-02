"""
Compare commercial models with mean, std, CI over the original test set split.
"""

from evaluate import get_test_set_filenames, preprocess_crema, macro_avg_f1_score
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

"""
test = 'predictions/commercial_results/results_csv'

get_test_set_filenames(test)
[WindowsPath('predictions/commercial_results/results_csv/deepaffect.csv'),
 WindowsPath('predictions/commercial_results/results_csv/empath.csv'),
 WindowsPath('predictions/commercial_results/results_csv/vokaturi.csv')]

data_path = Path("predictions/commercial_results/results_csv")
paths = [path for path in data_path.glob('**/*') if path.is_file()]
"""

def get_f1_scores(pred_path):
    #bootstrap_path = 'predictions_bootstrap'
    #pred_path = "predictions/commercial_results/results_csv"
    target_intended_observed_ls = []
    model_ls = []
    subset_ls = []
    f1_score = []
    
    data_path = Path(pred_path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    paths.sort()

    for path in tqdm(paths):
        model = path.stem
        emotions = [
            'Filename',	
            'Anger',
            'Happy',
            'Neutral',	
            'Sad']
        
        pred_df = pd.read_csv(
            path,
            usecols=(emotions)
            )
        
        col_names = {k: "{}_pred".format(k[:3].upper()) if k != 'Filename'
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
            cat_pred_df.columns

            y_true = cat_pred_df[cat_pred_df.filter(regex="{}_".format(target_intended_observed))
                                 .columns]. \
                                drop(columns = ['{}_DIS'.format(target_intended_observed), 
                                                '{}_FEA'.format(target_intended_observed)]). \
                                to_numpy()
            y_pred = cat_pred_df[cat_pred_df.filter(regex='pred').columns].to_numpy()
            
            target_intended_observed_ls.append(target_intended_observed)    
            model_ls.append(model)
            subset_ls.append(subset)
            f1_score.append(macro_avg_f1_score(y_true, y_pred))
        
        
    f1_df = pd.DataFrame(data=zip(target_intended_observed_ls,
                                      model_ls,
                                      subset_ls,
                                      f1_score), 
                             columns=['target_intended_observed',
                                      'model', 
                                      'subset',
                                      'f1_score'])
    return f1_df


if __name__ == '__main__':
    df = get_f1_scores("predictions/commercial_results/results_csv")

    output_path = Path('f1_score_results/commercial/commercial_models.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index = False)
