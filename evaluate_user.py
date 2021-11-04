"""
Compare Our models with mean, std, CI over the 100 bootstrapped sets
"""
from evaluate import get_test_set_filenames, preprocess_crema, scores, get_confidence_interval
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

#Calculate F1-Score & Recall and return DataFrame
def get_metrics(bootstrap_path):
    #bootstrap_path = 'predictions_bootstrap'
    target_intended_observed_ls = []
    model_intended_observed_ls = []
    model_ls = []
    bootstrap_ls = []
    subset_ls = []
    f1_score = []
    recall = []
    
    data_path = Path(bootstrap_path)
    paths = [path for path in data_path.glob('**/*') if path.is_file()]
    paths.sort()
    
    for path in tqdm(paths):
        model_intended_observed = path.parts[-3]
        model = path.parts[-2]
        bootstrap = int(path.parts[-1].split(".")[0].split("_")[-1])
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
            target_intended_observed = key.split("_")[0]
            subset = key.split("_")[-1]

            cat_pred_df = cat.merge(pred_df, on = 'filename')

            y_true = cat_pred_df \
                .drop(columns=cat_pred_df.filter(regex='.+pred').columns) \
                .filter(regex='{}_'.format(target_intended_observed)) \
                .to_numpy()
            y_pred = cat_pred_df.filter(regex='.+pred').to_numpy()

            target_intended_observed_ls.append(target_intended_observed)    
            model_intended_observed_ls.append(model_intended_observed)
            model_ls.append(model)
            subset_ls.append(subset)
            bootstrap_ls.append(bootstrap)

            scores_dict = scores(y_true, y_pred)
            f1_score.append(scores_dict['f1-score'])
            recall.append(scores_dict['recall'])


    f1_df = pd.DataFrame(data=zip(target_intended_observed_ls,
                                      model_intended_observed_ls,
                                      model_ls,
                                      bootstrap_ls,
                                      subset_ls,
                                      f1_score,
                                      recall), 
                             columns=['target_intended_observed',
                                      'model_intended_observed',
                                      'model', 
                                      'bootstrap', 
                                      'subset',
                                      'f1_score',
                                      'recall'])
    return f1_df

if __name__ == '__main__':    
    df = get_metrics('predictions_bootstrap')

    output_path = Path('bias_results/user/user_models_all.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index = False)

    output_path = Path('bias_results/user/user_models.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    get_confidence_interval(df, 0.95).to_csv(output_path, index = False) 
