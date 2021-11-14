import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample
from tqdm import tqdm

# Warning import #
import warnings
warnings.filterwarnings('ignore')

TRAIN_DATA_PATH = 'Data/svm_features/train.p'
TEST_DATA_PATH = 'Data/svm_features/test.p'
OUTPUT_PATH = 'predictions_bootstrap/observed/svm/svm.csv'
INTENDED_OBSERVED = 'observed'
USE_BOOTSTRAP = True
NUM_BOOTSTRAP_SAMPLES = 100
EMOTIONS = (
    'Anger',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
)


def hyperparameter_search(X_train, y_train):
    # Set C and Gamma parameters list
    G_list = [0.001, 0.005, 0.01, 0.125]
    C_list = [1, 2, 3, 4, 5, 7, 10, 20, 50, 128]

    # Set the parameters for cross-validation
    parameters = [
        {
            'estimator__kernel': ['rbf'],
            'estimator__C': C_list,
            'estimator__gamma': G_list
        }]

    # Initialize SVM model
    model = MultiOutputClassifier(SVC(decision_function_shape='ovr'))

    # Cross Validation
    # add fit to find best parameters
    cv = GridSearchCV(
        model, parameters, cv=3, verbose=0, n_jobs=-1, refit=True) \
        .fit(X_train, y_train)

    # Print Best parameters
    print("Best parameters set found on train set:")
    print(cv.best_params_)


def train_predict(
        X_train,
        y_train,
        X_test,
        y_test,
        filenames_test,
        output_path):
    # Scale train and test dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Feature selection
    multi_selected_features = []
    for i in range(len(y_train[0])):
        # k-highest scores analysis on features
        Kbest = SelectKBest(k="all")
        selected_features = Kbest.fit(X_train, y_train[:, i])
        multi_selected_features.append(selected_features)
    multi_selected_features = [
        selected_features.pvalues_
        for selected_features in multi_selected_features]
    multi_selected_features = np.asarray(multi_selected_features)
    multi_selected_features = np.min(multi_selected_features, axis=0)

    alpha = 0.01
    print("Number of p-values > à 1% : {}".format(
        np.sum(multi_selected_features > alpha)))

    # Remove non-significant features
    X_train = X_train[:, np.where(multi_selected_features < alpha)[0]]
    X_test = X_test[:, np.where(multi_selected_features < alpha)[0]]

    # Initialize PCA
    pca = PCA(n_components=140)

    # Apply PCA on train and test set
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # hyperparameter_search(X_train, y_train)

    # Fit best
    model = MultiOutputClassifier(SVC(
        kernel='rbf', C=10, gamma=0.005, decision_function_shape='ovr')) \
        .fit(X_train, y_train)

    # Prediction
    pred = model.predict(X_test)

    print(classification_report(y_test, pred, zero_division=0))

    data = pd.DataFrame(pred, columns=EMOTIONS)
    data.insert(0, 'Filename', filenames_test)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, header=True, index=False)


def main():
    # Load datas from pickle
    with open(TRAIN_DATA_PATH, 'rb') as file:
        data = pickle.load(file)
        filenames_train = data['filenames']
        X_train = data['features']
        y_train = data['{}_label_vectors'.format(INTENDED_OBSERVED)]

    if not USE_BOOTSTRAP:
        with open(TEST_DATA_PATH, 'rb') as file:
            data = pickle.load(file)
            filenames_test = data['filenames']
            X_test = data['features']
            y_test = data['{}_label_vectors'.format(INTENDED_OBSERVED)]

        train_predict(
            X_train, y_train, X_test, y_test, filenames_test, OUTPUT_PATH)
    else:
        for i in tqdm(range(NUM_BOOTSTRAP_SAMPLES)):
            bootstrap_indices = resample(
                list(range(len(X_train))),
                replace=True,
                n_samples=len(X_train),
                random_state=i)
            out_of_bootstrap_indices = \
                set(range(len(X_train))) - set(bootstrap_indices)

            filenames_test_oob = [
                filenames_train[i] for i in out_of_bootstrap_indices]
            X_test_oob = np.asarray([
                X_train[i] for i in out_of_bootstrap_indices])
            y_test_oob = np.asarray([
                y_train[i] for i in out_of_bootstrap_indices])

            X_train_bootstrap = np.asarray([
                X_train[i] for i in bootstrap_indices])
            y_train_bootstrap = np.asarray([
                y_train[i] for i in bootstrap_indices])

            output_path = Path(OUTPUT_PATH)
            output_path = output_path.parent \
                / (output_path.stem + f'_{i}' + output_path.suffix)

            train_predict(
                X_train_bootstrap,
                y_train_bootstrap,
                X_test_oob,
                y_test_oob,
                filenames_test_oob,
                output_path)


if __name__ == '__main__':
    main()
