from glob import glob
import os
import pickle
import itertools
import pandas as pd
import numpy as np

### Warning import ###
import warnings
warnings.filterwarnings('ignore')

### Graph imports ###
import matplotlib.pyplot as plt

### Sklearn imports ###
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

# Load datas from pickle
[X_train, y_train] = pickle.load(open("/content/drive/MyDrive/IS4152/Data/SVM_features/CREMA_preprocessing_train.p", 'rb'))
[X_test, y_test] = pickle.load(open("/content/drive/MyDrive/IS4152/Data/SVM_features/CREMA_preprocessing_test.p", 'rb'))

# Scale train and test dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

multi_selected_features = []

for i in range(len(y_train[0])):
    # k-highest scores analysis on features
    Kbest = SelectKBest(k="all")
    selected_features = Kbest.fit(X_train, y_train[:,i])
    multi_selected_features.append(selected_features)

multi_selected_features = [selected_features.pvalues_ for selected_features in multi_selected_features]
multi_selected_features = np.asarray(multi_selected_features)

multi_selected_features = np.min(multi_selected_features, axis=0)

# Plot P-values
plt.figure(figsize=(20, 10))
plt.plot(multi_selected_features)
plt.title("p-values for each features", fontsize=22)
plt.xlabel("Features")
plt.ylabel("P-value")
plt.show()

# Display Comment
alpha = 0.01
print("Number of p-values > à 1% : {}".format(np.sum(multi_selected_features > alpha)))

# Remove non-significant features
X_train = X_train[:,np.where(multi_selected_features < alpha)[0]]
X_test = X_test[:,np.where(multi_selected_features < alpha)[0]]

# Covariance matrix
cov = pd.DataFrame(X_train).cov()

# Eigen values of covariance matrix
eig = np.linalg.svd(cov)[1]

# Plot eigen graph
fig = plt.figure(figsize=(20, 10))
plt.title('Decrease of covariance matrix eigen values', fontsize = 22)
plt.plot(eig, '-*', label = "eig-value")
plt.legend(loc = 'upper right')
plt.show()

# Initialize PCA
pca = PCA(n_components=140)

# Apply PCA on train and test set
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Hyperparameter tuning, takes a very long time to run

# Set C and Gamma parameters list
G_list = [0.001, 0.005, 0.01, 0.125]
C_list = [1, 2, 3, 4, 5, 7, 10, 20, 128]
# G_list = [1, 0.1, 0.01 , 0.001, 0.0001]
# C_list = [0.1, 1, 10, 100, 1000]

# Set the parameters for cross-validation
parameters = [{'estimator__kernel': ['rbf'], 'estimator__C': C_list, 'estimator__gamma': G_list}]

# Initialize SVM model
model = MultiOutputClassifier(SVC(decision_function_shape='ovr'))

# Cross Validation 
# add fit to find best parameters
cv = GridSearchCV(model, parameters, cv=3, verbose=0, n_jobs=-1, refit=True).fit(X_train, y_train)

# Print Best parameters
print("Best parameters set found on train set:")
print(cv.best_params_)

# Confusion matrix plot function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=22)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

# Fit best mode
model = MultiOutputClassifier(SVC(kernel='rbf', C=1, gamma=0.005, decision_function_shape='ovr')).fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Score
score = model.score(X_test, y_test)

# Build dataFrame
# df_pred = pd.DataFrame({'Actual': y_test, 'Prediction': score})

# Print Score
print('Accuracy Score on test dataset: {}%'.format(np.round(100 * score,2)))

print(classification_report(y_test, pred, zero_division=0))

# # Compute confusion matrix
# confusion = confusion_matrix(y_test, pred)

# # Plot non-normalized confusion matrix
# plt.figure(figsize=(15, 15))
# plot_confusion_matrix(confusion, classes=set(y_test),normalize=True,
#                       title='Confusion matrix on train set with gender differentiation')

# save the model to local
pickle.dump(model, open('/content/gdrive/MyDrive/IS4152/Data/SVM_model/acted/Test_MODEL_CLASSIFIER.p', 'wb'))

# Save label encoder
#pickle.dump(lb, open("/content/gdrive/MyDrive/IS4152/Data/SVM_model/MODEL_ENCODER.p", "wb"))

# Save PCA
pickle.dump(pca, open("/content/gdrive/MyDrive/IS4152/Data/SVM_model/acted/Test_MODEL_PCA.p", "wb"))

# Save MEAN and STD of each features
MEAN = multi_selected_features.mean(axis=0)
STD = multi_selected_features.std(axis=0)
pickle.dump([MEAN, STD], open("/content/gdrive/MyDrive/IS4152/Data/SVM_model/acted/Test_MODEL_SCALER.p", "wb"))

# Save feature parameters
stats = ['mean', 'std', 'kurt', 'skew', 'q1', 'q99']
features_list = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy', 'spectral_flux', 'sprectral_rolloff']
win_step = 0.01
win_size = 0.025
nb_mfcc = 12
diff = 0
PCA = True
DICO = {'stats':stats, 'features_list':features_list, 'win_size':win_size, 'win_step':win_step, 'nb_mfcc':nb_mfcc, 'diff':diff, 'PCA':PCA}
pickle.dump(DICO, open("/content/gdrive/MyDrive/IS4152/Data/SVM_model/acted/Test_MODEL_PARAM.p", "wb"))
