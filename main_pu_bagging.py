### START ###

## Import Libraries ##
# Not all of these libraries are nessassry (code was taking from a large project)

import pandas as pd

import random

import pickle

from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    ParameterGrid,
    ParameterSampler,
)

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest,
    GradientBoostingClassifier,
    BaggingClassifier,
)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils import resample
from sklearn.inspection import permutation_importance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, shapiro
from scipy import stats

from utils.generate_eval_metrics import generate_eval_metrics

import numpy as np

from ucimlrepo import fetch_ucirepo 
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
print(breast_cancer_wisconsin_diagnostic) 
y = breast_cancer_wisconsin_diagnostic.data.targets 
print(y)

## Import input dataset (target variable must be 1 (positive instance) or 0 (unlabelled instance)) ##

print("Importing Wisconsin Breast Cancer Diagnostic Data....")

df = breast_cancer_wisconsin_diagnostic
df

print("Original Dataset Counts")
print(df['Diagnosis'].value_counts()) 


# Split input dataset into features and target.

y = df["Diagnosis"] 
X = df.drop("Diagnosis", axis=1) 


# Split input dataset into train (80%) and test (20%) for modelling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## STANDARD RANDOM FOREST (test without PU Bagging) ##

# Use a standard (balanced class) random forest classifier (without PU Bagging method) to compare results with PU Bagging Method Below.

rf = RandomForestClassifier(class_weight="balanced", max_leaf_nodes=8, random_state=1)
rf = rf.fit(X_train, y_train) #using hidden pos y train
y_pred = rf.predict(X_test)
 
generate_eval_metrics(y_test, y_pred)




## PU BAGGING ##

#High-Level Summary of Next Step in Process:

#This process generates 100 bootstrap samples from the training data, allowing for resampling. These bootstrap samples are then balanced, where 
#the majority class is subsampled to match the minority class within each bootstrap. Once the bootstrap is balanced, the process identifies all cases that 
#are not included in each initial bootstrap sample before balancing (Out-Of-Bag, or OOB). A random forest classifier is then fitted to the rebalanced 
#sample (X and Y). The model is applied to the test data, and the predictions for positive cases (avoidance) are recorded and stored. Additionally, the 
#model is applied to the train data, and those predictions are stored as well (will be for hidden positives eval). This concludes the iterative phase.

#Next, the average of all stored bootstrap OOB predictions is calculated. This helps evaluate the model's performance on unseen data during training, 
#serving as a diagnostic tool to detect potential overfitting or underfitting. It can be used as a substitute for a separate validation dataset. 

#The predictions stored from the test and train data are aggregated using majority voting to make the final predictions (on both train and test). The 
#idea is that combining multiple models through voting is more reliable than any individual model, reducing bias and variance, and ultimately making 
#the model more robust.

# Set up parameters for bootstrapping 

# Bootstrap Sampling (take multiple samples from the training set)

n_iterations = 100  # Number of bootstrap samples
n_size = len(X_train) 
model = RandomForestClassifier() #using a random forest algorithm as the classifier

# Initialize array to store OOB scores for each point
oob_scores = np.zeros((len(X_train), n_iterations))  # Store OOB scores for each point in each iteration

# List to hold predictions from each bootstrap sample (for evaluation)
bootstrap_predictions = []

# Store model weights so they can be applied to new data
list_of_model_weights = []


for i in range(n_iterations):

    # Sample with replacement from the training data
    X_resample, y_resample = resample(X_train, y_train, n_samples=n_size, random_state=i)

    # Create balanced bootstrap sample. Ensure positives = unlabelled (random sample of unlabelled within bootstrap sample)
    X_unlab = X_resample[y_resample == 0]  # unlabelled (majoirty)
    X_pos = X_resample[y_resample == 1]  # positive (minority)
   
    # Use all of class 1 (positive class) without resampling
    X_pos_resample = X_pos
    y_pos_resample = np.ones(len(X_pos_resample))  # Corresponding labels for class 1

    # Sample with replacement from class 0 (negative class) to get a subsample
    X_neg_resample = resample(X_unlab, n_samples=len(X_pos), random_state=i)  # Sample same number as class 1
    y_neg_resample = np.zeros(len(X_neg_resample))  # Corresponding labels for class 0

    # Combine the resampled positive and negative class samples
    X_bal_resample = np.vstack([X_pos_resample, X_neg_resample])
    y_bal_resample = np.hstack([y_pos_resample, y_neg_resample])

    # Identify OOB points (points not included in the (balanced) resampled sample)
    oob_indices = np.setdiff1d(np.arange(n_size), np.unique(np.where(np.isin(X_train, X_bal_resample).all(axis=1))[0]))

    # Train the model on the balanced resampled dataset
    model.fit(X_bal_resample, y_bal_resample)

    # Apply model to OOB points and record the prediction probabilities for the positive class (class 1)
    oob_preds_proba = model.predict_proba(X_train.iloc[oob_indices])[:, 1]  # Probability of class 1 (positive)

    # Store OOB prediction scores for each point
    oob_scores[oob_indices, i] = oob_preds_proba

    # Make predictions on the test set
    predictions = model.predict(X_test)
    bootstrap_predictions.append(predictions)

    # Append model wieght from each bootstrap to the list of weights
    list_of_model_weights.append(model)


# Aggregate OOB scores (average across all bootstrap samples) - ONLY FOR QA PURPOSES
average_oob_scores = np.mean(oob_scores, axis=1)

# Aggregate predictions for the test set using majority voting
bootstrap_predictions = np.array(bootstrap_predictions)

# Convert the predictions to integers if necessary (in case they are floats)
bootstrap_predictions = bootstrap_predictions.astype(int)

# Majority voting on the predictions
final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=bootstrap_predictions)
 

## Generate Eval metrics ##

# Confusion Matrix

actual = y_test
predicted = final_predictions

cm_result  = confusion_matrix(actual, predicted) #simple confusion matrix

custom_labels = ['TARGET_VARIABLE_0', 'TARGET_VARIABLE_1'] # CHANGE TARGET_VARIABLE_0 = Unlabelled Instance (0), TARGET_VARIABLE_1 = Positive Instance (1)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_result, display_labels = custom_labels)
cm_display.plot()

print("Confusion Matrix:")
plt.show()

# Accuracy

accuracy = accuracy_score(y_test, final_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Precision

precision = precision_score(y_test, final_predictions)
print(f"Precision: {precision:.2f}")

# Recall
recall = recall_score(y_test, final_predictions)
print(f"Recall: {recall:.2f}")

# F1 Score

f1 = f1_score(y_test, final_predictions)
print(f"F1 Score: {f1:.2f}")


## HOW MODEL WEIGHTS WOULD BE APPLIED TO NEW DATA ##

#predictions = []

#for model in list_of_model_weights:

    #prediction = model.predict(new data)
    #predictions.append(prediction)

# EG. User 1 (model weights  x 100 - 100 bootstrap models)
#     User 2 (model weights  x 100 - 100 bootstrap models)
#                              '''


### END ###

# FEEL FREE TO ADD ADJUSTMENTS TO CODE TO MAKE IT EASIER FOR THE NEXT USER ##