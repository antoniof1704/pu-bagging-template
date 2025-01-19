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

import numpy as np


## Import input dataset (target variable must be 1 (positive instance) or 0 (unlabelled instance)) ##

df = pd.read_csv(FILE_NAME) # CHANGE 'FILE_NAME' TO CSV YOU WANT TO IMPORT
df

print("Original Dataset Counts")
print(df['TARGET_VARIABLE'].value_counts()) # CHANGE 'TARGET_VARIABLE' TO NAME OF TARGET VARIABLE 


# Split input dataset into features and target.

y = df["TARGET_VARIABLE"] # CHANGE 'TARGET_VARIABLE' TO NAME OF TARGET VARIABLE 
X = df.drop("TARGET_VARIABLE", axis=1) # CHANGE 'TARGET_VARIABLE' TO NAME OF TARGET VARIABLE 


# Split input dataset into train (80%) and test (20%) for modelling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


##Rationale for Hidding Positives in Training Set:

#A novel approach we have come up with to evaluate a positive-unlabelled learning model is to hide a small number of positive instances within the 
#unlabelled data in the training set. After training the model, you can rerun it on the same training data to generate predictions. By examining the 
#model’s predictions for the hidden positives, you can assess how effectively the model is identifying positive cases within the unlabelled data.

#This evaluation is performed on the training set (X_train) rather than the test set (X_test) because the model has already seen whether each case is 
#positive or unlabelled during training (y_train). This method provides a potential measure of the model's ability to recognize positives within the 
#unlabelled data it was exposed to during training.


# Hide some positives in the unlabelled data in the training set (will be part of the model eval later)

# Identify the indices where the target is 1 (positive class) in the training set
positive_indices_yt = y_train[y_train == 1].index

# Calculate how many positives to change (15% of the positives)
num_to_change = int(0.15 * len(positive_indices_yt))

# Randomly select 15% of the positive indices
np.random.seed(42)  # For reproducibility
pos_to_change = np.random.choice(positive_indices_yt, size=num_to_change, replace=False)

# Create a copy of the original train set to hide postives
y_train_hidden_pos = y_train.copy()

# Change those selected indices to 0 in the copy training set
y_train_hidden_pos.loc[pos_to_change] = 0

# Print value counts of the original and modified y train set
print(y_train.value_counts())
print(y_train_hidden_pos.value_counts())

# Print a list of the hidden positives in the original and modified y train set
print(y_train.loc[pos_to_change])
print(y_train_hidden_pos.loc[pos_to_change])

# Print list of indicies that have chnaged from positive to unlabelled
print(pos_to_change)


## Standard Random Forest (Test):

#To compare results with PU Bagging Method Below

# Create basic random forest classifier
rf = RandomForestClassifier(class_weight="balanced", max_leaf_nodes=8, random_state=1)
rf = rf.fit(X_train, y_train_hidden_pos) #using hidden pos y train
y_pred = rf.predict(X_test)
 

## Generate Eval metrics for Basic Random Forest ##

# Hidden Positives Identified in Training - Start

# Rerun model on the training data (since the model will have seen the y_train values) and get predictions
y_pred_hp = rf.predict(X_train)

# Add y_pred_hp to X_train as a new column and rename table
X_train_rd_pred = X_train.copy()
X_train_rd_pred['y_pred_hp'] = y_pred_hp

# Create a list from the training set with only the hidden pos indicies
pos_indices_in_rd_train = X_train_rd_pred.loc[pos_to_change]

# Assign total number of hidden positives in training set
hidden_positives_in_rd_train = len(pos_indices_in_rd_train)

# Sum the total number of hidden positives correctly identified by the model in the training set
correctly_identified_in_rd_train = sum(pos_indices_in_rd_train['y_pred_hp'] == 1)

# Calculate percentage of Hiden positives identified
percentage_identified_in_rd_train = correctly_identified_in_rd_train / hidden_positives_in_rd_train * 100

# Print the result
print(f"Percentage of Hidden Positives Identified in RD Train Dataset: {percentage_identified_in_rd_train:.2f}%")

# Print a list of the hidden positives in train and their corrosponding predictions (for QA)
print(X_train_rd_pred['y_pred_hp'].loc[pos_to_change])

# Hidden Positives Identified in Training - End

# Confusion Matrix

actual = y_test
predicted = y_pred

cm_result  = confusion_matrix(actual, predicted)

custom_labels = ['TARGET_VARIABLE_0', 'TARGET_VARIABLE_1'] # CHANGE TARGET_VARIABLE_0 = Unlabelled Instance (0), TARGET_VARIABLE_1 = Positive Instance (1)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_result, display_labels = custom_labels)

cm_display.plot()

print("Confusion Matrix:")
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision:.2f}")

# Recall
recall = recall_score(y_test, y_pred)
print(f"Recall: {recall:.2f}")

# F1 Score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1:.2f}")

 

# PU Bagging:

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
 

# Bootstrap Sampling (take multiple samples from the training set)

n_iterations = 100  # Number of bootstrap samples
n_size = len(X_train)
model = RandomForestClassifier()
#SVC(probability=True)

# Initialize array to store OOB scores for each point (for QA)
oob_scores = np.zeros((len(X_train), n_iterations))  # Store OOB scores for each point in each iteration

# List to hold predictions on train set from each bootstrap sample (for evaluation: % of hidden pos identified)
bootstrap_predictions_train = []

# List to hold predictions on test set from each bootstrap sample (for evaluation: conf matrix, accuracy, recall, F1, etc)
bootstrap_predictions_test = []

for i in range(n_iterations):
    # Sample with replacement from the training data
    X_resample, y_resample = resample(X_train, y_train_hidden_pos, n_samples=n_size, random_state=i)

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

    # Make predictions on the whole training set (for hidden pos evaluation)
    predictions_train = model.predict(X_train)
    bootstrap_predictions_train.append(predictions_train)

    # Make predictions on the test set (for standard eval metrics)
    predictions_test = model.predict(X_test)
    bootstrap_predictions_test.append(predictions)

# Aggregate OOB scores (average across all bootstrap samples)
average_oob_scores = np.mean(oob_scores, axis=1)


## Predicitions for Training Set
# Aggregate predictions for the train set using majority voting
bootstrap_predictions_train = np.array(bootstrap_predictions_train)

# Convert the predictions to integers if necessary (in case they are floats)
bootstrap_predictions_train = bootstrap_predictions_train.astype(int)

# Majority voting on the train predictions
final_predictions_train = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=bootstrap_predictions_train)


## Predicitions for Test Set
# Aggregate predictions for the test set using majority voting
bootstrap_predictions_test = np.array(bootstrap_predictions_test)
 
# Convert the predictions to integers if necessary (in case they are floats)
bootstrap_predictions_test = bootstrap_predictions_test.astype(int)

# Majority voting on the test predictions
final_predictions_test = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=bootstrap_predictions_test)



## Generate Eval metrics ##

# Hidden Positives Identified in Training - Start

# Use Predictions (from majority voting) from the Training Set

# Add final_predictions_train to X_train as a new column and rename table
X_train_pu_pred = X_train.copy()
X_train_pu_pred['final_predictions_train'] = final_predictions_train

# Create a list from the training set with only the hidden pos indicies
pos_indices_in_pu_train = X_train_pu_pred.loc[pos_to_change]

# Assign total number of hidden positives in training set
hidden_positives_in_pu_train = len(pos_indices_in_pu_train)

# Sum the total number of hidden positives correctly identified by the model in the training set
correctly_identified_in_pu_train = sum(pos_indices_in_pu_train['final_predictions_train'] == 1)

# Calculate percentage of Hiden positives identified
percentage_identified_in_pu_train = correctly_identified_in_pu_train / hidden_positives_in_pu_train * 100

# Print the result
print(f"Percentage of Hidden Positives Identified in PU Train Dataset: {percentage_identified_in_pu_train:.2f}%")

# Print a list of the hidden positives in train and their corrosponding predictions (for QA)
print(X_train_pu_pred['final_predictions_train'].loc[pos_to_change])

# Hidden Positives Identified in Training - End


# Confusion Matrix
actual = y_test
predicted = final_predictions_test

cm_result  = confusion_matrix(actual, predicted)

custom_labels = ['TARGET_VARIABLE_0', 'TARGET_VARIABLE_1'] # CHANGE TARGET_VARIABLE_0 = Unlabelled Instance (0), TARGET_VARIABLE_1 = Positive Instance (1)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_result, display_labels = custom_labels)

cm_display.plot()

print("Confusion Matrix:")
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, final_predictions_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Precision
precision = precision_score(y_test, final_predictions_test)
print(f"Precision: {precision:.2f}")

# Recall
recall = recall_score(y_test, final_predictions_test)
print(f"Recall: {recall:.2f}")

# F1 Score
f1 = f1_score(y_test, final_predictions_test)
print(f"F1 Score: {f1:.2f}")


### END ###

# FEEL FREE TO ADD ADJUSTMENTS TO CODE TO MAKE IT EASIER FOR THE NEXT USER ##
