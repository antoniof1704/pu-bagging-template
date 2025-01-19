# PU Bagging Template
A template for a PU Bagging approach (raw code). PU bagging is effective when reliable negatives can't be identified in unlabeled data. Bootstrapping creates resampled subsets, helping the model distinguish true positives from true negatives. This process infers the negative class distribution, improving classification and model robustness.

## PU Bagging Overview:

- Create a bootstrap sample from the training data by combining all positive data points with a random sample from the unlabeled points, with replacement.
- Build a classifier from this bootstrap sample, treating positive and unlabeled data points as positives and negatives, respectively.
- Apply the classifier to whatever unlabeled data points were not included in the random sample – hereafter called OOB (“out of bag”)     points – and record their scores.
- Repeat the three steps above many times and finally assign to each point the average of the OOB scores it has received.

Key Paper: 'A bagging SVM to learn from positive and unlabeled examples'(Mordelet and Vert, 2013) ttp://dx.doi.org/10.1016/j.patrec.2013.06.010

Algorithm for PU Bagging Method (Mordelet and Vert, 2013)

![image](https://github.com/user-attachments/assets/92502aa1-e35d-470b-a7fe-226d4bd6308a)

Process flow for PU bagging from another paper (Wang et al., 2024) (Paper Title: A PU‐learning based approach for cross‐site scripting attacking reality detection)

![image](https://github.com/user-attachments/assets/c64dfc26-cfff-49af-9feb-b2a7c16d2fad)


## High Level Summary of Bootstrap Process Within the Code:

This process generates 100 bootstrap samples from the training data, allowing for resampling. These bootstrap samples are then balanced, where the majority class (unlabelled) is subsampled to match the minority class (positive) within each bootstrap. Once the bootstrap is balanced, the process identifies all cases that are not included in each initial bootstrap sample before balancing (Out-Of-Bag, or OOB). A random forest classifier is then fitted to the rebalanced sample (X and Y). The model is applied to the OOB cases, and the predictions for positive cases are recorded and stored. Additionally, the model is applied to the test data, and those predictions are stored as well. This concludes the iterative phase.

Next, the average of all stored bootstrap OOB predictions is calculated. This helps evaluate the model's performance on unseen data during training, serving as a diagnostic tool to detect potential overfitting or underfitting. It can be used as a substitute for a separate validation dataset.

The predictions stored from the test data are aggregated using majority voting to make final predictions on unseen data (test set). The idea is that combining multiple models through voting is more reliable than any individual model, reducing bias and variance, and ultimately making the model more robust.

