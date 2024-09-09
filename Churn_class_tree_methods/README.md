## __This project predicts whether internet and telephone provider customers will churn__

We first do exploratory data analysis.
We then use one-hot encoding for categorical features and split data into training and test samples.
We then define a function that takes as arguments training model, 
dictionary with model parameter names as keys and corresponding values to consider during Grid Search Cross-Validation,
and labels and features for training and test data samples. THe function then performs fitting and evaluates 
performance of different tree based models.
