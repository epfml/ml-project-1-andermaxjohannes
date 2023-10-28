# Importing numpy and functions
import numpy as np
import helpers as h
from implementations import *
import matplotlib.pyplot as plt

# Setting a random seed
seed = 934586
np.random.seed(seed)

#### Setting some hyperparameters ########
K = 5                          # The number of folds for cross validation
gamma = 0.1                    # The step size for the numerical regressions
max_iter = 100                  # The maximum number of iterations for the numerical regressions
featureThreshold = 0.7          # The percent of each feature that must be valid for the feature not to be discarded
acceptableMissingValues = 5    # The number of invalid values a sample may have without being discarded

# Making a filename to save the predictions for these parameters to
savefile = f'./Predictions/regularizedLogistic_seed_{seed}_gamma_{gamma}_iter_{max_iter}_K_{K}_featShold_{featureThreshold}_missVals_{acceptableMissingValues}.csv'

# Loading the data
X, xHeader, Y, yHeader, indexedX, indexedXheader, indexedY, indexedYheader = loadTrainingData()
print('')
# Cleaning/feature engineering the data
yClean, xClean, xHeaderClean, removedFeatures = dataCleaning(Y,X,xHeader,featureThreshold,acceptableMissingValues)
print('')
# Making a balanced data set to force the model to not just predict negatively all the time
yBalanced, xBalanced, balancePrior = balanceData(yClean,xClean)
print('')
# Adding dummy variables and replacing the remaining invalid values by the mean
tx = makeTrainingData(xBalanced)
print(f'The resultant dataarray tx has shape {tx.shape}')

# Initializing the weights at zero
initial_w = np.zeros(tx.shape[1])

# Setting some lambdas to check for the best among them
lambdas = np.logspace(-1,-0.5,10)
# Looking for the best of the chosen lambdas
train_loss, test_loss, bestLambda, best_w = determineLambda(yBalanced,tx,initial_w,lambdas,max_iter,K,gamma)

# Plotting the training and testing errors as functions of lambda
plt.plot(lambdas,train_loss,label='Training Loss', color='g')
plt.plot(lambdas,test_loss,label='Testing Loss', color='r')
plt.xscale('log')
plt.legend()
plt.show()

# Training a model with logistic regression, with the chosen lambda
#reg_logistic_regression_fixed_lambda = lambda y, tx, initial_w, max_iters, gamma: reg_logistic_regression(y,tx,bestLambda,initial_w,max_iters,gamma)
#w_logistic, train_loss_logistic, test_loss_logistic = k_fold_cross_validation(yBalanced,tx,K,initial_w,max_iter,gamma,regressionFunction=reg_logistic_regression_fixed_lambda)


############## Making predictions ###############
# Loading the test data
xTest, xIndexedHeader = loadData('./Data/x_test.csv')
print(xTest.shape)

# Making predictions
pred_logistic = makePredictions(best_w,xTest[:,1:],xHeader,xHeaderClean)
# Counting predicted positive cases
print(f'The model predicts {np.sum(pred_logistic)} positive cases')

# Converting the predictions from 0/1 to -1/1, and making a prediction file ready for submission
pred_logistic[pred_logistic == 0] = -1
h.create_csv_submission(xTest[:,0], pred_logistic, f'./Predictions/regularizedLogistic_seed_{seed}_gamma_{gamma}_iter_{max_iter}_K_{K}_featShold_{featureThreshold}_missVals_{acceptableMissingValues}.csv')