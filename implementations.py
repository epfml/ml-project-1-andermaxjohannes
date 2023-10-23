'''
Implementations of the functions to be used in the project.

The six required functions, with their descriptions from the project description, are described below.
Any helper functions are sorted with the first function they are used for.

'''
######### Importing Numpy ##########
import numpy as np


########## Linear regression using gradient descent ##########

def MSE(e):
    '''
    Calculates the mean squared error of the submitted error-array
    Args:
        e: (N,) array of the error fr all N predictions
    Returns:
        Float value of the mean squared error'''
    return e.T @ e / len(e)

"""
def MAE(e): # Not used for now at least
    '''
    Calculates the mean absolute error of the submitted error-array
    Args:
        e: (N,) array of the error fr all N predictions
    Returns:
        Float value of the mean absolute error'''
    return np.sum(np.abs(e)) / len(e)
"""
    
def compute_loss(y, tx, w, lossFunction=MSE):
    """Calculate the loss using either MSE or MAE.
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        w: (d,) array with the model parameters
        lossFunction: The loss function of your choice. Defaults to mean square error
    Returns:
        Float value of the loss, given the selected loss-function
    """
    return lossFunction(y - tx @ w)

def compute_mse_gradient(y, tx, w):
    """Computes the gradient at w for the mean square error .
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        w: (d,) array of model parameters/weights
    Returns:
        (d,) array containing the gradient at w
    """
    return - tx.T @ (y-tx @ w) / len(y)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma): # Required function #1
    """The Gradient Descent (GD) algorithm for the mean square error
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        initial_w: (d,) array with the initialization for the model parameters
        max_iters: Integer denoting the maximum number of iterations of GD
        gamma: Float denoting the stepsize
    Returns:
        w: (d,) array with the final parameters
        loss: Float denoting the final loss
    """
    # Initializing w
    w = np.array(initial_w,dtype=float) # Making sure the array is a float array no matter the provided initialization
    
    for n in range(max_iters):
        grad = compute_mse_gradient(y,tx,w) * gamma # Updating w by a step in the negative gradient direction at the current w
        w -= grad
        #print(f'w: {w}, grad: {grad}')

    return w, compute_loss(y,tx,w) # Returning the final loss and the final parameters

def mse_gd_momentum(y, tx, initial_w, max_iters, gamma, beta=0.5):
    """The Gradient Descent (GD) algorithm for the mean square error, with momentum
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        initial_w: (d,) array with the initialization for the model parameters
        max_iters: Integer denoting the maximum number of iterations of GD
        gamma: Float denoting the stepsize
        beta: Float denoting the ratio between the former momentum and the current gradient
    Returns:
        w: (d,) array with the final parameters
        loss: Float denoting the final loss
    """
    # Initializing w
    w = np.array(initial_w,dtype=float) # Making sure the array is a float array no matter the provided initialization
    m = np.zeros(len(initial_w), dtype=float) # Initializing the momentum as the zero vector

    for n in range(max_iters):
        m = beta * m + (1-beta) * compute_mse_gradient(y,tx,w) # Weighted average of the former momentum and current gradient
        w -= m * gamma # Updating w by a step in the negative momentum direction

    return w, compute_loss(y,tx,w) # Returning the final loss and the final parameters



########## Linear regression using stochastic gradient descent ##########

'''
def compute_stoch_gradient(y,tx, w):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.
    Args:
        y: (B,) array of labels
        tx: (B,d) array of samples and their features
        w: (d,) array of model parameters
    Returns:
        (d,) array containing a stochastic gradient at w
    """
    return - tx.T @ (y - tx @ w) / len(y)
'''

def mini_batch(y,tx,B):
    '''Extract B random labels and their corresponding samples
    Args: 
        y: (N,) array of labels
        tx: (N,d) array of samples and their features
        B: Integer denoting the desired batch size
    Returns:
        yBatch: (B,) array of the randomly extracted labels
        txBatch: (B,d) array of the randomly extracted samples and their features
    '''
    shuffledIndexes = np.random.permutation(len(y)) # Produces an array of lenght N with the indices 0 to N-1 in a random permutation
    yBatch, txBatch = y[shuffledIndexes[0:B]], tx[shuffledIndexes[0:B]] # Extracts the samples of y and tx corresponding to the first B indices in our randomly permuted index array
    return yBatch, txBatch

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1): # Required function #2
    """The Stochastic Gradient Descent algorithm (SGD) for the mean square error
    Args:
        y: (N,) array with the labels
        tx: (N,d) with the samples and their features
        initial_w: (d,) array with the initialization for the model parameters
        batch_size: Integer denoting the desired number of data points to use for computing the stochastic gradient
        max_iters: Integer denoting the maximum number of iterations of SGD
        gamma: Float denoting the stepsize
    Returns:
        w: (d,) array with the final parameters
        loss: Float denoting the final loss
    """
    # Initializing w
    w = np.array(initial_w,dtype=float) # Making sure the array is a float array no matter the provided initialization

    for n in range(max_iters):
        batch_y, batch_x = mini_batch(y,tx,batch_size) # Extracting a batch of x's and corresponding y's to 
        w -= gamma * compute_mse_gradient(batch_y,batch_x,w) # Updating w by a step in the negative stochastic gradient descent direction
        
    return w, compute_loss(y,tx,w) # Returning the final loss and the final parameters

def mse_sgd_momentum(y, tx, initial_w, max_iters, gamma, batch_size=1, beta=0.5):
    """The Stochastic Gradient Descent algorithm (SGD) for the mean square error, but with momentum
    Args:
        y: (N,) array with the labels
        tx: (N,d) with the samples and their features
        initial_w: (d,) array with the initialization for the model parameters
        max_iters: Integer denoting the maximum number of iterations of SGD
        gamma: Float denoting the stepsize
        batch_size: Integer denoting the desired number of data points to use for computing the stochastic gradient
        beta: float between 0 and 1 denoting the ratio between the last and the next step, i.e. a momentum parameter
    Returns:
        w: (d,) array with the final parameters
        loss: Float denoting the final loss
    """
    # Initializing w
    w = np.array(initial_w,dtype=float) # Making sure the array is a float array no matter the provided initialization
    m = np.zeros(len(initial_w),dtype=float) # Initializing the momentum as zero

    for n in range(max_iters):
        batch_y, batch_x = mini_batch(y,tx,batch_size) # Extracting a batch of x's and corresponding y's
        m = beta * m + (1-beta) * compute_mse_gradient(batch_y,batch_x,w) # Updating the momentum by a linear combination of the current gradient and the former step
        w -= gamma * m # Updating w by a step in the negative momentum direction
        #print(f'{n+1}/{max_iters}: w: {w}')
        
    return w, compute_loss(y,tx,w) # Returning the final loss and the final parameters

########## Least squares regression using normal equations ##########

def least_squares(y,tx): # Required function #3
    '''Computes the weights to minimize the mean square error, by way of the normal equations
    Args:
        tx: (N,d) array with the samples and their features
        y: (N,) array with the labels
    Returns:
        w: (d,) array with the optimal least squares parameters
        loss: Float denoting the least squares loss of the solution w
    '''
    w = np.linalg.solve(tx.T@tx,tx.T@y)
    return w, compute_loss(y,tx,w)



########## Ridge regression using normal equations ##########

def ridge_regression(y, tx, lambda_): # Required function #4
    ''' Implement ridge regression for the normal equations
    Args:
        y: (N,) array with the labels
        tx: (N,d) array with the samples and their features
        lambda_: Float denoting how much of a 'punisment' should be given for complex solutions
    Returns:
        w: (d,) array of the optimal parameters
    '''

    """
    It can be shown that for the ridge estimator 
    beta^hat(lambda) = argmin_beta 1/n ||y-tx@beta||**2 + lambda*beta.T @ beta
    the ridge regression solution exists (even when X does not have full rank), and is given by
    (tx.T @ tx + n*lambda*I) @ beta^hat(lambda) = tx.T @ y
    where n =len(y), and I is the identity matrix
    """
    w = np.linalg.solve(2 * len(y) * lambda_ * np.identity(tx.shape[1]) + tx.T@tx , tx.T@y)  # Computing w by way of the normal equations
    return w, compute_loss(y,tx,w)



########## Logistic regression using gradient descent or SGD (y ∈ {0,1}) ##########

def logistic(z):
    """ Applies the logistic (also called sigmoid) function on z
    Args:
        z: (N,) array
    Returns:
        (n,) array
    """
    return 1/(1+np.exp(-z))

def logistic_loss(y, tx, w):
    """ Compute the cost by negative log likelihood.
    Args:
        y: (N,) array with labels
        tx: (N,D) array with samples and their features
        w: (D,) array with the parameters
    Returns:
        float non-negative loss
    """
    return - np.sum( y * np.log(logistic(tx @ w)) + (1-y)* np.log(1-logistic(tx @ w)) ) / y.shape[0]

def logistic_gradient(y, tx, w):
    """ Compute the gradient of the logistic loss
    Args:
        y: (N,) array with labels
        tx: (N,D) array with samples and their features
        w: (D,) array with the parameters
    Returns:
        (D, 1) array with the gradient of the logistic loss with respect to the parameters
    """
    return tx.T @ (logistic(tx@w)-y) / y.shape[0]

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ''' Use logistic regression for binary clasification
    Args:
        y: (N,) array with labels
        tx: (N,D) array with samples and their features
        initial_w: (D,) array with some initial parameters
        max_iters: integer of the maximum number of iterations
        gamma: float of the step length
    Return:
        w: The final parameters
        loss: The final loss
    '''
    w = np.array(initial_w, dtype=float)
    for i in range(max_iters):
        w += -gamma * logistic_gradient(y,tx,w)
    
    return w, logistic_loss(y,tx,w)

########## Regularized logistic regression using gradient descent or SGD (y ∈ {0,1}, with regularization term λ∥w∥2) ##########

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma): # Required function #6
    ''' Perform regularized logistic regression for binary classification
    Args:
        y: (N,) array with labels
        tx: (N,D) array with samples and their features
        lambda_: float of the penalization parameter
        initial_w: (D,) array with some initial parameters
        max_iters: integer of the maximum number of iterations
        gamma: float of the step length
    Returns:
        w: The final parameters
        loss: The final loss
    '''
    w = np.array(initial_w, dtype=float)
    for i in range(max_iters):
        w += -gamma * (logistic_gradient(y,tx,w) + 2*lambda_*w) # Updating the paramters by the gradient with the regularization term
    
    return w, logistic_loss(y,tx,w)



########## Data cleaning functions ##########

def removeBadFeatures(x,xHeader,threshold=0.7):
    ''' Function checking how many features have more than {threshold} parts valid entries, and removing the 'bad' features with less than {threshold} valid entries
    Args:
        x: (N,d) array of the dataset
        xHeader: (d,) array of the header for the dataset
        threshold: float between 0 and 1
    Returns:
        (N,d-b) array of the new dataset, where b is the number of features with less than threshold parts valid entries
        (d-b) array of the new header, where b is the number of features with less than threshold parts valid entries
    '''
    # Counting the number of valid values for each feature, and calculating the percentage of valid entries
    validFeatureVals = x.count(axis=0) # The number of valid entries for each feature
    validFeatureValsPercent = validFeatureVals/x.shape[0] # The percentage of valid entries for each feature

    # Finding the indices of all the features with number of features above and below a threeshold
    featureIndicesAboveThreeshold = np.argwhere(validFeatureValsPercent > threshold).flatten() # Finding the indices where there are more than threeshold percent valid entries
    
    # Printing the good vs bad features
    print(f'For a threshold of {threshold}, there are {len(featureIndicesAboveThreeshold)} good features, and {x.shape[1]-len(featureIndicesAboveThreeshold)} bad features')

    # Removing the features that appears less than {threeshold} of the time, and returning the others
    return x[:,featureIndicesAboveThreeshold], xHeader[featureIndicesAboveThreeshold]


def removeBadSamples(y,x,acceptableMissingValues):
    ''' Function checking how many samples miss more than {acceptableMissingValues} values, and removing those samples
    Args:
        y: (N,) array of the labels
        x: (N,d) array of the data
        acceptableMissingValues: integer between 0 and d
    Returns:
        (N-b) array of the new labels, where b is the number of samples missing more than {acceptableMissingValues}
        (N-b,d) array of the new dataset, where b is the number of samples missing more than {acceptableMissingValues}
    '''
    # Counting the number of remaining valid entries for each sample
    validSampleVals = x.count(axis=1)

    # Find the indices of the samples with more than {acceptableMissingValues} invalid missing
    sampleIndicesAboveThreeshold = np.argwhere(validSampleVals >= x.shape[1]-acceptableMissingValues).flatten()
    print(f'There remains in the data {len(sampleIndicesAboveThreeshold)} samples with at most {acceptableMissingValues} missing values')

    # Removing samples with more than {acceptableMissingValues} missing values
    return y[sampleIndicesAboveThreeshold], x[sampleIndicesAboveThreeshold]


def standardizeData(x):
    ''' Function for standardizing the data
    Args:
        x: (N,d) array
    Returns:
        (N,d) array where x has been subtracted its mean, and divided by its standard deviation
    '''
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0) # Subtract the mean and divide by the standard deviation

def balanceData(y,x):
    ''' Function for balancing the number of positive and negative cases in the dataset for regression, which may help find real correlations instead of just guessing based on the prior
    Args:
        y: (N,) array with labels
        x: (N,d) array with data
    Returns:
        (N_b,) array with labels, where N_b is min(positiveCase, negativeCases) *2
        (N_b,d) array with the balanced data
    '''
    # Extracting the indices of the positive and negative cases, bundling them in a tuple, and bundling their lengths in tuples
    positiveCases = np.where(y == 1)[0]
    negativeCases = np.where(y == -1)[0]
    casesIndices = (positiveCases,negativeCases)
    casesLengths = (len(positiveCases),len(negativeCases))

    # Finding which subset is the smallest and largest (aka are there more negative or positive cases), and setting the smallestSubsetLength to the length of the smallest subset
    smallestSubset = np.argmin(casesLengths)
    largestSubset = np.argmax(casesLengths)
    smallestSubsetLength = casesLengths[smallestSubset]

    # Storing the cases from the smallest subset in an array
    balancedY = np.zeros(smallestSubsetLength*2)
    balancedX = np.zeros((smallestSubsetLength*2,x.shape[1]))
    balancedY[:smallestSubsetLength] = (y[casesIndices[smallestSubset]]).flatten()
    balancedX[:smallestSubsetLength] = x[casesIndices[smallestSubset]]

    # Randomly choosing as many samples from the largest subset as there are in the smallest subset, and storing them in the balanced array
    randomSampleIndices = np.random.permutation(casesLengths[largestSubset])[:smallestSubsetLength]
    balancedX[smallestSubsetLength:] = (x[casesIndices[largestSubset]])[randomSampleIndices]
    balancedY[smallestSubsetLength:] = ((y[casesIndices[largestSubset]])[randomSampleIndices]).flatten()

    # Shuffling the balanced arrays so no model can learn to classify the entries by their position in the dataset
    shufflingIndices = np.random.permutation(balancedY.shape[0])
    shuffledBalancedX = balancedX[shufflingIndices]
    shuffledBalancedY = balancedY[shufflingIndices]

    return shuffledBalancedY, shuffledBalancedX

def makeTrainingData(x):
    ''' Function filling the invalid values with the mean (zero), and adding a dummy variable'''
    xClean = np.ma.filled(x,fill_value=0) # Replace the invalid entries by zeros (aka the mean)
    tx = np.c_[np.ones(xClean.shape[0]),xClean] # Adding a dummy feature
    return np.nan_to_num(tx) # For some reason, not all NaN values were filled with zeros, this should rectify that problem