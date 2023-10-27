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
    ''' Compute the cost by negative log likelihood.
    Args:
        y: (N,) array with labels
        tx: (N,D) array with samples and their features
        w: (D,) array with the parameters
    Returns:
        float non-negative loss
    '''
    #return np.sum(np.log( 1 + np.exp(-y * (tx@w))))
    first_term = y.T@np.log(logistic(tx@w))
    log_term = (1-y).T@np.log(1-logistic(tx@w))
    return -np.sum(first_term+log_term) / y.shape[0]
    #return - np.sum( y * np.log(logistic(tx @ w)) + (1-y)* np.log(1-logistic(tx @ w)) ) / y.shape[0]

def logistic_gradient(y, tx, w):
    ''' Compute the gradient of the logistic loss
    Args:
        y: (N,) array with labels
        tx: (N,D) array with samples and their features
        w: (D,) array with the parameters
    Returns:
        (D, 1) array with the gradient of the logistic loss with respect to the parameters
    '''
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
        w -= gamma * logistic_gradient(y,tx,w)
    
    return w, logistic_loss(y,tx,w)

########## Regularized logistic regression using gradient descent or SGD (y ∈ {0,1}, with regularization term λ∥w∥2) ##########

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar
    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """
    grad = logistic_gradient(y,tx,w)
    loss = logistic_loss(y,tx,w)
    
    loss_pen = loss+lambda_*np.sum(w.T@w)
    grad_pen = grad+lambda_*np.abs(w)*2
        
    return loss_pen,grad_pen

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

    for n in range(max_iters):
        #loss_pen, grad_pen = penalized_logistic_regression(y, tx, w, lambda_)
        w -= gamma * logistic_gradient(y,tx,w) + 2*lambda_*np.abs(w) # Updating the paramters by the gradient with the regularization term
    
    loss = logistic_loss(y,tx,w)
    return w, loss

def calculate_hessian(y, tx, w):
    """ Return the Hessian of the loss function.
    Args:
        y:  (N, 1)
        tx: (N, D)
        w:  (D, 1)
    Returns:
        (D,D) array of the hessian matrix
    logisticDiag = np.diag((logistic * (1-logistic)).flatten()) 
    """
    sig = logistic(tx@w)
    
    S = np.diag(sig-sig@sig.T)
    
    hess = (1/np.shape(y)[0])*(tx.T@np.diag(S)@tx)
    
    return hess


########## Loading data ##########

def loadData(dataPath):
    ''' Loads data and returns it as masked numpy array. A masked array contains information about which values are invalid, ensuring methods like .mean() ignores the masked values
    Args:
        dataPath: The file path of the data
    Returns:
        data: (N,d) masked numpy array, where N is the number of samples, and d is the dimension of the x values, or 1 if the data in question are the labels
        header: (d,) array with the column names
    '''
    data = np.genfromtxt(dataPath, delimiter=',', skip_header=1, dtype=float, usemask=True) # Loading the data as a masked array (with usemask=True), skipping the header, and specifying that the values are floats
    header = np.genfromtxt(dataPath, delimiter=',', dtype=str, max_rows=1) # Loading the first row of the csv file, i.e. the header
    return data , header

def loadTrainingData():
    ''' Loads the medical training data and nothing else. Wrapper function
    Returns:
        X, xHeader, Y, yHeader, indexedX, indexedXheader, indexedY, indexedYheader
    '''
    x, xHeader = loadData('./Data/x_train.csv')
    y, yHeader = loadData('./Data/y_train.csv')
    y[y == -1] = 0
    unIndexedX, unIndexedXHeader = x[:,1:], xHeader[1:]
    unIndexedY, unIndexedYHeader = y[:,1:], yHeader[1:]

    print(f'Data successfully loaded, there are {unIndexedX.shape[1]} features and {y.shape[0]} samples, the shapes of the unindexed data is:\ny: {unIndexedY.shape}, x: {unIndexedX.shape}')

    return unIndexedX, unIndexedXHeader, unIndexedY.flatten(), unIndexedYHeader, x, xHeader, y, yHeader

########## Data cleaning functions ##########

def removeBadFeatures(x,xHeader,threshold=0.7):
    ''' Function checking how many features have more than {threshold} parts valid entries, and removing the 'bad' features with less than {threshold} valid entries
    Args:
        x: (N,d) array of the dataset
        xHeader: (d,) array of the header for the dataset
        threshold: float between 0 and 1
    Returns:
        (N,d-b) array of the new dataset, where b is the number of features with less than threshold parts valid entries
        (d-b,) array of the new header, where b is the number of features with less than threshold parts valid entries
        (b,) array with the removed features
    '''
    # Counting the number of valid values for each feature, and calculating the percentage of valid entries
    validFeatureVals = x.count(axis=0) # The number of valid entries for each feature
    validFeatureValsPercent = validFeatureVals/x.shape[0] # The percentage of valid entries for each feature

    # Finding the indices of all the features with number of features above and below a threeshold
    featureIndicesAboveThreeshold = np.argwhere(validFeatureValsPercent > threshold).flatten() # Finding the indices where there are more than threeshold percent valid entries
    featureIndicesBeneathThreeshold = np.argwhere(validFeatureValsPercent < threshold).flatten()

    # Printing the good vs bad features
    print(f'For a threshold of {threshold}, there are {len(featureIndicesAboveThreeshold)} good features, and {x.shape[1]-len(featureIndicesAboveThreeshold)} bad features')

    # Removing the features that appears less than {threeshold} of the time, and returning the others
    return x[:,featureIndicesAboveThreeshold], xHeader[featureIndicesAboveThreeshold], xHeader[featureIndicesBeneathThreeshold]


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
    negativeCases = np.where(y == 0)[0]
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

    print(f'Created a balanced subset of the data, with {2*smallestSubsetLength} samples, {smallestSubsetLength} each of positive and negative samples')
    
    ### Models trained from this dataset will overestimate the probability of positive (or negative) cases,
    ### therefore we should use some bayesian probabilities to update probabilities
    prior = 2*smallestSubsetLength/len(y) # The probability of a random sample being in the balanced data
    return shuffledBalancedY, shuffledBalancedX, prior

def makeTrainingData(x):
    ''' Function filling the invalid values with the mean (zero), and adding a dummy variable'''
    xClean = np.ma.filled(x,fill_value=0) # Replace the invalid entries by zeros (aka the mean)
    tx = np.c_[np.ones(xClean.shape[0]),xClean] # Adding a dummy feature
    print('Added dummy variable and replaced invalid entries with zeros')
    return np.nan_to_num(tx) # For some reason, not all NaN values were filled with zeros, this should rectify that problem

def detectOutliers(y, x, outlierThreshold=10):
    xStandardized = standardizeData(x)

    stdAwayFromMean = np.abs(xStandardized)

    # Assuming normal distribution, approx 68% of the data falls within 1 std, 95% within 2 std, 99.7% within 3 std
    # Therefore one may consider any entries in stdAwayFromMean > {outlierThreeshold=3} to be outliers - except that the data is not neccesarily normally distributed, so the threshold should be higher
    inlierRows = np.where(np.all(stdAwayFromMean < outlierThreshold, axis=1))[0]
    
    outlierRows = np.where(np.any(stdAwayFromMean > outlierThreshold, axis=1))[0]
    #print(outlierRows.shape)
    #print(inlierRows.shape)
    #print(x.shape)

    print(f'Removed {len(outlierRows)} samples with outliers more than {outlierThreshold} standard deviations from the mean. There remains {len(inlierRows)} samples in the dataset.')

    return y[inlierRows], x[inlierRows]

def dataCleaning(y,x,xHeader,featureThreshold=0.7,acceptableMissingValues=5):
    ''' Function for removing features with to few valid entries, samples with to few valid entries, samples with outliers, and standardizing the data.
    Args: 
        y: (N,) array of the labels
        x: (N,d) array of the samples and their features
        xHeader: (d,) array of the feature titles
        featureThreshold: float between 0 and 1 of the percent of a features entries must be valid to keep the feature
        acceptableMissingValues: integer of the number of entries for each feature that may be invalid entries
    Returns:
        (N-c) array of labels, where c is the number of samples with too many missing entries, plus the number of samples with outliers
        (N-c,d-b) array of the data, where b is the number of removed features
        (d-b,) array of the feature titles
        (b,) array of the removed feature titles
    '''
    # Removing bad features and samples
    xFeaturesRemoved, xHeaderFeaturesRemoved, removedFeatures = removeBadFeatures(x,xHeader,featureThreshold)
    ySamplesRemoved, xSamplesRemoved = removeBadSamples(y,xFeaturesRemoved,acceptableMissingValues)
    print(f'The number of invalid entries remaing in the dataset is {xSamplesRemoved.size - xSamplesRemoved.count()}\nThat is {(xSamplesRemoved.size - xSamplesRemoved.count())/xSamplesRemoved.size} parts of the whole dataset')
    
    # Removing outliers
    yOutliersRemoved, xOutliersRemoved = detectOutliers(ySamplesRemoved,xSamplesRemoved)

    # Standardizing the data by subtraction of the mean and dividing by the standard deviation
    xStandardized = standardizeData(xOutliersRemoved)
    print('Standardized data by subtracting the mean and dividing by the standard deviation')

    return yOutliersRemoved, xStandardized, xHeaderFeaturesRemoved, removedFeatures



########## K Fold Cross Validation #########

def k_fold_cross_validation_sets(y,x,K):
    ''' Function for making K separate training sets out of the provided dataset
    Args:
        y: (N,) array of the labels
        x: (N,d) array of the data with its features
        K: Integer number of separate trainingsets
    Yields:
        y_k: (N/K,) array of the chosen labels. N/K is N//K + 1 for the first sets, and N//K for the rest of the sets
        x_k: (N/K,d) array of the data
    '''
    N = len(y)      # Saving the number of samples as an integer
    batchSize = N // K  # Calculating the batch size
    residual = N - K*batchSize  # Checking how many samples would not be included in sets of size N//K

    indices = np.random.permutation(N) # Randomly permuted indices of the provided dataset
    
    for k in range(K):
        if k < residual: # If the samples 'in' the residual has not 'been used', we include them
            indices_k = indices[k*(batchSize+1):(k+1)*(batchSize+1)] # Indices of the elements for each k batch. Here included one extra samples 'from' the residual
        else:
            indices_k = indices[residual+k*batchSize:residual+(k+1)*batchSize] # Indices of the elements for each k batch
        
        yield y[indices_k], x[indices_k] # Yield returns the first set, and next time the function is called the code continues, so the for loop repeats and yields the next set

def k_fold_cross_validation(y,tx,K,initial_w,max_iters,gamma, regressionFunction=logistic_regression, lossFunction=logistic_loss):
    ''' Performing regression on K separate subsets of the provided training set, and returning the average parameters
    Args:
        y: (N,) array of the labels
        tx: (N,d) array of the data and its features
        initital_w: (d,) array with some initialization of the parameters
        max_iters: integer of the maximum iterations per regression
        gamma: float of the step size
        regressionFunction: The function of the chosen type of regression
        lossFunction: The function of the chosen type of loss
    Returns:
        w_avg: (d,) array of the resultant parameters averaged over the cross validation runs
    ''' 
    crossValidationSets = k_fold_cross_validation_sets(y,tx,K)
    
    w, loss = np.zeros((K,tx.shape[1])), np.zeros(K)
    
    for k in range(K):
        y_k, tx_k = next(crossValidationSets)

        w[k], loss[k] = regressionFunction(y_k, tx_k, initial_w, max_iters, gamma)

        print(f'Run {k+1} yielded a loss improvement from {lossFunction(y_k,tx_k,initial_w)} to {lossFunction(y_k,tx_k,w[k])}')
    w_avg = np.sum(w,axis=0) / K
    
    print(f'''-----------------------------------------------------------------------------------------
Averaging the parameters, the loss improves from {lossFunction(y,tx,initial_w)} to {lossFunction(y,tx,w_avg)}''')
    return w_avg, lossFunction(y_k,tx_k,initial_w)


########## Making final predictions ##########

def makePredictions(w,xTest,xHeader,xHeaderFeaturesRemoved, prior=1.0):
    ''' Function making predictions based on provided parameters and data
    Args:
        w: (d,) array with the parameters
        x: (N,D) array with the data
        xHeader: (D,) array with all the features
        xHeader: (d,) array with the features that are actually used
        prior: float denoting the probability of a random sample being in the model training data
    Returns:
        (N,) boolean array of the predictions
    '''
    standardX = standardizeData(xTest)
    removedFeaturesX = standardX[:,np.nonzero(np.isin(xHeader, xHeaderFeaturesRemoved))[0]]
    predictionSet = makeTrainingData(removedFeaturesX)
    probabilities = prior * logistic(predictionSet@w) # The prob of the model being applicable times the prob from the model
    return (np.sign(probabilities-0.5)+1)/2 # Shifting the probs to be negative for negative preds, and vice versa, taking the sign, shifting the preds up to be zero or two, diving by to so the preds are zero or one


######## Calculating the recall of our prediction #####################

def calculate_recall(y_true, y_predicted):
    true_positives = 0
    false_positives = 0
    for i in y_true:
        if i == 1 and y_predicted == 1:
            true_positives += 1
        if i == 1 and y_predicted == 0:
            false_positives += 1
    return true_positives / (true_positives + false_positives)