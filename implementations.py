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

def logistic_function(z):
    ''' The logistic function dependent on z, often called the sigmoid function
    Args:
        z: Array-like
    Returns:
        Array-like of same dimensions as the input, 
    '''
    return 1/(1+np.exp(-z)) # Calculating and returning the logistic function

def compute_logistic_loss(y, tx, w):
    ''' Computing the logistic loss, defined as sum_{i=1}^n(y_i w.T x_i + log(logistic(-w.T x_i)))
    Args:
        y: (N,) array of labels
        tx: (N,d) array of samples and their features
        w: (d,) arry of the parameters
    Returns:
        Float denoting the logistic loss
    '''
    first_term = y.T@np.log(logistic_function(tx@w))
    log_term = (1-y).T@np.log(1-logistic_function(tx@w))
    loss = -(1/np.shape(y)[0])*np.sum(first_term+log_term)
    return loss
    #return np.sum(y * w.T@tx  +  np.log10(logistic_function(-w.T@tx)))

def compute_logistic_gradient(y, tx, w):
    ''' Computing the gradient of the logistic function, sum_{i=1}^n((y_i - logistic(w.T x_i)) x_i)
    Args:
        y: (N,) array of labels
        tx: (N,d) array of samples and their features
        w: (d,) arry of the parameters
    Returns:
        (d,) array with the logistic gradient
    '''
    grad = (1/np.shape(y)[0])*tx.T@(logistic_function(tx@w)-y)
    return grad
    #return np.sum((y-logistic_function(w.T@tx)) * tx, axis=0)

def logistic_regression(y, tx, initial_w, max_iters, gamma): # Required function #5
    ''' Gradient descent with logistic loss
    Args:
        y: (N,) array of labels
        tx: (N,d) array of samples and their features
        initial_w: (d,) array of initialization values for the parameters
        max_iters: Integer denoting the maximum number of GD steps
        gamma: Float denoting the step size
    Returns:
        w: (d,) array with the final parameters
        loss: Float denoting the final logistic loss
    '''
    # Initializing w
    w = np.array(initial_w,dtype=float)

    for n in range(max_iters):
        grad = compute_logistic_gradient(y,tx,w)
        w -= gamma * grad
    
    return w, compute_logistic_loss(y,tx,w)




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

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    grad = compute_logistic_gradient(y,tx,w)
    loss = compute_logistic_loss(y,tx,w)
    
    loss_pen = loss+lambda_*np.sum(w.T@w)
    grad_pen = grad+lambda_*np.abs(w)*2
        
    return loss_pen,grad_pen

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma): # Required function #6
    """return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> gradient
    array([[-0.08370763],
           [ 0.2467104 ],
           [ 0.57712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    w = np.array(initial_w,dtype=float)

    for n in range(max_iters):
        loss_pen, grad_pen = penalized_logistic_regression(y, tx, w, lambda_)
        w -= gamma * grad_pen
    
    return w, loss_pen
    #pass

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a hessian matrix of shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_hessian(y, tx, w)
    array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate Hessian: TODO
    #print('shape tx: ',np.shape(tx),'shape w: ',np.shape(w), np.shape(tx@w),np.shape(sigmoid(tx@w)))
    sig = logistic_function(tx@w)
    
    S = np.diag(sig-sig@sig.T)
    
    hess = (1/np.shape(y)[0])*(tx.T@np.diag(S)@tx)
    
    return hess