def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    >>> sigmoid(np.array([0.1]))
    array([0.52497919])
    >>> sigmoid(np.array([0.1, 0.1]))
    array([0.52497919, 0.52497919])
    """
    phi = (1 + np.exp(-t))**(-1)
    return phi
    #raise NotImplementedError



def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(4).reshape(2, 2)
    >>> w = np.c_[[2., 3.]]
    >>> round(calculate_loss(y, tx, w), 8)
    1.52429481
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    #print(np.shape(tx),np.shape(y))
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    
    first_term = y.T@np.log(sigmoid(tx@w))
    log_term = (1-y).T@np.log(1-sigmoid(tx@w))
    loss = -(1/np.shape(y)[0])*np.sum(first_term+log_term)
    return loss
    # ***************************************************
    #raise NotImplementedError




def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> calculate_gradient(y, tx, w)
    array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    #grad=1/y.shape[0]*(-tx.T@y+tx.T@sigmoid(tx@w))
                  
    grad = (1/np.shape(y)[0])*tx.T@(sigmoid(tx@w)-y)
    return grad
    # ***************************************************
    raise NotImplementedError("Calculate gradient")


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> gamma = 0.1
    >>> loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    >>> round(loss, 8)
    0.62137268
    >>> w
    array([[0.11037076],
           [0.17932896],
           [0.24828716]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    grad = calculate_gradient(y, tx, w)
    loss = calculate_loss(y,tx,w)
    w_new = w-gamma*grad
    
    return loss,w_new
    # ***************************************************
    raise NotImplementedError

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
    sig = sigmoid(tx@w)
    
    S = np.diag(sig-sig@sig.T)
    
    print('tx ', np.shape(tx), 'diag ', np.shape((S)),'tx ', np.shape(tx))
    hess = (1/np.shape(y)[0])*(tx@np.diag(S)@tx)
    
    return hess
    # ***************************************************
    raise NotImplementedError



def logistic_regression(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
        hessian: shape=(D, D)

    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> loss, gradient, hessian = logistic_regression(y, tx, w)
    >>> round(loss, 8)
    0.62137268
    >>> gradient, hessian
    (array([[-0.10370763],
           [ 0.2067104 ],
           [ 0.51712843]]), array([[0.28961235, 0.3861498 , 0.48268724],
           [0.3861498 , 0.62182124, 0.85749269],
           [0.48268724, 0.85749269, 1.23229813]]))
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    #hess = calculate_hessian(y,tx,w)
    return loss,grad#,hess
    # ***************************************************
    #raise NotImplementedError



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
    # return loss, gradient, and Hessian: TODO
    loss,grad = logistic_regression(y, tx, w)
    #print(loss)
    #print('grad ',(np.shape(grad)),'lamb ', (lambda_),'w ',np.shape(np.abs(w)*2))
    1/y.shape[0]*(-tx.T@y+tx.T@sigmoid(tx@w)+ lambda_*abs(w)*2*y.shape[0])
    
    grad_pen = grad+lambda_*np.abs(w)*2
    loss_pen = loss+lambda_*np.sum(w.T@w)
    
    return loss,grad_pen
    # ***************************************************
    #raise NotImplementedError



def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    >>> np.set_printoptions(8)
    >>> y = np.c_[[0., 1.]]
    >>> tx = np.arange(6).reshape(2, 3)
    >>> w = np.array([[0.1], [0.2], [0.3]])
    >>> lambda_ = 0.1
    >>> gamma = 0.1
    >>> loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
    >>> round(loss, 8)
    0.63537268
    >>> w
    array([[0.10837076],
           [0.17532896],
           [0.24228716]])
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    #loss,grad,hess = logistic_regression(y, tx, w)
    # ***************************************************
    #raise NotImplementedError
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    #loss,grad,hess = logistic_regression(y, tx, w)
    loss,grad_pen = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*(grad_pen)
    # ***************************************************
    #raise NotImplementedError
    return w, loss



def max_k_fold_cross_valid_sets(y,k_fold):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    #np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    #print(np.array(k_indices))
    return np.array(k_indices)

def max_cross_valid(y, x, k_fold, initial_w,max_iter,gamma,lambda_, regressionFunction=learning_by_penalized_gradient,lossFunction=calculate_loss):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train: TODO
    loss_train_test = []
    loss_tr_arr = []
    loss_te_arr = []
    for k in range(k_fold):
        k_indices = max_k_fold_cross_valid_sets(y, k_fold)
        #print(np.shape(k_indices))
        test_ind = k_indices[k]
        train_ind = (k_indices[np.arange(len(k_indices))!=k]).flatten()

        x_train = x[train_ind]
        x_test = x[test_ind]

        y_train = y[train_ind].flatten()
        y_test = y[test_ind]
        #print('x_train: ', np.shape(x_train),'y_train: ',np.shape(y_train))

        if regressionFunction==reg_logistic_regression:
            weights, loss_train = regressionFunction(y_train, x_train,lambda_, initial_w, max_iter, gamma)
            
        else:
            weights, loss_train = regressionFunction(y_train, x_train, initial_w, max_iter, gamma)
            
        #print(weights)
        loss_tr_test = lossFunction(y_train, x_train, weights)
        loss_test = lossFunction(y_test, x_test, weights)
       
        loss_train_test.append(loss_tr_test)
        loss_tr_arr.append(loss_train/k_fold)
        loss_te_arr.append(loss_test/k_fold)
        print(f'Run {k+1} yielded a loss improvement from {loss_train} to {loss_test}')
        print('______________________')
    
    
  
    return loss_tr_arr,loss_te_arr,loss_train_test
   