# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform Standardization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The Standardized input data.
    - y: The Standardized true labels.
    """
    normalisedX = preprocessOnOne(X)
    normalisedY = preprocessOnOne(y)

    
    return normalisedX, normalisedY

def preprocessOnOne(X): 
    """
    performs the standardization (z-score normalisation) on a single input array X
    returns the normalised version of the input
    Note: This is the normal distrubtion so most values will now be between [3,-3]
    """
    n = X.size
    meanx = np.sum(X)/n

    sqrdDiffx = np.sum((X - meanx) ** 2) 
    stdx = np.sqrt(sqrdDiffx / n)
    normalisedX = (X - meanx) / stdx 

    return normalisedX

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (n instances over p features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (n instances over p+1).
    """
    X_bias = np.column_stack((np.ones(X.shape[0]), X))
    return X_bias

def compute_loss(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the loss associated with the current set of parameters (single number).
    """
    # First compute the prediction function
    n = X.shape[0]
    thetaT = theta.reshape(-1,1)
    yT = y.reshape(-1,1)
    y_hat = X @ thetaT
    diff = y_hat - yT

    if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
        return np.inf
    J = np.sum((y_hat - yT) ** 2 / (2*n))

    """
    This is to check your work and understand wether the correct computations are occuring
    

    print("First few predictions:", y_hat[:5].flatten())
    print("First few actuals:", y[:5].flatten())
    print("Squared Errors:", ((y_hat - y)[:5] ** 2).flatten())
    print("MSE (J):", J)
    """
    return J

def compute_gradient(X, y, theta):
    n = y.shape[0]
    grad = (X.T @ (X @ theta - y)) / n
    if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
        return np.full_like(theta, np.inf)
    return grad

def gradient_descent(X, y, theta, eta, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    for i in range(num_iters):
        grad = compute_gradient(X, y, theta)
        theta = theta - eta*grad
        loss = compute_loss(X, y, theta)

        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print(f"Gradient Descent diverged at iteration {i}")
            break
        J_history.append(loss)

    return theta, J_history
    


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    X_T = X.T
    pinv_theta = []
    try: 
        pinv_theta = np.linalg.solve(X_T @ X, X_T @ y) 
    except np.linalg.LinAlgError: 
        lambda_reg = 1e-8
        pinv_theta = np.linalg.solve(X_T @ X + lambda_reg * np.eye(X.shape[1]), X_T @ y)

    return pinv_theta

def gradient_descent_stop_condition(X, y, theta, eta, max_iter, epsilon=1e-8):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than epsilon. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (n instances over p features).
    - y: True labels (n instances).
    - theta: The parameters (weights) of the model being learned.
    - eta: The learning rate of your model.
    - max_iter: The maximum number of iterations.
    - epsilon: The threshold for the improvement of the loss value.
    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the loss value in every iteration
    curr_iters = 0 
    prev_loss = np.inf
    for i in range(max_iter):
        grad = compute_gradient(X, y, theta)
        theta = theta - eta*grad

        loss = compute_loss(X, y, theta)
        J_history.append(loss)

        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print(f"Stopping early at eta={eta}: invalid theta")
            break   
        if abs(prev_loss - loss) < epsilon:
            break

        prev_loss = loss

    return theta, J_history

def find_best_learning_rate(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of eta and train a model using 
    the training dataset. Maintain a python dictionary with eta as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - eta_dict: A python dictionary - {eta_value : validation_loss}
    """
    
    etas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    eta_dict = {} # {eta_value: validation_loss}
    np.random.seed(42)

    """ Training """
    for eta in etas:
        theta = np.random.random(size=X_train.shape[1]) 
        theta, _  = gradient_descent_stop_condition(X_train, y_train, theta, eta, iterations, epsilon=1e-8)
        """Validating"""
        val_loss = compute_loss(X_val, y_val, theta)
        eta_dict[eta] = val_loss if np.isfinite(val_loss) else np.inf
    return eta_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_eta, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_eta: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best eta value.             #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (n instances over p features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly