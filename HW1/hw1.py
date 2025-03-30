###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

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
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(y)
    X_std = np.std(X, axis=0)
    Y_std = np.std(y)
    X_standardized = (X - X_mean) / X_std
    Y_standardized = (y - Y_mean) / Y_std
    
    return X_standardized, Y_standardized

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
    
    J = np.sum(((X @ theta) - y) ** 2) / (2 * len(y))
    return J

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
    ###########################################################################
    for t in range(num_iters):
        theta = theta - (eta / len(y)) * (X.T @ (X @ theta - y))
        J = compute_loss(X, y, theta)
        J_history.append(J)
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
    
    pinv_theta = (np.linalg.inv(X.T @ X)) @ X.T @ y
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
   
    for t in range(max_iter):
        theta = theta - (eta / len(y)) * (X.T @ (X @ theta - y))
        J = compute_loss(X, y, theta)
        J_history.append(J)
        if t > 0 and abs(J - J_history[-1]) < epsilon:
            break
            
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
    
    for eta in etas:
        np.random.seed(42)
        theta = np.random.random(size=X_train.shape[1])
        theta_itr, J_history = gradient_descent_stop_condition(X_train, y_train, theta, eta, iterations)
        eta_dict[eta] = compute_loss(X_val, y_val, theta_itr)
        
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
    # number of features
    m = X_train.shape[1]
    
    # Apply bias trick inside the function
    X_train_with_bias = apply_bias_trick(X_train)
    X_val_with_bias = apply_bias_trick(X_val)
    
    while len(selected_features) < 5:
        best_feature = None
        best_loss = float("inf")

        for col_feat in range(m):
            if col_feat in selected_features:
                continue
                
            # Select current features plus new candidate
            current_features = selected_features + [col_feat]
            
            # Create subsets with selected features (plus bias)
            X_train_subset = np.column_stack([X_train_with_bias[:,0], X_train[:,current_features]])  # Keep bias column
            X_val_subset = np.column_stack([X_val_with_bias[:,0], X_val[:,current_features]])
            
            # Initialize theta for this feature set
            theta_i = np.random.random(len(current_features) + 1)  # +1 for bias
            
            # Train model with these features
            theta_itr, _ = gradient_descent_stop_condition(X_train_subset, y_train, theta_i, best_eta, iterations)
            
            # Evaluate on validation set
            loss = compute_loss(X_val_subset, y_val, theta_itr)

            if loss < best_loss:
                best_loss = loss
                best_feature = col_feat
        selected_features.append(best_feature)

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
    
    # Create squared features
    df_squares = df_poly ** 2
    df_squares.columns = [f"{col}^2" for col in df_squares.columns]
    
    # Combine original and squared features
    df_poly = pd.concat([df_poly, df_squares], axis=1)
    
    # Create pairwise interaction features
    pairwise_features = {}
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            f1, f2 = df.columns[i], df.columns[j]
            pairwise_features[f"{f1}*{f2}"] = df[f1] * df[f2]
            
    df_pairs = pd.DataFrame(pairwise_features)
    
    # Combine all features
    df_poly = pd.concat([df_poly, df_pairs], axis=1)
    
    return df_poly
