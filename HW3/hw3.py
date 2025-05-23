import numpy as np

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = 0.0
    for sample in k:
        factorial = 1
        for i in range(1, sample + 1):
            factorial *= i
        log_p += sample * np.log(rate) - rate - np.log(factorial)
    return log_p

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = np.sum(samples)/ len(samples)
    return mean

def possion_confidence_interval(lambda_mle, n, alpha=0.05):
    """
    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # Use norm.ppf to compute the inverse of the normal CDF
    from scipy.stats import norm
    lower_bound = lambda_mle - np.sqrt(lambda_mle/n) * norm.ppf((1 - alpha) / 2)
    upper_bound = lambda_mle + np.sqrt(lambda_mle/n) * norm.ppf((1 - alpha) / 2)

    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.array([poisson_log_pmf(samples, rate)for rate in rates])
    return likelihoods

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.2,
            (0, 1): 0.1,
            (1, 0): 0.25,
            (1, 1): 0.45
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.3,
            (0, 1): 0.1,
            (1, 0): 0.2,
            (1, 1): 0.4
        }  # P(X=x, C=c)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.2,
            (1, 0): 0.35,
            (1, 1): 0.3
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.09,
            (0, 0, 1): 0.04,
            (0, 1, 0): 0.21,
            (0, 1, 1): 0.06,
            (1, 0, 0): 0.06,
            (1, 0, 1): 0.16,
            (1, 1, 0): 0.14,
            (1, 1, 1): 0.24,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y


    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C


def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    p = (1 / np.sqrt(2 * np.pi * (std ** 2))) * np.e ** (-((x - mean) ** 2 / (2 * (std ** 2))))
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        These distributions are fit to specified training data.
        
        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class label to calculate the class conditionals for.
        """
        self.dataset = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.mean = np.mean(self.dataset[:, :-1], axis=0)
        self.std = np.std(self.dataset[:, :-1], axis=0)
        self.size = dataset.shape[0]

    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        return self.dataset.shape[0] / self.size
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific classc conditionals fitted to the training data.
        """
        likelihood = 1.0
        for i, feature in enumerate(x):
            likelihood *= normal_pdf(feature, self.mean[i], self.std[i])
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint = self.get_prior() * self.get_instance_likelihood(x)
        print(joint)
        return self.get_prior() * self.get_instance_likelihood(x)

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        self.obj0 = ccd0
        self.obj1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        pred = 0 if self.obj0.get_instance_joint_prob(x) > self.obj1.get_instance_joint_prob(x) else 1
        return pred
    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal desnity function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = mean.shape[0]
    diff = x - mean
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))
    pdf = (np.e ** (-0.5 * (diff.T @ np.linalg.inv(cov) @ diff))) / denominator
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The mean and cov matrix should be computed from a given training data set
        (You can use the numpy function np.cov to compute the sample covarianve matrix).
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the parameters for.
        """
        self.c_data = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.mean = np.mean(self.c_data[:, :-1], axis=0)
        self.size = dataset.shape[0]

        diff = self.c_data[:, :-1] - self.mean
        self.cov = diff.T @ diff / diff.shape[0]
        
        
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        prior = self.c_data.shape[0] / self.size
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate classc conditionals fitted to the training data.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        print(joint_prob)
        return joint_prob


def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicits the class label from a feature vector.
        
    Ouput
        - Accuracy = #Correctly Classified / number of test samples
    """
    count = 0
    for sample in test_set:
        res = map_classifier.predict(sample[:-1])
        if res == sample[-1]:
            count += 1
    return count / test_set.shape[0]

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the probabilites for a discrete naive bayes
        class conditional distribution for a given class label.
        The probabilites of each feature-specific class conditional
        are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class label to calculate the probabilities for.
        """
        self.c_dataset = dataset[dataset[:, -1] == class_value]
        self.class_value = class_value
        self.size = dataset.shape[0]
        self.num_classes = np.unique(dataset[:, -1]).size
        self.c_dataset_size = self.c_dataset.shape[0]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class, as computed from the training data.
        """
        prior = None
        prior = (self.c_dataset_size + 1) / (self.size + self.num_classes)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conidtionals fitted to the training data.
        """
        likelihood = 1.0
        for j, v in enumerate(x):
            count_v = np.sum(self.c_dataset[:, j] == v)
            likelihood *= ((count_v + 1) / (self.c_dataset_size + np.unique(v).size))
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob
