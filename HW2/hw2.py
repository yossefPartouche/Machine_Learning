import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    # Extract the label set 
    label = data[:, -1]
    # count freq of each class 
    _, counts = np.unique(label, return_counts=True)
    total_count = label.shape[0]
    # convert to proportion/probability
    proportions = counts/total_count
    # apply Gini Formula
    gini = 1 - np.sum(proportions**2)
    return gini

def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    # Extract the label set
    label = data[:, -1]
    # count freq of each class
    _, counts = np.unique(label,return_counts=True)
    total_count = label.shape[0]
    # convert to proportions/probabilities
    p= counts/total_count
    # apply Entropy Formula
    entropy = np.sum(p*np.log2(p))
    entropy = - entropy

    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the data instances associated with the node
        self.terminal = False # True iff node is a leaf
        self.feature = feature # column index of feature/attribute used for splitting the node
        self.pred = self.calc_node_pred() # the class prediction associated with the node
        self.depth = depth # the depth of the node
        self.children = [] # the children of the node (array of DecisionNode objects)
        self.children_values = [] # the value associated with each child for the feature used for splitting the node
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to use for measuring goodness of a split
        self.gain_ratio = gain_ratio # True iff GainRatio is used to score features
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node's prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        # extract the label set 
        label = self.data[:, -1]
        # count freq of each class
        values, count = np.unique(label, return_counts=True)
        pred = values[np.argmax(count)]
        return pred
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
            according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        # store number of samples (|S|)
        s_size = self.data.shape[0]
        # Extract the feature values and their sizes
        feature_values, sizes = np.unique(self.data[:, feature], return_counts=True)
        
        # Gain ratio flag is set to false
        if not self.gain_ratio:
            # calculate impurity of the feature set as a whole
            s_impurity = self.impurity_func(self.data)
            imp_weighted = 0.0
            for val, subset_size in zip(feature_values, sizes):
                # Create a subset of the data for each feature value
                subset = self.data[self.data[:, feature] == val]
                groups.update({val: subset})
                # Calculate the impurity of the subset
                subset_impurity = self.impurity_func(subset) ##### must use impurity_func not entropy
                # Calculate the weighted impurity
                imp_weighted += (subset_size / s_size) * subset_impurity
            goodness = s_impurity - imp_weighted
        # Gain ratio flag is set to true
        else:
            info_gain = calc_entropy(self.data)
            split_info = 0
            for val, subset_size in zip(feature_values, sizes):
                # Create a subset of the data for each feature value
                subset = self.data[self.data[:, feature] == val]
                groups.update({val: subset})
                split_info += (subset_size / s_size) * np.log2(subset_size / s_size)
            goodness = info_gain / (- split_info)

        return goodness, groups
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        goodness, _ = self.goodness_of_split(self.feature)
        size = self.data.shape[0]
        self.feature_importance = (size/n_total_sample)*goodness
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.
        This function has no return value
        """
        # Check if the node is a leaf
        if self.depth >= self.max_depth:
            self.terminal = True
            return
        
        # Check if the node is a leaf
        best_goodness = -1
        best_feature = None
        best_groups = None
        num_features = self.data.shape[1] - 1
        
        # find best feature to split
        for col_idx in range(num_features):
            goodness, groups = self.goodness_of_split(col_idx) # gos recieves feature col idx as input
            
            # Check if the goodness is larger than the current best
            if goodness > best_goodness:
                best_goodness = goodness
                best_feature = col_idx
                best_groups = groups
        self.feature = best_feature

        # The chi-squared test should be added here
        if best_feature is not None and self.chi < 1:
            dog = len(best_groups)-1

            #check if we can find a threshold
            if dog in chi_table and self.chi in chi_table[dog]:
                threshold = chi_table[dog][self.chi]

                if self.chi_square_test(best_groups) < threshold:
                    self.terminal = True
                    return

        
        for val, subset in best_groups.items():
            # Create a new DecisionNode for each subset
            child = DecisionNode(
                data=subset,
                impurity_func=self.impurity_func,
                depth=self.depth + 1,
                chi=self.chi,
                max_depth=self.max_depth,
                gain_ratio=self.gain_ratio
            )
            # Add the child to the current node
            self.add_child(child, val)

    def chi_square_test(self, groups):
        chi_square_value = 0

        total_samples = self.data.shape[0]
        class_values, class_counts = np.unique(self.data[:, -1], return_counts=True)
        overall_class_probs = {class_val: count/total_samples for class_val, count in zip(class_values, class_counts)}

        #Calculate chi-square statistic
        for val, subset in groups.items():
            subset_size = subset.shape[0]

            # For each class
            for class_val in class_values:
                # Observed count
                observed = np.sum(subset[:, -1] == class_val)
                
                # Expected count (what would be expected by chance)
                expected = subset_size * overall_class_probs[class_val]
                
                # Add to chi-square statistic if expected count is non-zero
                if expected > 0:
                    chi_square_value += ((observed - expected) ** 2) / expected
        return chi_square_value

            
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the training data used to construct the tree
        self.root = None # the root node of the tree
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.chi = chi # the P-value cutoff used for chi square pruning
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.gain_ratio = gain_ratio #
        
    def depth(self):
        return self.root.depth

    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        self.root = DecisionNode(
            data=self.data,
            impurity_func=self.impurity_func,
            chi=self.chi,
            max_depth=self.max_depth,
            gain_ratio=self.gain_ratio
        )
        # Initialize a queue with the root node
        queue = [self.root]
        
        # While queue is not empty
        while len(queue) > 0:
            # Dequeue the first node
            node = queue.pop(0)
            labels = node.data[:, -1]

            # If all samples have same class label y, designate node as a leaf
            if len(np.unique(labels)) == 1:
                node.terminal = True
                continue
            
            node.split()
            node.calc_feature_importance(self.data.shape[0])
            
            # If the node is not a leaf, add its children to the queue
            if not node.terminal:
                for child in node.children:
                    queue.append(child)

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        node = self.root
        
        # Traverse the tree until a leaf node is reached
        while not node.terminal:
            # Get the feature value for the current node
            feature_value = instance[node.feature]
            # Find the child node corresponding to the feature value
            if feature_value in node.children_values:
                index = node.children_values.index(feature_value)
                node = node.children[index]
            else:
                break

        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        count_correct = 0

        for instance in dataset:
            # Predict the label of the row
            pred = self.predict(instance)
            # Check if the prediction is correct
            if pred == instance[-1]:
                count_correct += 1

        accuracy = (count_correct / dataset.shape[0]) * 100

        return accuracy
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        t = DecisionTree(data=X_train, impurity_func=calc_gini, feature=-1, chi=1, max_depth=max_depth, gain_ratio=False)
        t.build_tree()
        training_acc = t.calc_accuracy(X_train)
        training.append(training_acc)
        validate_acc = t.calc_accuracy(X_validation)
        validation.append(validate_acc)
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        t = DecisionTree(data=X_train, impurity_func=calc_gini, chi= chi_val)
        t.build_tree()
        training_acc = t.calc_accuracy(X_train)
        chi_training_acc.append(training_acc)
        validate_acc = t.calc_accuracy(X_test)
        chi_validation_acc.append(validate_acc)
        
        def find_max_depth(node):
            if node is None:
                return 0
            
            if node.terminal: # If it's a leaf node
                return node.depth
            
            # Find the maximum depth of all children
            if not node.children: 
                return node.depth
            
            return max(find_max_depth(child) for child in node.children)
        
        depth.append(find_max_depth(t.root))

    return chi_training_acc, chi_validation_acc, depth

def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    if node is None:
        return 0
    n_nodes = 1
    for child in node.children:
        n_nodes += count_nodes(child)
    return n_nodes






