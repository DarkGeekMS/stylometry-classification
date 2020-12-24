"""
NOTE: This class is not used inside the main code, as it yields lower performance than sklearn MultinomialNB().
NOTE: This implementation is inspired by some open source implementations of Naive Bayes.
"""
import math
from collections import defaultdict

class CustomMultinomialNB:
    """
    Custom Implementation of Naive Bayes classifier
    """
    def __init__(self):
        # initialize priors and posteriors
        self.priors = None
        self.posteriors = None

    def calculate_priors(self, labels_list):
        # calculate class priors
        prior_dict = defaultdict(int)
        # count class labels frequencies
        for class_label in labels_list:
            prior_dict[class_label] += 1
        self.priors = dict(prior_dict)

    def calculate_posteriors(self, samples_list, labels_list):
        # calculate and fit posterior probabilities
        # initialize a list of dictionaries, one for each attribute
        posteriors_dictlist = [dict() for x in range(len(samples_list))]
        # initialize a default dict for each class label, for each attribute dictionary
        for attribute_dict in posteriors_dictlist:
            for class_label in self.priors.keys():
                # Start at 1 for Laplace smoothing
                attribute_dict[class_label] = defaultdict(lambda:1)
        # count the number of instances for each conditional probability [P(Attribute=attr_instance | Class)]
        for col in range(len(samples_list)):
            for row in range(len(samples_list[col])):
                posteriors_dictlist[col][labels_list[row]][samples_list[col][row]] += 1
            # keep track of all attribute possibilites
            attr_set = set()
            for label in posteriors_dictlist[col].keys():
                for attr in posteriors_dictlist[col][label].keys():
                    attr_set.add(attr)
            # add attributes with counts of 1 (Laplace Smoothing) when no occurances for a given class
            for label in posteriors_dictlist[col].keys():
                for attr in attr_set:
                    if attr not in posteriors_dictlist[col][label].keys():
                        posteriors_dictlist[col][label][attr] = 1
        self.posteriors = posteriors_dictlist

    def fit(self, X, Y):
        # fit training samples X and labels Y with supervised Naive Bayes model
        # fit prior and posterior probabilities to the model
        self.calculate_priors(Y)
        self.calculate_posteriors(X, Y)

    def predict(self, X):
        # predict the classes for a set of samples using the trained model
        # check whether the model is fit
        if (self.priors is None or self.posteriors is None):
            raise ValueError("Naive Bayes model is not trained yet!")
        predictions = []
        n_test_instances = len(X[0])
        # make a prediction for every sample in the test set
        for test_row in range(n_test_instances):
            label_predict_probs = []
            # calculate prediction probability for each class label
            for label in self.priors.keys():
                label_count = self.priors[label]
                # prior log probability [log(P(label))]
                label_prob = math.log(label_count / n_test_instances)
                # sum the prediction probability and log(posterior probabilities) to avoid underflow
                # Dividing by the number of labels + number of attribute values (Laplace Smoothing)
                for test_col in range(len(X)):
                    attr = X[test_col][test_row]
                    posterior_prob = self.posteriors[test_col][label][attr] / \
                            (label_count + len(self.posteriors[test_col][label]))
                    label_prob += math.log(posterior_prob)
                # turn log probabilitiy back in probability
                label_prob = math.exp(label_prob)
                label_predict_probs.append((label_prob, label))
            # sort the predictions from high-low and predict the label with the highest probability
            label_predict_probs.sort(reverse=True)
            predictions.append(label_predict_probs[0][1])
        return predictions
