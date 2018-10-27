import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # since gradient of hinge loss is either 1 or 0, we can use a count
        gradient_factor = 0
        margin = 1
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            if correct_class_score < scores[j] + margin:
                loss += scores[j] - correct_class_score + margin
                dW[:, j] += X[i]
                gradient_factor += 1
        # https://stats.stackexchange.com/questions/4608/gradient-of-hinge-loss
        dW[:, y[i]] += (-1) * gradient_factor * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add derivative of regularization to the gradient
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    num_train = X.shape[0]
    
    # scores for all X
    scores = X.dot(W)
    
    responses = scores[list(range(num_train)), y].reshape(num_train, -1)
    
    # complete function
    scores += 1 - responses
    
    # we don't add scores of correct class
    scores[list(range(num_train)), y] = 0
    
    # evaluate standard loss function
    loss = np.sum(np.fmax(scores, 0)) / num_train
    
    # add regularization
    loss += reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    X_mask = np.zeros(scores.shape)
    
    X_mask[scores > 0] = 1
    
    X_mask[np.arange(num_train), y] = -np.sum(X_mask, axis=1)
    
    dW = X.T.dot(X_mask)
    
    dW /= num_train
    
    dW += 2 * reg * W
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
