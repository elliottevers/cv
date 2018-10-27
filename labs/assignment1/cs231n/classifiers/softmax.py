import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in xrange(num_train):

      scores = X[i].dot(W)
 
      scores = scores - scores.max()
    
      normalization_factor = np.sum(np.exp(scores))
        
      score_unnormalized= np.exp(scores[y[i]])

      loss += -1 * np.log(
          score_unnormalized / normalization_factor
      )

      # differentiate cost with respect to weights
      dW[:, y[i]] += -1 * (normalization_factor - score_unnormalized) / normalization_factor * X[i]
    
      for j in xrange(num_classes):

          if j == y[i]:
              continue
   
          dW[:, j] += np.exp(scores[j]) / normalization_factor * X[i]

  loss /= num_train
  loss += reg * np.sum(W**2)

  dW /= num_train
  dW += 2 * reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]

  num_train = X.shape[0]
    
  scores = X.dot(W)

  scores = scores - scores.max()
    
  scores_unnormalized = np.exp(scores)

  normalization_factor = np.sum(
      scores_unnormalized,
      axis=1
  )
    
  score_unnormalized = scores_unnormalized[range(num_train), y]

  loss_unnormalized = -1 * np.sum(np.log(score_unnormalized / normalization_factor))
    
  reg_term = reg * np.sum(W**2)

  loss = loss_unnormalized/num_train + reg_term

  s = np.divide(
      scores_unnormalized,
      normalization_factor.reshape(
          num_train,
          1
      )
  )

  s[range(num_train), y] = - (normalization_factor - score_unnormalized) / normalization_factor
    
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

