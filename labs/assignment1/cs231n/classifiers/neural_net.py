from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    
    def relu(signal):
        return np.maximum(0, signal)
    
    def softmax(scores):
        # for numeric stability, shift the values of scores so that the max is 0
        scores -= scores.max()
    
        unnormalized_scores = np.exp(scores)

        normalization_factor = np.sum(unnormalized_scores, axis=1).reshape(-1, 1)
    
        return unnormalized_scores/normalization_factor
    
    
    def z(weights, activations, bias):
        return np.dot(activations, weights) + bias
        
    def regularization(reg_strength, W1, W2):
        return reg_strength * (np.sum(W1**2) + np.sum(W2**2))
    
    
    def cross_entropy(correct, predicted):
        
        products = correct * np.log(
            predicted
        )
        
        return -1 * np.sum(
            products,
            axis=1
        )
    
    
    w1 = np.copy(W1)
    
    w2 = np.copy(W2)
    
    z1 = z(w1, X, b1)
    
    a1 = relu(z1)
    
    z2 = z(w2, a1, b2)
    
    scores = z2    
   
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    num_train = X.shape[0]
    
    a2 = softmax(z2)
    
    correct_probs = np.zeros(
        (X.shape[0], self.output_size)
    )
    
    correct_probs[np.arange(y.shape[0]), y] = 1

    loss = np.mean(
        cross_entropy(
            correct_probs, # correct
            a2 # predicted
        )
    ) + regularization(
        reg,
        w1,
        w2
    )
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
        
    ### useful derivatives
    d_z2_d_a1 = lambda w2, a1, b2: w2
        
    # derivative of relu
    d_a1_d_z1 = np.vectorize(
        lambda z1: 1 if z1 >= 0 else 0
    )

    d_z1_d_w1 = lambda w1, x, b1: x
    
    d_z2_d_w2 = lambda w2, a1, b2: a1
    

   
    
    predicted = a2
    
    targets = np.zeros((num_train, self.output_size))
    targets[np.arange(N),y] = 1 # full mass at gt values
    
    # derivative of loss with respect to score
    # https://www.ics.uci.edu/~pjsadows/notes.pdf
    grad_z2 = predicted - targets
    
    # use average of gradient in backprop
    grad_z2 /= num_train

    
    # should be 3 x 10
    grads['W2'] = np.matmul(
        d_z2_d_w2(w2, a1, b2).T, # therefore, 10 x 5
        grad_z2 # should be 3 x 5
    )
    
    grads['b2'] = np.sum(
        grad_z2,
        axis=0,
        keepdims=True
    )
       

    # should be 10 x 5
    grad_a1 = np.matmul(
        d_z2_d_a1(w2, a1, b2), # should be 10 x 3
        grad_z2.T # should be 3 x 5
    )
    

    grad_z1 = d_a1_d_z1(z1) * grad_a1.T
   
    
    # should be 10 x 4
    grads['W1'] = np.matmul(
        d_z1_d_w1(w1, X, b1).T, # 5 x 4
        grad_z1 # should be 10 x 5
    )
    
    grads['b1'] = np.sum(
        grad_z1,
        axis=0,
        keepdims=True
    )

    grads['W1'] += 2*reg*W1
    grads['W2'] += 2*reg*W2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      # randomize indices
    
      batch_ind = np.random.choice(num_train, batch_size)
      X_batch = X[batch_ind]
      y_batch = y[batch_ind]

      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      
#       import ipdb; ipdb.set_trace()
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1'].reshape(self.hidden_size,)
      self.params['b2'] -= learning_rate * grads['b2'].reshape(self.output_size,)
    
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    def relu(signal):
        return np.maximum(0, signal)
    
    def softmax(scores):
        # for numeric stability, shift the values of scores so that the max is 0
        scores -= scores.max()
    
        unnormalized_scores = np.exp(scores)

        normalization_factor = np.sum(unnormalized_scores, axis=1).reshape(-1, 1)
    
        return unnormalized_scores/normalization_factor
    
    
    def z(weights, activations, bias):
        return np.dot(activations, weights) + bias
        
    def regularization(reg_strength, W1, W2):
        return reg_strength * (np.sum(W1**2) + np.sum(W2**2))
    
    
    def cross_entropy(correct, predicted):
        
        products = correct * np.log(
            predicted
        )
        
        return -1 * np.sum(
            products,
            axis=1
        )
    
    
    z1 = z(
        self.params['W1'],
        X,
        self.params['b1']
    )
    
    a1 = relu(z1)
    
    z2 = z(
        self.params['W2'],
        a1,
        self.params['b2']
    )
    
    a2 = softmax(z2)
    
    y_pred = np.argmax(
        a2,
        axis=1
    )
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


