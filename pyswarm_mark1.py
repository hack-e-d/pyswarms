# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Import PySwarms
import pyswarms as ps
# loading breast cancer dataset
data = load_breast_cancer()

# Store the features as X and the labels as y
X = data.data
y = data.target
print("Shape of X",X.shape)
# Forward propagation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss. It receives a set of parameters that must be
    rolled-back into the corresponding weights and biases.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """
    # Neural network architecture(x has 30 attributes and y has 2 classes)
    n_inputs = 30
    n_hidden = 20
    n_classes = 2

    # Roll-back the weights and biases
    W1 = params[0:600].reshape((n_inputs,n_hidden))    #30x20=600
    b1 = params[600:620].reshape((n_hidden,))          #20
    W2 = params[620:660].reshape((n_hidden,n_classes)) #20x2=40
    b2 = params[660:662].reshape((n_classes,))         #2

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    # Compute for the softmax of the logits
    exp_scores = np.exp(logits)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = 569 # Number of samples
    corect_logprobs = -np.log(probs[range(N), y])
    loss = np.sum(corect_logprobs) / N

    return loss

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
dimensions = (30 * 20) + (20 * 2) + 20 + 2
optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=10)

print("Cost (Loss value) :",cost)

#saving the weights to file
np.save('weights_mark1.npy', pos)

def predict(X, pos):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        Input Iris dataset
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    n_inputs = 30
    n_hidden = 20
    n_classes = 2

    # Roll-back the weights and biases
    W1 = pos[0:600].reshape((n_inputs,n_hidden))
    b1 = pos[600:620].reshape((n_hidden,))
    W2 = pos[620:660].reshape((n_hidden,n_classes))
    b2 = pos[660:662].reshape((n_classes,))

    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    logits = z2          # Logits for Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred

print("Prediction accuracy :",str((predict(X,pos)==y).mean()))

