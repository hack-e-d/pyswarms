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

#loading weights
pos=np.load("weights_mark1.npy")

print("Loaded weights :",pos)

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

