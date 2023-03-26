"""
The main code for the back propagation assignment. See README.md for details.
"""
import math
from typing import List

import numpy as np
import scipy
from scipy.special import expit, logit


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights correspond to transformations from one layer to the next, so
        the number of layers is equal to one more than the number of weight
        matrices.

        :param layer_weights: A list of weight matrices
        """
        self.layer_weights = layer_weights
        self.predict_variables = {}

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """
        self.predict_variables = {}
        input_times_weights = np.dot(input_matrix,self.layer_weights[0])
        self.predict_variables["input_times_weights"] = input_times_weights
        input_times_weights_sigmoid = expit(input_times_weights)
        self.predict_variables["input_times_weights_sigmoid"] = input_times_weights_sigmoid
        weights_again = np.dot(input_times_weights_sigmoid, self.layer_weights[1])
        self.predict_variables["weights_again"] = weights_again
        mid = expit(weights_again)
        self.predict_variables["mid"] = mid
        if len(self.layer_weights) == 3:
            weights_thrice = np.dot(mid, self.layer_weights[2])
            self.predict_variables["weights_thrice"]= weights_thrice
            final = expit(weights_thrice)
            self.predict_variables["final"] = final
            return final
        final = mid
        self.predict_variables["final"] = final
        return final


    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        self.predict(input_matrix)
        binary_array = np.where(self.predict_variables.get('final')>=.5, 1, 0)
        return binary_array

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure to calculate the
        gradients. In the following description, × is matrix multiplication,
        ⊙ is element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, error_L, between last layer's activations,
        h_L, and the output matrix, y:

        error_L = h_L - y

        Then, for each layer l in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.
        First, calculate g_l as the element-wise product of the error for the
        next layer, error_{l+1}, and the sigmoid gradient of the next layer's
        weighted sum (before the activation function), a_{l+1}.

        g_l = (error_{l+1} ⊙ sigmoid'(a_{l+1}))⊤

        Then calculate the gradient matrix for layer l as the matrix
        multiplication of g_l and the layer's activations, h_l, divided by the
        number of input examples, N:

        grad_l = (g_l × h_l)⊤ / N

        Finally, calculate the error that should be backpropagated from layer l
        as the matrix multiplication of the weight matrix for layer l and g_l:

        error_l = (weights_l × g_l)⊤

        Once this procedure is complete for all layers, the grad_l matrices
        are the gradients that should be returned.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """
        hidden_layers = len(self.layer_weights) -1
        output_pred = self.predict(input_matrix)
        error= np.subtract(output_pred, output_matrix)
        sg1 = 0
        if hidden_layers == 1:
            sg1 = 1.0/(1.0 + np.exp(self.predict_variables["weights_again"]))
            sg1 = sg1*(1-sg1)
        if hidden_layers == 2:
            sg1 = 1.0/(1.0 + np.exp(self.predict_variables.get("weights_thrice")))
            sg1 = sg1*(1-sg1)
        g_la =np.multiply((error), sg1).T
        g_lb = np.dot(g_la, self.predict_variables.get("input_times_weights_sigmoid")).T
        g_l = g_lb/len(input_matrix)
        if hidden_layers == 2:
            sg3 = 1.0/(1.0 + np.exp(self.predict_variables.get("weights_again")))
            sg3 = sg3*(1-sg3)
            g_la = np.multiply(error, sg3).T
            g_2 = np.dot(g_la, self.predict_variables.get("mid")).T/len(input_matrix)
        grads = np.dot(self.layer_weights[1], g_la)
        sg2 = 1.0/(1.0 + np.exp(self.predict_variables.get("input_times_weights")))
        sg2= sg2*(1-sg2)
        grad_1 = np.multiply(grads.T, sg2)
        grad_1 = np.dot(input_matrix.T, grad_1)/len(input_matrix)
        if hidden_layers == 2:
            return grad_1, g_2, g_l
        return grad_1,g_l

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        i = 0
        while i < iterations:
            training = self.gradients(input_matrix, output_matrix)
            for gradients,weights in zip(training, self.layer_weights):
                weights -= learning_rate*gradients
            i+=1
