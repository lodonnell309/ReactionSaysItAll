import numpy as np


class SGD():
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.regularization_rate = reg

    def update(self, model):
        """
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        """

        # Apply L2 penalty to the model. Update the gradient dictionary in the model
        # The term is 1/2 * lambda * sum of squared W
        # During backpropagation, the gradient of L2 penalty term w.r.t W becomes lambda*W.
        for k, v in model.gradients.items():
            if 'W' in k:
                model.gradients[k] = model.gradients[k] + self.regularization_rate * model.weights[k]

        # Update model weights based on the learning rate and gradients
        for k, v in model.weights.items():
            if 'W' in k:
                model.weights[k] -= self.learning_rate * model.gradients[k]