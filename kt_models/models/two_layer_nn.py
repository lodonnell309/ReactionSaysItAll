import numpy as np


class TwoLayerNet():
    def __init__(self, input_size=48*48, num_classes=7, hidden_size=128):

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.input_size = input_size

        self.weights = dict()
        self.gradients = dict()

        # initialize weights
        np.random.seed(1024)
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        self.weights['W1'] = np.random.randn(self.input_size, self.hidden_size)
        self.weights['W2'] = np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        """
        The forward pass of the two-layer net. The activation function used in between the two layers
        is sigmoid, which is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        """

        w1 = self.weights['W1']
        w2 = self.weights['W2']
        b1 = self.weights['b1']
        b2 = self.weights['b2']

        ## forward process
        # fully-connected layer
        Z = X @ w1 + b1

        # sigmoid activation
        U0 = 1 / (1 + np.exp(-Z))

        # biases for w2
        T = U0 @ w2 + b2

        # softmax (generalization of the logistic function to multiple dimension)
        N = T.shape[0]
        p = np.exp(T) / np.reshape(np.sum(np.exp(T), axis = 1), (N, 1))

        ## compute the Cross-Entropy loss
        data_size = len(y)
        class_size = p.shape[1]

        y_true_probability = np.zeros((data_size, class_size))
        y_true_probability[np.arange(data_size), y] = 1

        loss = -np.sum(y_true_probability * np.log(p)) / data_size

        ## compute the accuracy of current batch
        y_pred = np.argmax(p, axis = 1)
        accuracy = np.sum(y_pred == y) / data_size

        ## backward process

        # loss w.r.t. T
        # convert y labels to one-hot-encoded array https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
        one_hot_y = np.zeros((y.shape[0], self.num_classes))
        one_hot_y[np.arange(y.shape[0]), y] = 1
        loss_wrt_T = (p - one_hot_y) / y.shape[0]

        self.gradients['b2'] = np.sum(loss_wrt_T, axis=0)

        # loss w.r.t. W2
        loss_wrt_W2 = U0.T @ loss_wrt_T
        self.gradients['W2'] = loss_wrt_W2

        # # loss w.r.t. U
        loss_wrt_U0 = loss_wrt_T @ w2.T

        # loss w.r.t. Z (sigmoid deriviative)
        U0_wrt_Z = U0 * (1 - U0)
        loss_wrt_Z = loss_wrt_U0 * U0_wrt_Z

        self.gradients['b1'] = np.sum(loss_wrt_Z, axis=0)

        # loss w.r.t. W1
        loss_wrt_W1 = X.T @ loss_wrt_Z
        self.gradients['W1'] = loss_wrt_W1

        return loss, accuracy
