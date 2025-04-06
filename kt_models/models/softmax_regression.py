import numpy as np

class SoftmaxRegression():
    def __init__(self, input_size=48*48, num_classes=7):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        self.input_size = input_size
        self.num_classes = num_classes
        np.random.seed(1024)
        # initialize weights of the single layer regression network. No bias term included.
        self.weights = dict()
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients = dict()
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        ## Implement the forward process

        # fully-connected layer
        Z = X @ self.weights['W1']

        # ReLU activation
        A = np.maximum(Z, 0)

        # softmax regression (generalization of the logistic function to multiple dimension)
        N = A.shape[0]
        p = np.exp(A) / np.reshape(np.sum(np.exp(A), axis = 1), (N, 1))

        ## Compute Cross-Entropy Loss based on prediction of the network and labels
        # convert y labels to one-hot-encoded array https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
        data_size = len(y)
        class_size = p.shape[1]

        y_true_probability = np.zeros((data_size, class_size))
        y_true_probability[np.arange(data_size), y] = 1

        loss = -np.sum(y_true_probability * np.log(p)) / data_size

        ## compute the accuracy of current batch
        y_pred = np.argmax(p, axis = 1)
        accuracy = np.sum(y_pred == y) / data_size

        if mode != 'train':
            return loss, accuracy

        ## Implement the backward process

        ## Compute the gradient of the loss with respect to the weights

        # loss w.r.t. A
        # convert y labels to one-hot-encoded array https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-one-hot-encoded-array-in-numpy
        one_hot_y = np.zeros((y.shape[0], self.num_classes))
        one_hot_y[np.arange(y.shape[0]), y] = 1

        loss_wrt_A = - (one_hot_y - p) / y.shape[0]

        # loss w.r.t. Z
        A_wrt_Z = np.maximum(Z, 0)
        A_wrt_Z[A_wrt_Z > 0] = 1
        loss_wrt_Z = loss_wrt_A * A_wrt_Z

        # loss w.r.t. W
        loss_wrt_W = X.T @ loss_wrt_Z

        self.gradients['W1'] = loss_wrt_W

        return loss, accuracy
