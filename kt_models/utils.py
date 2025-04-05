import time
import os
import numpy as np
import random
from PIL import Image

import matplotlib.pyplot as plt


def load_images(path):
    """
    Load the images from all folders in that path, assign the same
    numeric value to images in the same folder
    :param path: the path of the image folders
    :return:
        data: A list of list where each sub-list with 48x48 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    """
    datas = []
    labels = []

    label = 0

    os.chdir(path)

    # access each folder
    for folder in os.listdir():

        if '.' not in folder:
            images = os.listdir(folder)

            # access each image
            for image in images:
                image_path = folder + '/' + image
                # get the RGB value of image
                im = Image.open(image_path)

                # update the list of data and label
                datas.append(list(im.getdata()))
                labels.append(label)
            label += 1

    return datas, labels


def load_trainval():
    """
    Load training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    print("Loading training data...")
    data, label = load_images('data/train')
    assert len(data) == len(label)
    print("Training data loaded with {count} images".format(count=len(data)))

    total_count = len(data)
    train_ct = int(total_count * 0.8)

    train_data = data[:train_ct]
    train_label = label[:train_ct]

    val_data = data[train_ct:]
    val_label = label[train_ct:]

    return train_data, train_label, val_data, val_label


def load_test():
    """
        Load  testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    # Load training data
    print("Loading testing data...")
    test_data, test_label = load_images('../test')
    assert len(test_data) == len(test_label)
    print("Testing data loaded with {count} images".format(count=len(test_data)))

    return test_data, test_label


def generate_batched_data(data, label, batch_size=32, shuffle=False, seed=None):
    """
    Turn raw data into batched forms
    :param data: A list of list containing the data where each inner list contains 48x48
                 elements corresponding to pixel values in images: [[pix1, ..., pix2304],
                 ..., [pix1, ..., pix2304]]
    :param label: A list containing the labels of data
    :param batch_size: required batch size
    :param shuffle: Whether to shuffle the data: true for training and False for testing
    :return:
        batched_data: (List[np.ndarray]) A list whose elements are batches of images.
        batched_label: (List[np.ndarray]) A list whose elements are batches of labels.
    """
    batched_data = []
    batched_label = []
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    if shuffle:
        indices = np.arange(len(data))
        random.shuffle(indices)
        data = np.array(data)[indices]
        label = np.array(label)[indices]
    else:
        data = np.array(data)
        label = np.array(label)

    for starting in range(0, len(data), batch_size):
        ending = min(starting + batch_size, len(data) - 1)

        to_batch_data = data[starting : ending]
        to_batch_label = label[starting: ending]

        batched_data.append(to_batch_data)
        batched_label.append(to_batch_label)

    return batched_data, batched_label


def train(epoch, batched_train_data, batched_train_label, model, optimizer, debug=True):
    """
    A training function that trains the model for one epoch
    :param epoch: The index of current epoch
    :param batched_train_data: A list containing batches of images
    :param batched_train_label: A list containing batches of labels
    :param model: The model to be trained
    :param optimizer: The optimizer that updates the network weights
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_train_data, batched_train_label)):

        start_time = time.time()
        loss, accuracy = model.forward(input, target)

        optimizer.update(model)
        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]

        forward_time = time.time() - start_time
        if idx % 10 == 0 and debug:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Batch Time {batch_time:.3f} \t'
                   'Batch Loss {loss:.4f}\t'
                   'Train Accuracy ' + "{accuracy:.4f}" '\t').format(
                epoch, idx, len(batched_train_data), batch_time=forward_time,
                loss=loss, accuracy=accuracy))
    epoch_loss /= len(batched_train_data)
    epoch_acc = hits / count_samples

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc


def evaluate(batched_test_data, batched_test_label, model, debug=True):
    """
    Evaluate the model on test data
    :param batched_test_data: A list containing batches of test images
    :param batched_test_label: A list containing batches of labels
    :param model: A pre-trained model
    :return:
        epoch_loss: The average loss of current epoch
        epoch_acc: The overall accuracy of current epoch
    """
    epoch_loss = 0.0
    hits = 0
    count_samples = 0.0
    for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):

        loss, accuracy = model.forward(input, target, mode='valid')

        epoch_loss += loss
        hits += accuracy * input.shape[0]
        count_samples += input.shape[0]
        if debug:
            print(('Evaluate: [{0}/{1}]\t'
                   'Batch Accuracy ' + "{accuracy:.4f}" '\t').format(
                idx, len(batched_test_data), accuracy=accuracy))
    epoch_loss /= len(batched_test_data)
    epoch_acc = hits / count_samples

    return epoch_loss, epoch_acc


def plot_curves


if __name__ == "__main__":
    train_data, train_label, val_data, val_label = load_trainval()
    test_data, test_label = load_test()
    train_batched_data, train_batched_label = generate_batched_data(train_data, train_label)
    val_batched_data, val_batched_label = generate_batched_data(val_data, val_label)
    test_batched_data, test_batched_label = generate_batched_data(test_data, test_label)


