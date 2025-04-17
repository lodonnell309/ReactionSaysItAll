import time
import os
import numpy as np
import random
import torch
from PIL import Image

import matplotlib.pyplot as plt


def load_images(image_file, model_type, transform=False):
    """
    Load the images from all folders in that path, assign the same
    numeric value to images in the same folder
    :param path: the path of the image folders
    :return:
        data: A list of list where each sub-list with 48x48 elements
              corresponding to the pixels in each image
        labels: A list containing labels of images
    """
    # get the RGB value of image
    im = Image.open(image_file)

    if transform != False:
        gray_image = im.convert("L")
        # Apply the transform
        output = transform(gray_image)
    else:
      if model_type == 'SoftmaxRegression' or model_type == 'TwoLayerNet':
        im = list(im.getdata())
      # update the list of data and label
      output = np.array(im) / 255

    return output


def load_trainval(model_type, shuffle=False, seed=None):
    """
    Load training data with labels
    :return:
        train_data: A list of list containing the training data
        train_label: A list containing the labels of training data
        val_data: A list of list containing the validation data
        val_label: A list containing the labels of validation data
    """
    print("Loading training data...")

    data = []
    label = []
    folders = []

    l = 0
    path = 'data/train'

    # access each folder
    for folder in os.listdir(path):
        if '.' not in folder:
            folders.append(folder)
            images = os.listdir(path + '/' + folder)

            # access each image
            for image in images:
                image_file = path + '/' + folder + '/' + image
                normalized_image_data = load_images(image_file, model_type)
                data.append(normalized_image_data)
                label.append(l)

            l += 1

    assert len(data) == len(label)

    total_count = len(data)
    train_ct = int(total_count * 0.7)

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

    train_data = data[:train_ct]
    train_label = label[:train_ct]
    val_data = data[train_ct:]
    val_label = label[train_ct:]

    print("Training data loaded with {count} images".format(count=len(train_data)))
    print("Validation data loaded with {count} images".format(count=len(val_data)))

    return train_data, train_label, val_data, val_label, folders


def load_test(model_type):
    """
        Load  testing data with labels
        :return:
            data: A list of list containing the testing data
            label: A list containing the labels of testing data
        """
    print("Loading testing data...")
    data = []
    label = []

    l = 0
    path = 'data/test'

    # access each folder
    for folder in os.listdir(path):
        if '.' not in folder:
            images = os.listdir(path + '/' + folder)

            # access each image
            for image in images:
                image_file = path + '/' + folder + '/' + image
                normalized_image_data = load_images(image_file, model_type)
                data.append(normalized_image_data)
                label.append(l)

            l += 1

    assert len(data) == len(label)
    print("Testing data loaded with {count} images".format(count=len(data)))

    data = np.array(data)
    label = np.array(label)

    return data, label


def generate_batched_data(data, label, batch_size=32):
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

    for starting in range(0, len(data), batch_size):
        ending = min(starting + batch_size, len(data))
        to_batch_data = data[starting: ending]
        to_batch_label = label[starting: ending]
        batched_data.append(to_batch_data)
        batched_label.append(to_batch_label)

    return batched_data, batched_label


def train(model_type, epoch, batched_train_data, batched_train_label, model, optimizer,
          criterion, debug=True):
    """
    A training function that trains the model for one epoch
    """

    epoch_loss = 0.0
    total_correct = 0
    total_size = 0.0
    runtime = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, (input, target) in enumerate(
            zip(batched_train_data, batched_train_label)):

        start_time = time.time()

        if model_type == 'SoftmaxRegression' or model_type == 'TwoLayerNet':
            loss, accuracy = model.forward(input, target)
            optimizer.update(model)

        elif model_type == 'CNN':
            model.train()

            input = torch.tensor(input, dtype=torch.float32).to(device)
            input = input.unsqueeze(1)
            target = torch.tensor(target).long().to(device)

            optimizer.zero_grad()
            output = model.forward(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            batch_size = target.shape[0]
            _, pred = torch.max(output, dim=-1)
            correct = pred.eq(target).sum() * 1.0
            accuracy = correct / batch_size

        epoch_loss += loss
        total_correct += accuracy * input.shape[0]
        total_size += input.shape[0]

        runtime.append(time.time() - start_time)

    epoch_loss /= len(batched_train_data)
    epoch_acc = total_correct / total_size

    if debug:
        print("* Average Accuracy of Epoch {} is: {:.4f}".format(epoch, epoch_acc))
    return epoch_loss, epoch_acc, runtime


def evaluate(model_type, batched_test_data, batched_test_label, model, criterion,
             debug=True):
    """
    Evaluate the model on validaiton and test data
    """

    epoch_loss = 0.0
    total_correct = 0
    total_size = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'SoftmaxRegression' or model_type == 'TwoLayerNet':
        for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):
            loss, accuracy = model.forward(input, target, mode='valid')

            epoch_loss += loss
            total_correct += accuracy * input.shape[0]
            total_size += input.shape[0]

    elif model_type == 'CNN':
        model.eval()

        with torch.no_grad():
            for idx, (input, target) in enumerate(zip(batched_test_data, batched_test_label)):
                input = torch.tensor(input, dtype=torch.float32).to(device)
                input = input.unsqueeze(1)
                target = torch.tensor(target).long().to(device)

                output = model.forward(input)
                loss = criterion(output, target)

                _, pred = torch.max(output, 1)

                epoch_loss += loss
                total_correct += pred.eq(target).sum() * 1.0
                total_size += input.shape[0]

    epoch_loss /= len(batched_test_data)
    epoch_acc = total_correct / total_size

    return epoch_loss, epoch_acc


def plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, lr, r, model_type):
    """
    Plot learning curves with matplotlib. Make sure training loss and validation loss are plot in the same figure and
    training accuracy and validation accuracy are plot in the same figure too.
    :param train_loss_history: training loss history of epochs
    :param train_acc_history: training accuracy history of epochs
    :param valid_loss_history: validation loss history of epochs
    :param valid_acc_history: validation accuracy history of epochs
    :return: None, save two figures in the current directory
    """
    plt.plot([i for i in range(len(train_loss_history))],
             [train_loss_history[i].item() for i in range(len(train_loss_history))],
             label='Train')
    plt.plot([i for i in range(len(valid_loss_history))],
             [valid_loss_history[i].item() for i in range(len(valid_loss_history))],
             label='Valid')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('outputs/learning_loss_curve_'+model_type+'.png')

    plt.figure(0)
    plt.plot([i for i in range(len(train_acc_history))],
             [train_acc_history[i].item() for i in range(len(train_acc_history))],
             label='Train')
    plt.plot([i for i in range(len(valid_acc_history))],
             [valid_acc_history[i].item() for i in range(len(valid_acc_history))],
             label='Valid')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.savefig('outputs/learning_accuracy_curve_'+model_type+'.png')

def lets_predict(model, resize_image_transform, real_life_file, model_type):
    model.eval()

    image_tensor = load_images(real_life_file, model_type, transform=resize_image_transform)

    if model_type == 'SoftmaxRegression':
        return None
    elif model_type == 'TwoLayerNet':
        return None
    elif model_type == "CNN":
        image_tensor = image_tensor.unsqueeze(1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, dim=1)
    return pred

if __name__ == "__main__":
    train_data, train_label = load_trainval()
    # test_data, test_label = load_test()
    # train_batched_data, train_batched_label = generate_batched_data(train_data, train_label)
    # val_batched_data, val_batched_label = generate_batched_data(val_data, val_label)
    # test_batched_data, test_batched_label = generate_batched_data(test_data, test_label)
    # print(train_data[0])


