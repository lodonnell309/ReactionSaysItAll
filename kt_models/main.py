import argparse
import yaml
import copy
import numpy
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models import TwoLayerNet, SoftmaxRegression, CNN
from optimizer import SGD
from utils import load_trainval, load_test, generate_batched_data, train, evaluate, plot_curves, lets_predict

parser = argparse.ArgumentParser(description='CS7643 Project')
parser.add_argument('--config',  # required in the command line
                    default='./configs/config_exp.yaml')
parser.add_argument('--final_file_name',  # required in the command line
                    default=None)


def main():
    train_loss_history, train_acc_history, valid_loss_history, valid_acc_history, best_model, resize_image_transform, folders = run()

    plot_curves(train_loss_history, train_acc_history, valid_loss_history, valid_acc_history,
                lr = args.learning_rate, r = args.regularization_rate, model_type = args.type)

    if max(valid_acc_history) > 0.3 and args.final_file_name != None:
        the_image = args.final_file_name

        pred = lets_predict(best_model, resize_image_transform,
                            real_life_file = the_image,
                            model_type = args.type)

        print('Predicted emotion for '+the_image+' : '+folders[pred])


def run():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.full_load(f)

    for key in config:  # train, model
        for k, v in config[key].items():  # train has batch_size, learning_rate, etc; model has type and hidden_size
            setattr(args, k, v)  # https://docs.python.org/3/library/functions.html#setattr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    train_data, train_label, val_data, val_label, folders = load_trainval(model_type=args.type, shuffle=True, seed=1024)

    test_data, test_label = load_test(model_type=args.type)
    resize_image_transform = None
    criterion = None

    # Create a model
    if args.type == 'SoftmaxRegression':
        model = SoftmaxRegression()
        optimizer = SGD(learning_rate=args.learning_rate, reg=args.regularization_rate)

    elif args.type == 'TwoLayerNet':
        model = TwoLayerNet(hidden_size=args.hidden_size)
        optimizer = SGD(learning_rate=args.learning_rate, reg=args.regularization_rate)

    elif args.type == "CNN":
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.regularization_rate,
            momentum = args.momentum
        )

        # Adjust the real-life images to fit my model's setup
        # Resize or Crop Input Images
        resize_image_transform = transforms.Compose(
            [
                transforms.Resize((48, 48)),  # force image to match training size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # match training normalization
            ]
        )

    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []

    best_accuracy = 0.0
    best_model = None

    for epoch in range(args.epochs):
        print("Epoch: {epoch_num}".format(epoch_num=epoch))

        if args.type == "CNN":
            # adjust learning rate
            temp_epoch = epoch + 1
            if temp_epoch <= args.warmup:
                lr = args.learning_rate * temp_epoch / args.warmup
            elif temp_epoch > args.steps[1]:
                lr = args.learning_rate * 0.01
            elif temp_epoch > args.steps[0]:
                lr = args.learning_rate * 0.1
            else:
                lr = args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # train on the train data and get the model
        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=args.batch_size)
        epoch_loss, epoch_accuracy, runtime = train(args.type, epoch,
                                                    batched_train_data, batched_train_label,
                                                    model, optimizer, criterion, args.debug)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_accuracy)

        # evaluate on validation data
        batched_val_data, batched_val_label = generate_batched_data(val_data, val_label,
                                                                    batch_size=args.batch_size)
        valid_loss, valid_accuracy = evaluate(args.type, batched_val_data, batched_val_label,
                                              model, criterion, args.debug)

        if args.debug:
            print("* Validation Accuracy: {accuracy:.4f}".format(accuracy=valid_accuracy))

        valid_loss_history.append(valid_loss)
        valid_acc_history.append(valid_accuracy)

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_model = copy.deepcopy(model)

    # evaluate on test data with the best model

    batched_test_data, batched_test_label = generate_batched_data(test_data, test_label,
                                                                   batch_size=args.batch_size)
    _, test_accuracy = evaluate(args.type, batched_test_data, batched_test_label,
                                best_model, criterion)

    if args.debug:
        print("Average Runtime per Epoch: {x}".format(x =sum(runtime) / len(runtime)))
        print("Final Accuracy on Train Data: {accuracy:.4f}".format(accuracy=epoch_accuracy))
        print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_accuracy))

    return (train_loss_history, train_acc_history, valid_loss_history, valid_acc_history,
            best_model, resize_image_transform, folders)


if __name__ == '__main__':
    main()
