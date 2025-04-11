import argparse
import yaml
import copy
import torch
import torch.nn as nn

from models import TwoLayerNet, SoftmaxRegression, CNN
from optimizer import SGD
from utils import load_trainval, load_test, generate_batched_data, train, evaluate  # , plot_curves

parser = argparse.ArgumentParser(description='CS7643 Project')
parser.add_argument('--config',  # required in the command line
                    default='./configs/config_exp.yaml')


def main():
    train_loss_history, train_acc_history = run()

    # plot_curves(train_loss_history, train_acc_history,
    #             lr = args.learning_rate, r = args.regularization_rate)


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
    train_data, train_label, train_folder, train_file_name = load_trainval(model_type=args.type)

    test_data, test_label, test_folder, test_file_name = load_test(model_type=args.type)

    # Create a model
    if args.type == 'SoftmaxRegression':
        model = SoftmaxRegression()
        optimizer = SGD(learning_rate=args.learning_rate, reg=args.regularization_rate)
        criterion = None

    elif args.type == 'TwoLayerNet':
        model = TwoLayerNet(hidden_size=args.hidden_size)
        optimizer = SGD(learning_rate=args.learning_rate, reg=args.regularization_rate)
        criterion = None

    elif args.type == "CNN":
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.regularization_rate
            # , momentum = args.momentum
        )

    train_loss_history = []
    train_acc_history = []

    best_accuracy = 0.0
    best_model = None

    for epoch in range(args.epochs):
        print("Epoch: {epoch_num}".format(epoch_num=epoch))

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
        batched_train_data, batched_train_label, batched_folder, batched_file = generate_batched_data(train_data,
                                                                                                      train_label,
                                                                                                      train_folder,
                                                                                                      train_file_name,
                                                                                                      batch_size=args.batch_size,
                                                                                                      shuffle=True,
                                                                                                      seed=1024)

        epoch_loss, epoch_accuracy = train(args.type, epoch,
                                           batched_train_data, batched_train_label,
                                           batched_folder, batched_file,
                                           model, optimizer, criterion, args.debug)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_accuracy)

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model = copy.deepcopy(model)

    # evaluate on test data with the best model

    batched_test_data, batched_test_label, batched_folder, batched_file = generate_batched_data(test_data, test_label,
                                                                                                test_folder,
                                                                                                test_file_name,
                                                                                                batch_size=args.batch_size)
    _, test_accuracy = evaluate(args.type, batched_test_data, batched_test_label, batched_folder, batched_file,
                                best_model, criterion)

    if args.debug:
        print("Final Accuracy on Train Data: {accuracy:.4f}".format(accuracy=epoch_accuracy))
        print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_accuracy))

    return train_loss_history, train_acc_history


if __name__ == '__main__':
    main()
