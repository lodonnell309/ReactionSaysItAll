import argparse
import yaml
import copy

from models import TwoLayerNet, SoftmaxRegression
from optimizer import SGD
from utils import load_trainval, load_test, generate_batched_data, train, evaluate#, plot_curves

parser = argparse.ArgumentParser(description='CS7643 Project')
parser.add_argument('--config', # required in the command line
                    default='./config_exp.yaml')


def main():
    train_loss_history, train_acc_history = run()

    # plot_curves(train_loss_history, train_acc_history,
    #             lr = args.learning_rate, r = args.regularization_rate)


def run():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.full_load(f)

    for key in config: # train, model
        for k, v in config[key].items(): # train has batch_size, learning_rate, etc; model has type and hidden_size
            setattr(args, k, v) # https://docs.python.org/3/library/functions.html#setattr

    # Prepare data
    train_data, train_label = load_trainval()
    test_data, test_label = load_test()

    # Create a model
    if args.type == 'SoftmaxRegression':
        model = SoftmaxRegression()

    # Optimizer
    optimizer = SGD(learning_rate=args.learning_rate, reg=args.regularization_rate)

    train_loss_history = []
    train_acc_history = []

    best_accuracy = 0.0
    best_model = None

    for epoch in range(args.epochs):
        print("Epoch: {epoch_num}".format(epoch_num=epoch))

        # train on the train data and get the model
        batched_train_data, batched_train_label = generate_batched_data(train_data, train_label,
                                                                        batch_size=args.batch_size,
                                                                        shuffle=True,
                                                                        seed=1024)
        epoch_loss, epoch_accuracy = train(epoch, batched_train_data, batched_train_label,
                                           model, optimizer, args.debug)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_accuracy)

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_model = copy.deepcopy(model)

    # evaluate on test data with the best model
    
    batched_test_data, batched_test_label = generate_batched_data(test_data, test_label,
                                                                  batch_size=args.batch_size)
    _, test_accuracy = evaluate(batched_test_data, batched_test_label, best_model)

    if args.debug:
        print("Final Accuracy on Train Data: {accuracy:.4f}".format(accuracy=epoch_accuracy))
        print("Final Accuracy on Test Data: {accuracy:.4f}".format(accuracy=test_accuracy))

    return train_loss_history, train_acc_history


if __name__ == '__main__':
    main()
