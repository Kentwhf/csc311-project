from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = torch.sigmoid(torch.mm(inputs, torch.t(self.g.weight)) + self.g.bias)
        out = torch.sigmoid(torch.mm(out, torch.t(self.h.weight)) + self.h.bias)
        #####################################################################
        #                       END OF YOUR CwODE                            #
        #####################################################################
        return out


def train(model, lr, train_data, zero_train_data, valid_data, num_epoch, lamb=0):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    epoch_list = []
    train_loss_list = []
    valid_acc_list = []


    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id], requires_grad=True).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.) + model.get_weight_norm() * lamb/2
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        epoch_list.append(epoch)
        train_loss_list.append(train_loss)
        valid_acc_list.append(valid_acc)
    return epoch_list, train_loss_list, valid_acc_list

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    # Set model hyperparameters.
    # Q2c
    num_question = train_matrix.shape[1]
    k_list = [10, 50, 100, 200, 500]
    lr_list = [0.001, 0.01, 0.1]
    num_epoch_list = [5, 20, 40]
    fig, ax = plt.subplots(nrows=len(num_epoch_list), ncols=len(lr_list), figsize=(18,12))
    for i, num_epoch in enumerate(num_epoch_list):
        for j, lr in enumerate(lr_list):
            for k in k_list:
                model = AutoEncoder(num_question, k)
                # Set optimization hyperparameters.
                epochs, train_losses, val_accs = train(model, lr, train_matrix, zero_train_matrix,
                  valid_data, num_epoch)

                ax[i][j].plot(epochs, val_accs, label=f"K = {k}");
                ax[i][j].set_xticks(np.arange(0, num_epoch, num_epoch//5));
                ax[i][j].set_ylabel("Validation Accuracy %")
                ax[i][j].set_title(f"Epoch: {num_epoch}, LR: {lr}", fontsize=10);
                fig.savefig("../figs/Q3c.png");

            ax[i][j].legend(loc='upper left', prop={'size': 6});
    fig.savefig("../figs/Q3c.png");

    # Q2d
    k_star = 100
    lr_star = 0.01
    epoch_star = 50
    model_star = AutoEncoder(num_question, k_star)
    epochs, train_losses, val_accs = train(model_star, lr_star, train_matrix, zero_train_matrix,
                  valid_data, epoch_star)

    fig, ax = plt.subplots(ncols=2, figsize=(12, 28));
    # Plot Train Loss
    ax[0].plot(epochs, train_losses, label="train loss");
    ax[0].set_title("Train Loss by Iteration");
    ax[0].set_xlabel("Epoch");
    ax[0].set_ylabel("Loss");

    # Plot Validation Accuracy
    ax[1].plot(epochs, val_accs, label="validation");
    ax[1].set_title("Validation Accuracy by Iteration");
    ax[1].set_xlabel("Epoch");
    ax[1].set_ylabel("Accuracy");

    fig.savefig("../figs/Q3d.png");

    # Q2e
    lamb_list = [0.001, 0.01, 0.1, 1]
    fig, ax = plt.subplots(ncols=2, figsize=(12, 28));
    for lamb in lamb_list:
        model_star = AutoEncoder(num_question, k_star)
        epochs, train_losses, val_accs = train(model_star, lr_star, train_matrix, zero_train_matrix,
                                           valid_data, epoch_star, lamb)
        ax[0].plot(epochs, train_losses, label=f"lamb={lamb}");
        ax[1].plot(epochs, val_accs, label=f"lamb={lamb}");

    ax[0].set_title("Train Loss by Iteration");
    ax[0].set_xlabel("Epoch");
    ax[0].set_ylabel("Loss");
    ax[0].legend();

    ax[1].set_title("Validation Accuracy by Iteration");
    ax[1].set_xlabel("Epoch");
    ax[1].set_ylabel("Accuracy");
    ax[1].legend();
    fig.savefig("../figs/Q3e.png");


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
