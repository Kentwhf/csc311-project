from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    print(theta.shape)
    print(beta.shape)
    # temp = -np.isnan(data).astype(int) + 1
    for i in range(len(theta)):
        for j in range(len(beta)):
            if data[i, j] == 1 or 0:
                log_lklihood += data[i, j] * (theta[i][0] - beta[j][0]) - np.log(1 + np.exp(theta[i][0] - beta[j][0]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return log_lklihood

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    temp = (data == 1).astype(int)
    for i in range(len(theta)):
        theta[i] -= lr * (-np.sum(sigmoid(theta[i] - beta)) + np.sum(temp[i, :]))
    for j in range(len(beta)):
        beta[j] -= lr * (np.sum(sigmoid(theta - beta[j])) - np.sum(temp[:, j]))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(train_sparse_matrix, train_data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(shape=(train_sparse_matrix.shape[0], 1))
    beta = np.zeros(shape=(train_sparse_matrix.shape[1], 1))
    # np.random.seed(1)
    # theta = np.random.rand(train_sparse_matrix.shape[0], 1)
    # beta = np.random.rand(train_sparse_matrix.shape[1], 1)

    val_acc_lst, train_acc_lst = [], []
    val_log_likelihood, train_log_likelihood = [], []
    val_sparse_matrix = create_sparse_matrix(val_data)

    for i in range(iterations):
        # Log likelihood
        train_neg_lld = neg_log_likelihood(train_sparse_matrix, theta=theta, beta=beta)
        train_log_likelihood.append(-train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_sparse_matrix, theta=theta, beta=beta)
        val_log_likelihood.append(-val_neg_lld)

        train_score = evaluate(data=train_data, theta=theta, beta=beta)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(val_score)

        print("NLLK: {} \t Train Score: {} \t Val Score: {}".format(train_neg_lld, train_score, val_score))
        theta, beta = update_theta_beta(train_sparse_matrix, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def create_sparse_matrix(data):
    matrix = np.empty(shape=(542, 1774))
    matrix[:] = np.NaN
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        mark = data["is_correct"][i]
        matrix[u, q] = mark
    return matrix


def main():
    train_data = load_train_csv("../data")

    # You may optionally use the sparse matrix.
    train_sparse_matrix = load_train_sparse("../data").toarray()

    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr, iterations = 0.01, 10
    theta, beta, val_log_likelihood, train_log_likelihood = \
        irt(train_sparse_matrix, train_data, val_data, lr, iterations)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
