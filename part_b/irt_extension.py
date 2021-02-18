from utils import *
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt




def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def create_sparse_matrix(data):
    matrix = np.empty(shape=(542, 1774))
    matrix[:] = np.NaN
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        mark = data["is_correct"][i]
        matrix[u, q] = mark
    return matrix

def neg_log_likelihood(data, theta, beta, alpha, lower, upper):
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
    for idx, u_id in enumerate(data["user_id"]):
        q_id = data["question_id"][idx]

        k = alpha[q_id] * (theta[u_id] - beta[q_id])

        if data["is_correct"][idx] == 0:
            log_lklihood += np.log(lower[u_id] + (upper[u_id] - lower[u_id]) / (1 + np.exp(k)))
        elif data["is_correct"][idx] == 1:
            log_lklihood += np.log(lower[u_id] + (upper[u_id] - lower[u_id]) * sigmoid(k))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_params(data, lr, theta, beta, alpha, lower, upper):
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

    u_id_arr = np.array(data["user_id"])
    q_id_arr = np.array(data["question_id"])
    c_id_arr = np.array(data["is_correct"])

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    alpha_copy = alpha.copy()
    lower_copy = lower.copy()
    upper_copy = upper.copy()

    train_matrix = create_sparse_matrix(data)

    for i in range(len(theta)):
        lower_grad = []
        upper_grad = []
        theta_grad = 0.
        for j, q_id in enumerate(train_matrix[i, :]):
            if np.isnan(q_id):
                continue
            else:
                k = alpha_copy[j] * (theta_copy[i] - beta_copy[j])
                coef = (upper_copy[i] - lower_copy[i]) * alpha_copy[j]
                if q_id == 1:
                    theta_grad += coef * sigmoid(k) / (upper_copy[i] * np.exp(k) + lower_copy[i])
                    upper_grad.append(-np.exp(k)/(lower_copy[i] + upper_copy[i] * np.exp(k)))
                    lower_grad.append(1/(lower_copy[i] + upper_copy[i] * np.exp(k)))

                elif q_id == 0:
                    theta_grad += -coef * sigmoid(k) / (lower_copy[i] * np.exp(k) + upper_copy[i])
                    upper_grad.append(-1/(upper_copy[i] + lower_copy[i] * np.exp(k)))
                    lower_grad.append(np.exp(k)/(upper_copy[i] + lower_copy[i] * np.exp(k)))

        theta[i] += lr * theta_grad
        upper[i] += lr * np.mean(upper_grad)
        lower[i] += lr * np.mean(lower_grad)

    # upper[upper >= 1] = 1
    # lower[lower <= 0] = 0

    for j in range(len(beta)):
        beta_grad = 0.
        alpha_grad = 0.

        for i, u_id in enumerate(train_matrix[:, j]):
            if np.isnan(u_id):
                continue
            else:
                k = alpha_copy[j] * (theta_copy[i] - beta_copy[j])
                coef = (upper_copy[i] - lower_copy[i]) * alpha_copy[j]
                if u_id == 1:
                    beta_grad += -coef * sigmoid(k) / (upper_copy[i] * np.exp(k) + lower_copy[i])
                    alpha_grad += -coef * sigmoid(k) * (theta_copy[i] - beta_copy[j]) / (
                            upper_copy[i] * np.exp(k) + lower_copy[i])
                elif u_id == 0:
                    beta_grad += coef * sigmoid(k) / (lower_copy[i] * np.exp(k) + upper_copy[i])
                    alpha_grad += coef * sigmoid(k) * (theta_copy[i] - beta_copy[j]) / (
                            lower_copy[i] * np.exp(k) + upper_copy[i])

        beta[j] += lr * beta_grad
        alpha[j] += lr * alpha_grad

    return theta, beta, alpha, lower, upper


def irt(train_data, val_data, lr, iterations):
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

    theta = np.zeros(542)
    beta = np.zeros(1774)
    alpha = np.ones(1774)

    lower = np.ones(542) * 0.00001
    upper = np.ones(542)

    val_acc_lst, train_acc_lst = [], []
    val_log_likelihood, train_log_likelihood = [], []

    for i in range(iterations):
        # Log likelihood
        train_neg_lld = neg_log_likelihood(train_data, theta=theta, beta=beta, alpha=alpha, lower=lower, upper=upper)
        train_log_likelihood.append(train_neg_lld)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha, lower=lower, upper=upper)
        val_log_likelihood.append(val_neg_lld)

        train_score = evaluate(data=train_data, theta=theta, beta=beta, alpha=alpha, lower=lower, upper=upper)
        train_acc_lst.append(train_score)
        val_score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, lower=lower, upper=upper)
        val_acc_lst.append(val_score)
        #
        print("NLLK: {} \t Train Score: {} \t Validation Score: {}".format(train_neg_lld, train_score, val_score))
        theta, beta, alpha, lower, upper = update_params(train_data, lr, theta, beta, alpha, lower, upper)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, lower, upper, val_log_likelihood, train_log_likelihood


def evaluate(data, theta, beta, alpha, lower, upper):
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
        x = (theta[u] - beta[q]) * alpha[q]
        p_a = sigmoid(x) * (upper[u] - lower[u]) + lower[u]
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])




def evaluate_2(data, theta, beta):
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

    # Part B
    lr, iterations = 0.01, 10
    theta, beta, alpha, lower, upper, val_log_likelihood, train_log_likelihood = \
        irt(train_data, val_data, lr, iterations)

    # Part C
    val_score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, lower=lower, upper=upper)
    test_score = evaluate(data=test_data, theta=theta, beta=beta, alpha=alpha, lower=lower, upper=upper)
    print(val_score)
    print(test_score)
    return theta, beta, alpha, lower, upper, val_log_likelihood, train_log_likelihood, val_score, test_score

if __name__ == "__main__":
    theta, beta, alpha, lower, upper, val_nllk, train_nllk, val_score, test_score = main()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    # Analysis

    ### Load Part A Model
    theta_old = np.load("../part_a/theta.npy")
    beta_old = np.load("../part_a/beta.npy")
    val_nllk_old = np.load("../part_a/val_nllk.npy")
    train_nllk_old = np.load("../part_a/train_nllk.npy")

    fig, ax = plt.subplots(ncols=2, figsize=(18, 6))
    ax[0].plot(train_nllk_old, label="base nllk");
    ax[0].plot(train_nllk, label="new nllk");
    ax[0].legend();
    ax[0].set_title("Training Negative Log-Likelihood");
    ax[0].set_xticks(np.arange(0, 20, 5));

    ax[1].plot(val_nllk_old, label="base nllk");
    ax[1].plot(val_nllk, label="new nllk");
    ax[1].legend();
    ax[1].set_title("Validation Negative Log-Likelihood");
    ax[1].set_xticks(np.arange(0, 20, 5))
    fig.savefig("../figs/nllk_comparison.png")
    plt.show()

    fig, ax = plt.subplots(ncols=2, figsize=(18, 6))

    ax[0].hist(theta_old, label="base", bins=100);
    ax[0].hist(theta, label="new", bins=100);
    ax[0].legend();
    ax[0].set_title("Theta Distribution");
    # ax[0].set_xticks(np.arange(0, 20, 5));

    ax[1].hist(beta_old, label="base", bins=100);
    ax[1].hist(beta, label="new", bins=100);
    ax[1].legend();
    ax[1].set_title("Beta Distribution");
    # ax[1].set_xticks(np.arange(0, 20, 5))
    # fig.savefig("../figs/nllk_comparison.png")
    plt.show()

    fig, ax = plt.subplots(ncols=2, figsize=(18, 6))

    ax[0].plot(theta_old[np.argsort(theta_old)], label="base");
    ax[0].plot(theta[np.argsort(theta_old)], label="new");
    ax[0].legend();
    ax[0].set_title("Theta Sorted Comparison");
    # ax[0].set_xticks(np.arange(0, 20, 5));

    ax[1].plot(beta_old[np.argsort(beta_old)], label="base");
    ax[1].plot(beta[np.argsort(beta_old)], label="new");
    ax[1].legend();
    ax[1].set_title("Beta Sorted Comparison");
    # ax[1].set_xticks(np.arange(0, 20, 5))
    # fig.savefig("../figs/nllk_comparison.png")
    plt.show()

    fig, ax = plt.subplots(ncols=3, figsize=(24, 6))

    ax[0].plot(theta[np.argsort(theta)], np.mean(
        lower[np.argsort(theta)] + (upper[np.argsort(theta)] - lower[np.argsort(theta)]) * sigmoid(
            alpha * (np.repeat(theta[np.argsort(theta)].reshape(-1, 1), 1774, axis=1) - beta)).T, axis=0), label="new");
    ax[0].plot(theta_old[np.argsort(theta_old)],
               np.mean(sigmoid(np.repeat(theta_old[np.argsort(theta_old)].reshape(-1, 1), 1774, axis=1) - beta_old),
                       axis=1), label="base");
    ax[0].legend();
    ax[0].set_title("Avg Correctness VS student ability");
    ax[0].set_ylabel("Probability");
    ax[0].set_xlabel("Theta");

    ax[1].scatter(beta[np.argsort(beta)], np.mean(
        lower + (upper - lower) * sigmoid(
            alpha[np.argsort(beta)] * (np.repeat(beta[np.argsort(beta)].reshape(-1, 1), 542, axis=1) - theta).T).T,
        axis=1
    ), label="new", s=5);
    ax[1].scatter(beta_old[np.argsort(beta_old)],
                  np.mean(sigmoid(np.repeat(beta_old[np.argsort(beta_old)].reshape(-1, 1), 542, axis=1) - theta_old),
                          axis=1), label="base", s=5);
    ax[1].legend();
    ax[1].set_title("Avg Correctness VS Question Difficulty");
    ax[1].set_ylabel("Probability");
    ax[1].set_xlabel("Beta");

    ax[2].scatter(alpha[np.argsort(alpha)], np.mean(
        lower + (upper - lower) * sigmoid(
            alpha[np.argsort(alpha)] * (np.repeat(beta[np.argsort(alpha)].reshape(-1, 1), 542, axis=1) - theta).T).T,
        axis=1
    ), label="new", s=5);
    ax[2].legend();
    ax[2].set_title("Avg Correctness VS Question Steepness");
    ax[2].set_ylabel("Probability");
    ax[2].set_xlabel("Alpha");

    plt.tight_layout()
    fig.savefig("../figs/avg_correctness.png");

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(alpha[np.argsort(beta)], beta[np.argsort(beta)],
               np.mean(
                   lower + (upper - lower) * sigmoid(alpha[np.argsort(beta)] * (
                               np.repeat(beta[np.argsort(beta)].reshape(-1, 1), 542, axis=1) - theta).T).T,
                   axis=1
               ), s=1)
    ax.set_xlabel("Alpha");
    ax.set_ylabel("Beta");
    ax.set_zlabel("Probability");

    fig, ax = plt.subplots(ncols=2, figsize=(18, 6))

    upper.sort()
    ax[0].hist(upper, bins=30);
    ax[0].legend();
    ax[0].set_title("Upper bound on correctness");
    ax[0].set_ylabel("Number of students");
    ax[0].set_xlabel("Upper");

    lower.sort()
    ax[1].hist(lower, bins=30);
    ax[1].legend();
    ax[1].set_title("Lower bound on correctness");
    ax[1].set_ylabel("Number of students");
    ax[1].set_xlabel("Lower");

    hard_question_ids = np.argsort(beta)[:89]
    easy_question_ids = np.argsort(beta)[-89:]
    hard_question_data = {"user_id": [], "question_id": [], "is_correct": []}
    easy_question_data = {"user_id": [], "question_id": [], "is_correct": []}
    for idx, question_id in enumerate(val_data["question_id"]):
        if question_id in list(hard_question_ids):
            hard_question_data["user_id"].append(val_data["user_id"][idx])
            hard_question_data["question_id"].append(val_data["question_id"][idx])
            hard_question_data["is_correct"].append(val_data["is_correct"][idx])
        if question_id in list(easy_question_ids):
            easy_question_data["user_id"].append(val_data["user_id"][idx])
            easy_question_data["question_id"].append(val_data["question_id"][idx])
            easy_question_data["is_correct"].append(val_data["is_correct"][idx])
    print("New Model Hard Question Validation Accuracy: ",evaluate(hard_question_data, theta, beta, alpha, lower, upper))
    print("Old Model Hard Question Validation Accuracy: ",evaluate_2(hard_question_data, theta_old, beta_old))
    print("New Model Easy Question Validation Accuracy: ",evaluate(easy_question_data, theta, beta, alpha, lower, upper))
    print("Old Model Easy Question Validation Accuracy: ",evaluate_2(easy_question_data, theta_old, beta_old))

    hard_question_ids = np.argsort(beta)[:89]
    easy_question_ids = np.argsort(beta)[-89:]
    hard_question_data = {"user_id": [], "question_id": [], "is_correct": []}
    easy_question_data = {"user_id": [], "question_id": [], "is_correct": []}
    for idx, question_id in enumerate(test_data["question_id"]):
        if question_id in list(hard_question_ids):
            hard_question_data["user_id"].append(test_data["user_id"][idx])
            hard_question_data["question_id"].append(test_data["question_id"][idx])
            hard_question_data["is_correct"].append(test_data["is_correct"][idx])
        if question_id in list(easy_question_ids):
            easy_question_data["user_id"].append(test_data["user_id"][idx])
            easy_question_data["question_id"].append(test_data["question_id"][idx])
            easy_question_data["is_correct"].append(test_data["is_correct"][idx])
    print("New Model Hard Question Test Accuracy: ", evaluate(hard_question_data, theta, beta, alpha, lower, upper))
    print("Old Model Hard Question Test Accuracy: ", evaluate_2(hard_question_data, theta_old, beta_old))
    print("New Model Easy Question Test Accuracy: ", evaluate(easy_question_data, theta, beta, alpha, lower, upper))
    print("Old Model Easy Question Test Accuracy: ", evaluate_2(easy_question_data, theta_old, beta_old))

    bad_student_ids = np.argsort(theta)[:27]
    good_student_ids = np.argsort(theta)[-27:]
    bad_student_data = {"user_id": [], "question_id": [], "is_correct": []}
    good_student_data = {"user_id": [], "question_id": [], "is_correct": []}
    for idx, user_id in enumerate(val_data["user_id"]):
        if user_id in list(bad_student_ids):
            bad_student_data["user_id"].append(user_id)
            bad_student_data["question_id"].append(val_data["question_id"][idx])
            bad_student_data["is_correct"].append(val_data["is_correct"][idx])
        if user_id in list(good_student_ids):
            good_student_data["user_id"].append(user_id)
            good_student_data["question_id"].append(val_data["question_id"][idx])
            good_student_data["is_correct"].append(val_data["is_correct"][idx])
    print("New Model Good Student Validation Accuracy: ", evaluate(good_student_data, theta, beta, alpha, lower, upper))
    print("Old Model Good Student Validation Accuracy: ",evaluate_2(good_student_data, theta_old, beta_old))
    print("New Model Bad Student Validation Accuracy: ",evaluate(bad_student_data, theta, beta, alpha, lower, upper))
    print("Old Model Bad Student Validation Accuracy: ",evaluate_2(bad_student_data, theta_old, beta_old))

    bad_student_ids = np.argsort(theta)[0:27]
    good_student_ids = np.argsort(theta)[-27:]
    bad_student_data = {"user_id": [], "question_id": [], "is_correct": []}
    good_student_data = {"user_id": [], "question_id": [], "is_correct": []}
    for idx, user_id in enumerate(test_data["user_id"]):
        if user_id in list(bad_student_ids):
            bad_student_data["user_id"].append(user_id)
            bad_student_data["question_id"].append(test_data["question_id"][idx])
            bad_student_data["is_correct"].append(test_data["is_correct"][idx])
        if user_id in list(good_student_ids):
            good_student_data["user_id"].append(user_id)
            good_student_data["question_id"].append(test_data["question_id"][idx])
            good_student_data["is_correct"].append(test_data["is_correct"][idx])
    print("New Model Good Student Test Accuracy: ",evaluate(good_student_data, theta, beta, alpha, lower, upper))
    print("Old Model Good Student Test Accuracy: ",evaluate_2(good_student_data, theta_old, beta_old))
    print("New Model Bad Student Test Accuracy: ",evaluate(bad_student_data, theta, beta, alpha, lower, upper))
    print("Old Model Bad Student Test Accuracy: ",evaluate_2(bad_student_data, theta_old, beta_old))