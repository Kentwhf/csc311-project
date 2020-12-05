from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    # print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def plot_accuracy(k_list, acc, filename, title):
    plt.plot(k_list, acc)
    plt.xlabel("k")
    plt.xticks(k_list)
    plt.grid(axis='x', color='0.95')
    plt.title(title)
    plt.savefig("../figs/" + filename + ".png")
    plt.show()


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    k_list = [1, 6, 11, 16, 21, 26]
    train_acc_by_user, val_acc_by_item = [], []

    for k in k_list:
        print("K: " + str(k))

        res = knn_impute_by_user(sparse_matrix, val_data, k)
        print("User based collaborative filtering: " + str(res))
        train_acc_by_user.append(res)

        res = knn_impute_by_item(sparse_matrix, val_data, k)
        print("Item based collaborative filtering: " + str(res))
        val_acc_by_item.append(res)

    plot_accuracy(k_list, train_acc_by_user, "Q1a", "User based validation accuracy")
    plot_accuracy(k_list, val_acc_by_item, "Q1c", "Item based validation accuracy")

    # Final test accuracy
    best_k = 11
    print(knn_impute_by_user(sparse_matrix, test_data, best_k))

    best_k = 21
    print(knn_impute_by_item(sparse_matrix, test_data, best_k))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
