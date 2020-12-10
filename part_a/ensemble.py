# TODO: complete this file.
import numpy as np
from utils import *
from part_a.knn import *
from part_a.item_response import *
from part_a.neural_network import *

def bootstrap_idx(data, num_resamples):
    return data[np.random.randint(0, len(data), (num_resamples, len(data)))]


def knn(matrix, k):
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    return mat

def ensemble_predict(data, knn_matrix, theta, beta, nn_zero_matrix, nn_model, threshold=0.5):

    knn_preds = knn_predict(data, knn_matrix) #>= threshold

    irt_preds = irt_predict(data, theta, beta) #>= threshold

    nn_preds = nn_predict(nn_zero_matrix, data, nn_model) #>= threshold

    return ((knn_preds + irt_preds + nn_preds)/3) >= threshold


def nn_predict(zero_data, valid_data, nn_model):
    nn_model.eval()
    nn_preds = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(zero_data[u]).unsqueeze(0)
        output = nn_model(inputs)

        guess = output[0][valid_data["question_id"][i]].item()
        nn_preds.append(guess)
    nn_preds = np.array(nn_preds)
    return nn_preds


def irt_predict(valid_data, theta, beta):
    irt_preds = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        irt_preds.append(p_a)
    irt_preds = np.array(irt_preds)
    return irt_preds


def knn_predict(valid_data, knn_matrix):
    knn_preds = []
    for i in range(len(valid_data["user_id"])):
        cur_user_id = valid_data["user_id"][i]
        cur_question_id = valid_data["question_id"][i]
        knn_preds.append(knn_matrix[cur_user_id, cur_question_id])
    knn_preds = np.array(knn_preds)
    return knn_preds


def evaluate(data, pred):
    return np.sum((data["is_correct"] == np.array(pred))) \
    / len(data["is_correct"])






def main():

    # Load data
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #
    # Set up new datasets
    num_resamples = 3
    resampled_train_matrix = bootstrap_idx(train_matrix, num_resamples)

    # resampled_train_matrix = np.concatenate([train_matrix, train_matrix, train_matrix]).reshape(3, len(train_matrix), -1)
    # KNN
    knn_matrix = knn(resampled_train_matrix[0], k=11)

    # Item Response Theory
    lr = 0.01
    iterations = 20
    theta, beta, val_log_likelihood, train_log_likelihood = \
        irt(resampled_train_matrix[1], train_data, val_data, lr, iterations)

    # Neural Network
    k = 100
    lr = 0.01
    epoch = 40
    lamb = 0.01

    nn_zero_train_matrix = resampled_train_matrix[2].copy()
    nn_zero_train_matrix[np.isnan(resampled_train_matrix[2])] = 0
    nn_zero_train_matrix = torch.FloatTensor(nn_zero_train_matrix)
    nn_train_matrix = torch.FloatTensor(resampled_train_matrix[2])

    nn_model = AutoEncoder(resampled_train_matrix[2].shape[1], k)
    epochs, train_losses, val_accs = train(nn_model, lr, nn_train_matrix, nn_zero_train_matrix,
                                           val_data, epoch, lamb)

    threshold = 0.5

    ensemble_train_pred = ensemble_predict(train_data, knn_matrix, theta, beta, nn_zero_train_matrix, nn_model, threshold)
    ensemble_train_acc = evaluate(train_data, ensemble_train_pred)

    ensemble_val_pred = ensemble_predict(val_data, knn_matrix, theta, beta, nn_zero_train_matrix, nn_model, threshold)
    ensemble_val_acc = evaluate(val_data, ensemble_val_pred)

    ensemble_test_pred = ensemble_predict(test_data, knn_matrix, theta, beta, nn_zero_train_matrix, nn_model, threshold)
    ensemble_test_acc = evaluate(test_data, ensemble_test_pred)

    print("Ensemble Train accuracy: ", ensemble_train_acc)
    print("Ensemble Validation accuracy: ", ensemble_val_acc)
    print("Ensemble Test accuracy: ", ensemble_test_acc)

    # Comparison with Individual Accuracy
    train_knn_pred = knn_predict(train_data, knn_matrix) >= threshold
    train_irt_pred = irt_predict(train_data, theta, beta) >= threshold
    train_nn_pred = nn_predict(nn_zero_train_matrix, train_data, nn_model) >= threshold

    val_knn_pred = knn_predict(val_data, knn_matrix) >= threshold
    val_irt_pred = irt_predict(val_data, theta, beta) >= threshold
    val_nn_pred = nn_predict(nn_zero_train_matrix, val_data, nn_model) >= threshold

    test_knn_pred = knn_predict(test_data, knn_matrix) >= threshold
    test_irt_pred = irt_predict(test_data, theta, beta) >= threshold
    test_nn_pred = nn_predict(nn_zero_train_matrix, test_data, nn_model) >= threshold

    knn_train_acc = evaluate(train_data, train_knn_pred)
    knn_val_acc = evaluate(val_data, val_knn_pred)
    knn_test_acc = evaluate(test_data, test_knn_pred)
    print("KNN Train accuracy: ", knn_train_acc)
    print("KNN Validation accuracy: ", knn_val_acc)
    print("KNN Test accuracy: ", knn_test_acc)

    irt_train_acc = evaluate(train_data, train_irt_pred)
    irt_val_acc = evaluate(val_data, val_irt_pred)
    irt_test_acc = evaluate(test_data, test_irt_pred)
    print("IRT Train accuracy: ", irt_train_acc)
    print("IRT Validation accuracy: ", irt_val_acc)
    print("IRT Test accuracy: ", irt_test_acc)

    nn_train_acc = evaluate(train_data, train_nn_pred)
    nn_val_acc = evaluate(val_data, val_nn_pred)
    nn_test_acc = evaluate(test_data, test_nn_pred)
    print("NN Train accuracy: ", nn_train_acc)
    print("NN Validation accuracy: ", nn_val_acc)
    print("NN Test accuracy: ", nn_test_acc)










if __name__ == "__main__":
    main()


