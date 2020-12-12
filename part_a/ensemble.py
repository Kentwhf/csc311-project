# TODO: complete this file.
import numpy as np
from utils import *
from part_a.knn import *
from part_a.item_response import *
from part_a.neural_network import *


def sample_by_student_only(data, num_resamples):

    # Sample by student
    resampled_data = []

    u_id_arr = np.array(data["user_id"])
    q_id_arr = np.array(data["question_id"])
    c_id_arr = np.array(data["is_correct"])

    for _ in range(num_resamples):
        rand_user_id = np.random.randint(0, len(np.unique(data["user_id"])), len(np.unique(data["user_id"])))
        sampled_dict = {"user_id": [], "question_id": [], "is_correct": []}
        for user_id in rand_user_id:
            sampled_dict["user_id"].extend(list(u_id_arr[u_id_arr == user_id]))
            sampled_dict["question_id"].extend(list(q_id_arr[u_id_arr == user_id]))
            sampled_dict["is_correct"].extend(list(c_id_arr[u_id_arr == user_id]))
        resampled_data.append(sampled_dict)
    return resampled_data



def sample_by_student_question(data, num_resamples):

    # Sample by student
    resampled_data = []
    u_id_arr = np.array(data["user_id"])
    q_id_arr = np.array(data["question_id"])
    c_id_arr = np.array(data["is_correct"])

    for _ in range(num_resamples):
        rand_id = np.random.randint(0, len(data["user_id"]), len(data["user_id"]))
        sampled_dict = {}
        sampled_dict["user_id"] = u_id_arr[rand_id]
        sampled_dict["question_id"] = q_id_arr[rand_id]
        sampled_dict["is_correct"] = c_id_arr[rand_id]
        resampled_data.append(sampled_dict)
    return resampled_data


# def knn(matrix, k):
#     nbrs = KNNImputer(n_neighbors=k)
#     # We use NaN-Euclidean distance measure.
#     mat = nbrs.fit_transform(matrix)
#     return mat
#
# def ensemble_predict_by_avg(data, knn_matrix, theta, beta, nn_zero_matrix, nn_model, threshold=0.5):
#
#     knn_preds = knn_predict(data, knn_matrix) #>= threshold
#     irt_preds = irt_predict(data, theta, beta) #>= threshold
#     nn_preds = nn_predict(nn_zero_matrix, data, nn_model) #>= threshold
#     return ((knn_preds + irt_preds + nn_preds)/3) >= threshold
#
#
# def ensemble_predict_by_majority(data, knn_matrix, theta, beta, nn_zero_matrix, nn_model, threshold=0.5):
#
#     knn_preds = knn_predict(data, knn_matrix) >= threshold
#     irt_preds = irt_predict(data, theta, beta) >= threshold
#     nn_preds = nn_predict(nn_zero_matrix, data, nn_model) >= threshold
#     return ((knn_preds.astype(int) + irt_preds.astype(int) + nn_preds.astype(int))/3) >= threshold


def ensemble_predict_irt(data, theta_list, beta_list, majority=False, threshold=0.5):
    ensemble_pred = np.zeros(len(data["is_correct"]))
    for idx, theta in enumerate(theta_list):
        temp_pred = irt_predict(data, theta, beta_list[idx])

        if majority:
            temp_pred = (temp_pred >= threshold).astype(int)

        ensemble_pred += temp_pred
    return ((ensemble_pred/len(theta_list)) >= threshold).astype(int)


# def nn_predict(zero_data, valid_data, nn_model):
#     nn_model.eval()
#     nn_preds = []
#     for i, u in enumerate(valid_data["user_id"]):
#         inputs = Variable(zero_data[u]).unsqueeze(0)
#         output = nn_model(inputs)
#
#         guess = output[0][valid_data["question_id"][i]].item()
#         nn_preds.append(guess)
#     nn_preds = np.array(nn_preds)
#     return nn_preds


def irt_predict(valid_data, theta, beta, binary=False, threshold=0.5):
    irt_preds = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        irt_preds.append(p_a)
    irt_preds = np.array(irt_preds)
    if binary:
        return (irt_preds >= threshold).astype(int)
    return irt_preds

#
# def knn_predict(valid_data, knn_matrix):
#     knn_preds = []
#     for i in range(len(valid_data["user_id"])):
#         cur_user_id = valid_data["user_id"][i]
#         cur_question_id = valid_data["question_id"][i]
#         knn_preds.append(knn_matrix[cur_user_id, cur_question_id])
#     knn_preds = np.array(knn_preds)
#     return knn_preds


def evaluate(data, pred):
    return np.sum((np.array(data["is_correct"]) == np.array(pred))) / len(data["is_correct"])


def main():

    # Load data
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Set up new datasets
    num_resamples = 3
    resampled_train_data = sample_by_student_question(train_data, num_resamples)

    # Ensemble with 3 Item Response Theory Models
    lr = 0.01
    iterations = 20

    theta_list = []
    beta_list = []

    for i in range(num_resamples):
        print(f"Training Model {i}")
        theta, beta, val_log_likelihood, train_log_likelihood = \
            irt(resampled_train_data[i], val_data, lr, iterations)
        theta_list.append(theta)
        beta_list.append(beta)

    # Individual model performance
    individual_train_acc = []
    individual_val_acc = []
    individual_test_acc = []

    for idx in range(num_resamples):
        train_prediction = irt_predict(resampled_train_data[idx], theta_list[idx], beta_list[idx], binary=True)
        val_prediction = irt_predict(val_data, theta_list[idx], beta_list[idx], binary=True)
        test_prediction = irt_predict(test_data, theta_list[idx], beta_list[idx], binary=True)

        # print(f"Model {idx} Train accuracy: ", evaluate(resampled_train_data[idx], train_prediction))
        # print(f"Model {idx} Validation accuracy: ", evaluate(val_data, val_prediction))
        # print(f"Model {idx} Test accuracy: ", evaluate(test_data, test_prediction))

        individual_train_acc.append(evaluate(resampled_train_data[idx], train_prediction))
        individual_val_acc.append(evaluate(val_data, val_prediction))
        individual_test_acc.append(evaluate(test_data, test_prediction))

    print(f"Individual Avg Train accuracy: ", np.mean(individual_train_acc))
    print(f"Individual Avg Validation accuracy: ", np.mean(individual_val_acc))
    print(f"Individual Avg Test accuracy: ", np.mean(individual_test_acc))


    # ensemble_train_majority = ensemble_predict_irt(train_data, theta_list, beta_list, majority=True)
    ensemble_train_avg = ensemble_predict_irt(train_data, theta_list, beta_list, majority=False)

    # ensemble_val_majority = ensemble_predict_irt(val_data, theta_list, beta_list, majority=True)
    ensemble_val_avg = ensemble_predict_irt(val_data, theta_list, beta_list, majority=False)

    # ensemble_test_majority = ensemble_predict_irt(test_data, theta_list, beta_list, majority=True)
    ensemble_test_avg = ensemble_predict_irt(test_data, theta_list, beta_list, majority=False)

    # print("Ensemble Train Majority accuracy: ", evaluate(train_data, ensemble_train_majority))
    print("Ensemble Train Average accuracy: ", evaluate(train_data, ensemble_train_avg))

    # print("Ensemble Validation Majority accuracy: ", evaluate(val_data, ensemble_val_majority))
    print("Ensemble Validation Average accuracy: ", evaluate(val_data, ensemble_val_avg))

    # print("Ensemble Test Majority accuracy: ", evaluate(test_data, ensemble_test_majority))
    print("Ensemble Test Average accuracy: ", evaluate(test_data, ensemble_test_avg))


if __name__ == "__main__":
    main()


