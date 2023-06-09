import pandas as pd
import numpy as np
from math import sqrt
import statistics
import matplotlib.pyplot as plt

df_train = pd.read_csv("pendigits_training.txt", header=None, delim_whitespace=True)

print(df_train.head())

df_test = pd.read_csv("pendigits_test.txt", header=None, delim_whitespace=True)

print(df_test.head())

mean_train = []
std_train = []
for col_train in range(len(df_train.columns) - 1):
    mean = np.mean(df_train[col_train])
    std = np.std(df_train[col_train])
    df_train[col_train] = (df_train[col_train] - mean) / std
    mean_train.append(mean)
    std_train.append(std)

for col_test in range(len(df_test.columns) - 1):
    df_test[col_test] = (df_test[col_test] - mean_train[col_test]) / std_train[col_test]

print(df_train.head())

print(df_test.head())

X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values


def euclidean_distance(test_col, train_col):
    dist = 0.0
    for i in range(len(test_col)):
        dist += (test_col[i] - train_col[i]) ** 2
    dist = sqrt(dist)
    return dist


def handle_tie(neighbours):
    equal = neighbours.count(neighbours[0])
    flag = ''
    for i in range(len(neighbours)):
        if neighbours.count(neighbours[i]) == equal:
            flag = 'E'
        else:
            flag = 'N'
    return flag


def predict(k, X_train, y_train, X_test, y_test):
    y_predict = np.zeros(len(X_test))
    y_predict = y_predict.astype('int')
    for i in range(len(X_test)):
        distances = []
        for j in range(len(X_train)):
            dist = euclidean_distance(X_test[i], X_train[j])
            distances.append((dist, j))

        distances.sort(key=lambda x: x[0])
        index = [x[1] for x in distances]
        index = np.array(index)
        y_train_sorted = y_train[index]
        neighbours = y_train_sorted[:k]
        neighbours = neighbours.tolist()
        flag = handle_tie(neighbours)
        if k % 2 == 0:
            if flag == 'E':
                y_predict[i] = neighbours[0]
            else:
                y_predict[i] = statistics.mode(neighbours)
        else:
            y_predict[i] = statistics.mode(neighbours)
        print("Row {} ----> Actual label = {}, Predicted label = {}".format(i, y_test[i], y_predict[i]))
    return y_predict


def accuracy(y_actual, y_predict):
    correct_classified_points = 0
    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i]:
            correct_classified_points += 1
    acc = (correct_classified_points / len(y_actual)) * 100.0
    print("Correct classified instances = {}, and total numbers of instances = {}, accuracy = {}".format(correct_classified_points, len(y_actual), acc))
    return acc, correct_classified_points, len(y_actual)


def display(X_train, y_train, X_test, y_test, k):
    dict = {}
    for k in k:
        print("k = ",k)
        y_pred  = predict(k, X_train,y_train, X_test, y_test)
        acc, ccp, total = accuracy(y_test, y_pred)
        print("#########################################################################################################")
        print("#########################################################################################################")
        dict[k] = [acc, ccp, total]
    return dict


k = [*range(1, 10)]
dict = display(X_train, y_train, X_test, y_test, k)
acc_list = []
for key in dict.keys():
    acc_list.append(dict[key][0])

plt.plot(k, acc_list, color='orange', marker='o')
plt.xlabel("K number")
plt.ylabel("accuracy")
plt.savefig("accuracy of each k.png")
plt.show()

f = open("knn_output.txt",'w')

for key in dict.keys():
    f.write("K = {} ------> accuracy = {},  Correct classified instances = {}, Total instances = {} \n".format(key, dict[key][0], dict[key][1], dict[key][2]))

f.write("As K increases, the KNN fits a smoother curve to the data. This is because a higher value of K reduces the edginess by taking more data \n"
        "into account, thus reducing the overall complexity and flexibility of the model, increasing the value of K improves the score to \n"
        "a certain point, after which it again starts dropping. \n")
f.write("In our example, the model yields the best results at K=4.")

f.close()
