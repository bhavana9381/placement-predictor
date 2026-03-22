import csv
import io
import random

def load_data(file):
    X = []
    Y = []

    # Convert bytes → string
    file = io.StringIO(file.getvalue().decode("utf-8"))

    reader = csv.reader(file)
    next(reader)

    for row in reader:
        if len(row) == 0:
            continue

        try:
            features = list(map(float, row[:-1]))
            label = int(row[-1])

            X.append(features)
            Y.append(label)

        except:
            continue

    return X, Y


def accuracy(model, X, Y):
    correct = 0

    for i in range(len(X)):
        if model.predict(X[i]) == Y[i]:
            correct += 1

    return correct / len(X) if len(X) > 0 else 0


def train_test_split(X, Y, test_ratio=0.2):
    data = list(zip(X, Y))
    random.shuffle(data)

    split = int(len(data) * (1 - test_ratio))

    train_data = data[:split]
    test_data = data[split:]

    X_train, Y_train = zip(*train_data)
    X_test, Y_test = zip(*test_data)

    return list(X_train), list(Y_train), list(X_test), list(Y_test)