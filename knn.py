import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for i in range(len(X_test)):
            distances = np.sqrt(np.sum(np.square(self.X_train - X_test[i]), axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.y_train[j] for j in nearest_indices]
            most_common = Counter(nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions


def input_training_data():
    X_train = []
    y_train = []
    num_samples = int(input("Введите количество обучающих примеров: "))
    for i in range(num_samples):
        print(f"Обучающий пример {i+1}:")
        features = input("Введите признаки через пробел: ").strip().split()
        label = int(input("Введите метку класса: "))
        X_train.append([float(f) for f in features])
        y_train.append(label)
    return np.array(X_train), np.array(y_train)




X_train, y_train = input_training_data()


knn = KNN(k=3)


knn.fit(X_train, y_train)


X_test = np.array([[4, 5], [6, 7]])


predictions = knn.predict(X_test)
print("Predictions:", predictions)
