import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

            labels = np.argmin(distances, axis=0)

            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

def input_data():
    X = []
    num_samples = int(input("Введите количество обучающих примеров: "))
    for i in range(num_samples):
        print(f"Обучающий пример {i+1}:")
        features = input("Введите признаки через пробел: ").strip().split()
        X.append([float(f) for f in features])
    return np.array(X)



X = input_data()


kmeans = KMeans(n_clusters=2)


kmeans.fit(X)


print("Центроиды:")
print(kmeans.centroids)
