import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import datasets


class MyGaussianNB:
    def __init__(self):
        self.classes = []
        self.prior_probs = {}
        self.mean = {}
        self.stds = {}
        pass

    def fit(self, x, y):
        sample = len(y)  # <=> X.shape[0]
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior_probs = dict(zip(self.classes, counts/sample))
        self.mean = {}
        self.stds = {}

        for c in self.classes:
            idx = np.argwhere(y == c)  # return index of class cl in y_data
            x_with_class_c = x[idx, :]
            self.mean[c] = np.mean(x_with_class_c, axis=0).flatten()
            self.stds[c] = np.std(x_with_class_c, axis=0).flatten()

    def predict(self, x):
        result = []
        for raw in x:
            class_probs = self.__compute_class_probs(raw)
            c = max(class_probs, key=class_probs.get)
            result.append(c)
        return result

    def __compute_class_probs(self, x):
        p_of_class = {}
        for c in self.classes:
            p_of_class[c] = self.prior_probs[c]
            for i, v in enumerate(x):
                p_of_class[c] *= self.__norm_pdf(v, self.mean[c][i], self.stds[c][i])
        return p_of_class

    def __norm_pdf(self, x, mean, std):
        exp = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (std * np.sqrt(2 * np.pi))) * exp


gnb = MyGaussianNB()
df = datasets.load_iris()
x_data, y_data = df.data, df.target
gnb.fit(x_data, y_data)
yp = gnb.predict(x_data)
print(accuracy_score(y_true=y_data, y_pred=yp))
