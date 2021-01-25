import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class MyGaussCateNB:
    """
    Using Naive Bayes with Mixed Data and Features
    In this case, i use
    (Gaussian and Categorical)
    Definition:
    'C' stands for Categorical and the other stands for Numerical
    """
    def __init__(self, inst):
        self.col_type = [1 if i == 'C' else 0 for i in inst]  # 0=>> using Gaussian, 1 =>> using Categorical
        self.row = 0
        self.col = 0
        self.prior_probs = {}
        self.classes = []
        self.probs = []
        self.stds = {}
        self.means = {}

    def fit(self, x, y):
        self.row = x.shape[0]
        self.col = len(y)
        self.classes, counts = np.unique(y, return_counts=True)
        self.prior_probs = dict(zip(self.classes, counts / self.col))
        self.probs = [0] * sum(self.col_type)

        # Split to data with categorical and data with gaussian
        idx_gauss = [i for i, v in enumerate(self.col_type) if v == 0]
        idx_cate = [i for i, v in enumerate(self.col_type) if v == 1]
        x_gauss = x[:, idx_gauss]
        x_cate = x[:, idx_cate]

        # processing for gaussian
        for c in self.classes:
            idx = np.argwhere(c == y)
            x_with_class_c = x_gauss[idx, :]
            self.means[c] = np.mean(x_with_class_c, axis=0).flatten()
            self.stds[c] = np.std(x_with_class_c, axis=0).flatten()

        # processing for categorical

        for i in range(x_cate.shape[1]):
            values = np.unique(x_cate[:, i])
            p = [0] * (len(values))
            for v in values:
                p[v] = [0] * (len(self.classes))
                for c in self.classes:
                    p[v][c] = sum((c == y) & (v == x_cate[:, i])) / sum(y == c)
            self.probs[i] = p  # probs is 3D columns, values, class

    def predict(self, x):
        result = []
        for raw in x:
            class_probs = self.__compute_class_probs(raw)
            ans = max(class_probs, key=class_probs.get)
            result.append(ans)
        return result

    def __compute_class_probs(self, x):
        p_of_class = {}
        for c in self.classes:
            p_of_class[c] = self.prior_probs[c]
            j = 0  # use for gaussian
            cnt = 0
            for i, v in enumerate(x):
                # gaussian
                cnt += 1
                if self.col_type[i] == 0:
                    p_of_class[c] *= self.__norm_pdf(v, self.means[c][j], self.stds[c][j])
                    j += 1
                else:
                    # categorical
                    col = i - j
                    val = self.probs[col][v][c]
                    p_of_class[c] *= val
        return p_of_class

    def __norm_pdf(self, x, mean, std):
        exp = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (std * np.sqrt(2 * np.pi))) * exp


# Input data
df = pd.read_csv("../Data/Purchase.csv")

# Preprocessing data
le = LabelEncoder()
for name in ['Gender', 'Age']:
    le.fit(np.unique(df[name]))
    df[name + 'Int'] = le.transform(df[name])
ID = df['User ID']
df.drop('User ID', axis=1, inplace=True)
x_data = df.drop('Purchased', axis=1)
y_data = df['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, test_size=0.2)

# Predict
cg = MyGaussCateNB(['N', 'C', 'C'])
cg.fit(x_data[['EstimatedSalary', 'GenderInt', 'AgeInt']].to_numpy(), y_data.to_numpy())
y_predict = cg.predict(x_data[['EstimatedSalary', 'GenderInt', 'AgeInt']].to_numpy())
print(accuracy_score(y_true=y_data, y_pred=y_predict))

