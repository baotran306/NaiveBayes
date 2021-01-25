import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import CategoricalNB


class MyCategoricalNB:
    # X: input with shape(samples, features)
    # y_data: row vector
    def __init__(self):
        self.row = 0
        self.col = 0
        self.classes = {}
        self.prior_classes = []
        self.probs = []
        pass

    def fit(self, x, y):
        self.row = x.shape[0]
        self.col = x.shape[1]
        self.classes = np.unique(y)
        self.prior_classes = [sum(col == y) / len(y) for col in self.classes]
        self.probs = [0] * self.col
        for i in range(self.col):
            values = np.unique(x[:, i])
            p = [0] * self.col
            for v in values:
                p[v] = [0] * len(values)
                for c in self.classes:
                    p[v][c] = sum((v == x[:, i]) & (c == y)) / sum(c == y)
            self.probs[i] = p  # self.probs is 3D (column, value, class) p[i][v][cl]

    def predict(self, x):
        result = []
        for raw in x:
            class_probs = self.__compute_class_probs(raw)
            c = max(class_probs, key=class_probs.get)
            result.append(c)
        return ["Yes" if i == 1 else "No" for i in result]

    def __compute_class_probs(self, x):
        p_of_class = {}
        for c in self.classes:
            posterior = 1
            for i in range(self.col):
                v = x[i]
                posterior *= self.probs[i][v][c]
            p_of_class[c] = posterior * self.prior_classes[c]
        return p_of_class


nbc = MyCategoricalNB()
df = pd.read_csv('../Data/PlayTennis.txt', delimiter='\t')
print(df)
x_data = df.drop('Play Tennis', axis=1)
y_data = df['Play Tennis']

le = preprocessing.LabelEncoder()
for cl in ['Outlook', 'Temperature', 'Humidity', 'Wind']:
    le.fit(np.unique(x_data[cl]))
    x_data[cl + 'Int'] = le.transform(x_data[cl])

x_train = x_data[['OutlookInt', 'TemperatureInt', 'HumidityInt', 'WindInt']]
y_data = le.fit(np.unique(y_data)).transform(y_data)
print(y_data)
nbc.fit(x_train.to_numpy(), y_data)
test = nbc.predict([[2, 0, 0, 0], [0, 1, 1, 1]])  # ['Sunny','Cool','High','Strong']['Overcast,'Hot','Normal','Weak']
print("----predict----")
print(test)
print("----Probs All----")
print(nbc.probs)
print("---- Probs of Class(no/(yes+no) and yes/(yes+no)) ----")
print(nbc.prior_classes)

# Using sklearn
print("---Predict using sklearn----")
clf = CategoricalNB()
clf.fit(x_train, y_data)
print(clf.predict([[2, 0, 0, 0]]))
