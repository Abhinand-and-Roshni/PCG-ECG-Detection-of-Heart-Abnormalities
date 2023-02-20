import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("ECG16_EncodedFeatures.csv")
X = data.iloc[:,:-1]
y=data[['Labels']]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state = 21)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape, X.shape

nb = BernoulliNB(fit_prior=False)
nb.fit(X_train, y_train)
probs = nb.predict_proba(X_train)

weights = np.mean(probs, axis=0)
#X_weighted = X * weights
print("1:", weights)
weights = np.resize(weights, (X.shape[0], X.shape[1]))
print("2:", weights)
X_weighted = X * weights

# X_weighted.shape

# weights[0]


X_train,X_test,y_train,y_test=train_test_split(X_weighted, y, test_size=0.2, random_state = 2)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape, X.shape

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression().fit(X_train, y_train)


from sklearn.metrics import accuracy_score
#X_test_weighted = X_test * weights
weights = np.resize(weights, (X_test.shape[0], X_test.shape[1]))
#print(weights)
X_test_weighted = X_test * weights
print(X_test_weighted)
y_pred = LR.predict(X_test_weighted)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for LR-Bayes:", accuracy)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
