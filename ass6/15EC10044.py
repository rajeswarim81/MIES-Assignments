from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn import datasets

data=datasets.load_iris()
naive_bayes=GaussianNB()
naive_bayes.fit(data.data, data.target)
y_true=data.target
y_pred=naive_bayes.predict(data.data)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


