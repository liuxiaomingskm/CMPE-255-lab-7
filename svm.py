import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def linear_svm():
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("./bill_authentication.csv")

    # see the data
    bankdata.shape

    # see head
    bankdata.head()

    # data processing
    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # train the SVM
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    # predictions
    y_pred = svclassifier.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames)

    # process
    X = irisdata.drop('Class', axis=1)
    y = irisdata['Class']

    # train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    return  X_train, X_test, y_train, y_test

def polynomial_kernel(X_train, X_test, y_train, y_test):
    # TODO
    # Trains, predicts and evaluates the model
    # train the SVM
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='poly',random_state=42)
    svclassifier.fit(X_train, y_train)

    # predictions
    y_pred = svclassifier.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("========================== POLYNOMIAL KERNEL ==========================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def gaussian_kernel(X_train, X_test, y_train, y_test):
    # TODO
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='rbf',random_state=42)
    svclassifier.fit(X_train, y_train)

    # predictions
    y_pred = svclassifier.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("========================== GAUSSIAN KERNEL ==========================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def sigmoid_kernel(X_train, X_test, y_train, y_test):
    # TODO
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='sigmoid',random_state=42)
    svclassifier.fit(X_train, y_train)

    # predictions
    y_pred = svclassifier.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("========================== SIGMOID KERNEL ==========================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def test():
    X_train, X_test, y_train, y_test=import_iris()
    polynomial_kernel(X_train, X_test, y_train, y_test)
    gaussian_kernel(X_train, X_test, y_train, y_test)
    sigmoid_kernel(X_train, X_test, y_train, y_test)
    # linear_svm()

test()
