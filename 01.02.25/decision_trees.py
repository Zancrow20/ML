from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



def main():
    # Чтение CSV файла
    df = pd.read_csv('AmesHousing.csv', sep=',')
    X = pd.DataFrame()
    X["Lot Frontage"] = df["Lot Frontage"]
    X["Lot Area"] = df["Lot Area"]
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    # score отрицательный, потому что в правой части > 1 из-за того, что знаменатель np.sum((y_test-np.mean(y_test))**2) < np.sum((y_test-y_pred)**2)
    # хотим score = 1, тогда нам надо np.sum((y_test-y_pred)**2) = 0 - лучший вариант (предсказания равны изначальным данным)
    score = 1-np.sum((y_test-y_pred)**2)/np.sum((y_test-np.mean(y_test))**2)
    print(score)

if __name__ == '__main__':
    main()