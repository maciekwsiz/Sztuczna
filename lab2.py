

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#Zadadnie 1
iris = datasets.load_iris()
print('Wyświetlanie elementów listy:: ', list(iris.keys()))
print('-----')
print('DESCR: zawiera informacje na temat opisu pliku, z którego zbierane są dane', iris['DESCR'])
print('Rodzaj pierwszego elementu z \'DESCR\': ', type(iris['DESCR'][0]))
print('Pięć elementów DESCR', iris['DESCR'][0:5])
print('-----')
print('target: jest tablicą', iris['target'])
print('Rodzaj pierwszego elementu z \'target\': ', type(iris['target'][0]))
print('Pięć elementów celu', iris['target'][0:5])
print('-----')
print('target_names: jest to tablica, która wyświetla typy irysów - setosa, versicolor, virginica', iris['target_names'])
print('Rodzaj pierwszego elementu z \'target_names\': ', type(iris['target_names'][0]))
print('Trzy elementy target_names', iris['target_names'][0:3])
print('-----')
print('feature_names: wyświetla informacje o długości i szerokości przegrody oraz długości i szerokości płatka w cm', iris['feature_names'])
print('Typ pierwszego elementu z \'feature_names\': ', type(iris['feature_names'][0]))
print('Cztery elementy feature_names', iris['feature_names'][0:4])
print('-----')
print('data: jest tablica', iris['data'])
print('Rodzaj pierwszego elementu od \'data\': ', type(iris['data'][0]))
print('Trzy elementy danych', iris['data'][0:3])
print('-----')
print('filename: wyświetla informacje o pliku, z którego pobrano przysłonę i jego lokalizacji na dysku', iris['filename'])
print('Rodzaj pierwszego elementu z \'filename\': ', type(iris['filename'][0]))
print('Piąte elementy z filename', iris['filename'][5])
print('-----')

# Wyświetlanie zestawu danych
iris_list = pd.DataFrame(iris['data'], columns=iris['feature_names'])
targets = map(lambda x: iris['target_names'][x], iris['target'])
iris_list['species'] = np.array(list(targets))
sns.pairplot(iris_list, hue='species')
plt.show()
iris_list.head(3)

#Zadadnie 2

# Dzielę kolekcję na funkcje i etykiety
X = iris.data
y = iris.target

# Korzystam z funkcji, aby podzielić zestaw na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Tworzę klasyfikator k-NN za pomocą parametru 8 neighbours
knn = KNeighborsClassifier(n_neighbors = 8)

# Uczę się klasyfikatora na zestawie do nauki
knn.fit(X_train, y_train)

# Przewiduję wartości dla zestawu testowego
y_pred = knn.predict(X_test)

print('Sprawdzam kilka pierwszych podanych wartości')
print(y_pred[:8])

print('Sprawdzam dokładność klasyfikatora')
print(knn.score(X_test, y_test))

# Tworzę płaszczyznę wszystkich możliwych wartości dla cech 0 i 2, w krokach co 0,1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Uczę klasyfikatora tylko na dwóch wybranych cechach
knn.fit(X_train[:, [0, 2]], y_train)

# Przewiduję każdy punkt 
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

print('Tworzę wykres konturowy')
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
plt.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
plt.show()

# Tworzę listę
list_n = [1,2,3,4,5,6,7,8]
precisions = []
for n_neighbors in list_n:

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    precision = knn.score(X_test, y_test)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    precisions.append(precision)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.bwr)
    plt.scatter(X[:, 0], X[:, 2], c=y, s=20, edgecolor='k')
    plt.show()

print(precisions)
plt.plot(list_n, precisions)
plt.show()