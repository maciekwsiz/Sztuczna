from sklearn.datasets import load_boston, load_diabetes
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

#zad. 1
# Załadowanie zestawu cech nieruchomości i ich cen
propertyOfBoston = load_boston()
# konwersja do pandas.DataFrame
listPropertyOfBoston = pd.DataFrame(propertyOfBoston['data'], columns=propertyOfBoston['feature_names'])
print('listPropertyOfBoston:')
print(listPropertyOfBoston)
# dołączanie informacji o cenie do reszty dataframe
listPropertyOfBoston['target'] = np.array(list(propertyOfBoston['target']))

# drukowanie danych
print('')
print('Przykładowe wartości funkcji:')
print(propertyOfBoston.data[:3])
print('')
print('Przykładowe wartości:')
print(propertyOfBoston.target[:3])
print('')
print('Elementy zestawu:')
print(list(propertyOfBoston.keys()))
print('')
print('Klucze w zestawie danych:')
print(propertyOfBoston.keys())
print('')
print('propertyOfBoston.DESCR')
print(propertyOfBoston.DESCR)

# 
rooms = propertyOfBoston['data'][:, np.newaxis, 3]
plt.scatter(rooms, propertyOfBoston['target'])
plt.show()

# Stworzenie regresora liniowego
linreg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(rooms, propertyOfBoston['target'], test_size = 0.3)
linreg.fit(X_train, y_train)

# prognoza cen
y_pred = linreg.predict(X_test)

# domyślna metryka
print('Default metric: ', linreg.score(X_test, y_test))

# wskaźnik (metryczny) r^2
print('Metric r2: ', r2_score(y_test, y_pred))

# współczynniki regresji
print('Regression coefficients', linreg.coef_)

# wykres regresji
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()

# Metryka 1
cv_score_r2 = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='r2')
print('metryka 1')
print(cv_score_r2)
print('')

# Metryka 2
cv_score_ev = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='explained_variance')
print('metryka 2')
print(cv_score_ev)
print('')

# Metryka 3
cv_score_mse = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='neg_mean_squared_error')
print('metryka 3')
print(cv_score_mse)
print('')

# Metryka 4
max_error = cross_val_score(linreg, rooms, propertyOfBoston.target, cv=5, scoring='neg_mean_squared_error')
print('metryka 4')
print(max_error)

#zad. bonus

diabetics = load_diabetes()
listOfDiabetics = pd.DataFrame(diabetics['data'], columns=diabetics['feature_names'])
listOfDiabetics['target'] = np.array(list(diabetics['target']))

print('Elements of the set:')
print(list(diabetics.keys()))
print('Keys in the data set:')
print(diabetics.keys())
print('diabetics.DESCR')
print(diabetics.DESCR)


#wiek diabetyków
age = diabetics['data'][:, np.newaxis, 0]
plt.scatter(age, diabetics['target'])
plt.show()

# tworzenie regresora liniowego
linreg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(age, diabetics['target'], test_size = 0.5)
linreg.fit(X_train, y_train)

# prognozowanie cukrzycy według wieku
y_pred = linreg.predict(X_test)

# domyślna metryka
print('default metric', linreg.score(X_test, y_test))

# wskaźnik (metryczny) r ^ 2
print('metric r2: ', r2_score(y_test, y_pred))

# współczynniki regresji
print('regression coefficients', linreg.coef_)

# wykres regresji
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=2)
plt.show()

# Metryka 1
cv_score_r2 = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='r2')
print('cv_score_r2')
print(cv_score_r2)
print('')

# Metryka 2
cv_score_ev = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='explained_variance')
print('cv_score_ev')
print(cv_score_ev)
print('')

# Metryka 3
cv_score_mse = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='neg_mean_squared_error')
print('cv_score_mse')
print(cv_score_mse)
print()

# Metryka 4
max_error = cross_val_score(linreg, age, diabetics.target, cv=5, scoring='max_error')
print('max_error')
print(max_error)
print('')