
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Zadanie 1

cars = fetch_openml('cars1')

# wizualizacja wszystkich funkcji
print(cars.keys())
print(cars['target'])
print(cars['DESCR'])
print(cars['feature_names'])
print(cars['url'])
print(cars['data'][0])
print(cars['categories'])
print(cars['details'])

# Dzielę kolekcję na funkcje i etykiety
# Wybieram funkcje 1 i 5
X = cars.data[:, [0, 4]]
print(X)
y = cars['target']
y = [int(elem) for elem in y]
print(y)

# Korzystam z funkcji, aby podzielić zestaw na zestaw treningowy i zestaw testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Tworzę klasyfikator z czterema klastrami (klasami)
kmn = KMeans(n_clusters=4)

# Uczę klasyfikatora danych treningowych
kmn.fit(X_train)

# Wydobywam punkty skupienia klastra
# Pokażę je na wykresie obok punktów z zestawu treningowego
central = kmn.cluster_centers_
fig, ax = plt.subplots(1, 2)

# pierwszy wykres to nasz zestaw do nauki, z prawdziwymi klasami
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)

# Teraz używam danych treningowych, aby sprawdzić, co klasyfikator o nich myśli
y_pred_train = kmn.predict(X_train)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=20)

# Dodam centra skupień na drugim wykresie
ax[1].scatter(central[:, 0], central[:, 1], c='red', s=50)
plt.show()

# Próbuję przewidzieć klasy samochodów dla zestawu testowego
y_pred = kmn.predict(X_test)

# Nowe klasy samochodów zapewniane przez klastrowanie
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20)

# Jak wyżej, wyświetlam centra klastrów
plt.scatter(central[:, 0], central[:, 1], c='red', s=50)
plt.show()

#  Zad. 2

# W klastrach przedtawilam samoochody w roznych kombinacjach (moc/spalanie silnika i zasieg).
# Klaster najwyzszy to samochody o duzej mocy i krotkim zasiegu.
# Klaster najnizszy to samochody o malej mocy i dalekim zasiegu.
