
import requests
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt - is not working for MAC OS X, need to add:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#getCurrencyAndDate function which return currencyCource
def getCurrencyAndDate(currency, dateFrom, dateTo):
    getDataFromUrl = requests.get('http://api.nbp.pl/api/exchangerates/rates/A/'+currency+'/'+dateFrom+'/'+dateTo).json()
    currencyCource = pd.DataFrame.from_dict(getDataFromUrl['rates'])
    currencyCource['effectiveDate'].astype('datetime64')
    currencyCource.set_index('effectiveDate', inplace=True)
    return currencyCource

#creating usdTable and eurTable
usdTable = getCurrencyAndDate('USD','2019-09-01','2019-09-30')
eurTable = getCurrencyAndDate('EUR','2019-09-01','2019-09-30')

#printing first 7 rows of the tables
print('------waluta usd-------')
print(usdTable.head(7))
print('------waluta eur-------')
print(eurTable.head(7))

#printing .info() and .describe() methods for USD course
print()
print('----- info () i opis () dla USD ------')
print(usdTable.info())
print(usdTable.describe())

#printing .info() and .describe() methods for EUR course
print()
print('----- info() i opis() dla EUR ------')
print(eurTable.info())
print(eurTable.describe())

#cleaning: deleting no
usdTable = getCurrencyAndDate('USD','2019-09-01','2019-09-30')
eurTable = getCurrencyAndDate('EUR','2019-09-01','2019-09-30')
usdTableCleaned = usdTable['mid'].head(7)
eurTableCleaned = eurTable['mid'].head(7)

#printing cleaned tables
print()
print('---- usdTable po usunięciu nie---- ')
print(usdTableCleaned)
print()
print('---- eurTable po usunięciu nie ---- ')
print(eurTableCleaned)

#printing chart of usdTableCleaned
plt.plot(usdTableCleaned)
print('---- wykres waluty usd ----')
plt.show()

#printing chart of eurTableCleaned
plt.plot(eurTableCleaned)
print('---- wykres waluty eur ----')
plt.show()

#printing correlation of usdTableCleaned
print('---- wykres z korelacją kursów -----')
print(np.corrcoef(usdTableCleaned, usdTableCleaned))

#drawing 2 graphs of courses
draw, (drawUsd, drawEur) = plt.subplots(2, sharex=True)
draw.suptitle('pomaranczowy - USD line , zielony - EUR line')
drawUsd.plot(usdTableCleaned, 'tab:pomaranczowy')
drawEur.plot(usdTableCleaned, 'tab:zielony')
plt.show()

