import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

df = pd.read_csv("bikes_rent.csv", sep=',')
# df.drop(columns=['cnt']) не меняет данные, а создает новый dataframe

# кросс-валидация - часть данных отводится под тесты и часть под тренировку
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['cnt']), df['cnt'])

# проверка корреляции признаков и удаление их из таблицы
prop_corr = df.corr()
# print(prop_corr)
sns.heatmap(prop_corr, annot=True, cmap='coolwarm')
# plt.show()
cnt = df.cnt
df = df.drop(columns = ["season", "atemp", "windspeed(mph)"])
df = (df-df.min())/(df.max() - df.min())
df.cnt = cnt
print(np.max(df['cnt']))

# linear regression and prediction
lr = LinearRegression()
lr.fit(X_train, y_train)
prediction = lr.predict(X_test)

#

# считаем ошибку предсказания
rmse = np.sqrt(np.mean((prediction-y_test)**2))
print(rmse)
print(np.max(df['cnt']))