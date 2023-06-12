import pandas as pd
import numpy as np
# 1
df = pd.read_csv("Weather.csv")
#print(df)

#2
#print(f"Кількість записів: {len(df)}\nКількість полів: {len(df.columns)}")

#3
M = 10
#print(df.iloc[M: M+5], df.iloc[::300*M], sep='\n')

#4
#print(df.dtypes)

#5
d = df['CET']
years, month, day = [], [], []

for i in d:
    i = i.split('-')
    i = ['0' + j if len(j) == 1 else j for j in i]
    years.append(i[0])
    month.append(i[1])
    day.append(i[2])
df.insert(1, "Year", years)
df.insert(2, "Month", month)
df.insert(3, "Day", day)

#6
#a
#nan_count = df[" Events"].isna().sum()
#print('6a. Кількість днів із порожнім значенням: ', nan_count)
#b
#df = df.sort_values(by=[" Mean Humidity", " Mean Wind SpeedKm/h"])
#print("6b. День коли середня вологість мінімальна:\n", df[["Day", " Mean Humidity", " Mean Wind SpeedKm/h"]].iloc[:1])
#c

#print("6c. Місяці, коли середня температура від нуля до п’яти градусів:\n")
#l = df.groupby(["Year", "Month"], as_index=False)["Day"].max().astype('int64')
#h = df[df["Mean TemperatureC"].isin(range(0, 6))].groupby(["Year", "Month"], as_index=False)["Day"].count().astype('int64')
#print(l.where(l == h["Day"]).dropna())
#7
#a
#print("Середня максимальна температура по кожному дню за всі роки", df.groupby(["Day"])["Max TemperatureC"].mean())

#b
#print("Кількість днів у кожному році з туманом:\n", df[df[' Events'].str.contains('Fog', na=False)].groupby("Year")[" Events"].count())

#8
from matplotlib import pyplot as plt
#df[' Events'].hist()
#plt.gca().spines['bottom'].set_position(('data', 0))
#plt.title("Кількість Events")
#plt.xlabel('Name Events')
#plt.ylabel('Number of each event')
#plt.xticks(rotation=45)
#plt.show()

#9
#labels = ['North', 'North-East', 'East', 'South-East', 'South', 'South-West', 'West', 'North-West']
#def num_filter(step):
#    return df[df['WindDirDegrees'].isin(range(45 * step-22, 45*step+23))]['WindDirDegrees'].count()
#
#q = [num_filter(i) for i in [c for c in range(0, 8)]]
#q[0] += num_filter(8)
#plt.title("Діаграма напрямків вітру")
#plt.pie(q, labels=labels)
#plt.show()

#10
#a Середню по кожному місяцю кожного року максимальну температуру;
#b Середню по кожному місяцю кожного року мінімальну точку роси.
#import matplotlib.pyplot as plt
#
#a = df.groupby(["Year", "Mounth"], as_index=False)['Max TemperatureC'].mean()
#b = df.groupby(["Year", "Mounth"], as_index=False)['Min DewpointC'].mean()
#
#z = a['Year'].astype('int64')
#x = a["Mounth"].astype('int64')
#y = a['Max TemperatureC'].astype('int64')
#z2 = b['Year'].astype('int64')
#x2 = b["Mounth"].astype('int64')
#y2 = b['Min DewpointC'].astype('int64')
#
#fig = plt.figure()
#ax = plt.axes(projection="3d")
#ax.scatter3D(z, x, y, color="green",  label='Max TemperatureC')
#ax.scatter3D(z2, x2, y2, color="blue", label='Min DewpointC')
#ax.legend()
#plt.title(' Середню по кожному місяцю кожного року \n максимальну температуру/ мінімальну точку роси')
#plt.xlabel('Рік')
#plt.ylabel('Місяць')
#ax.set_zlabel('Точка роси/Температура', rotation=60)
#plt.show()