# Importarea bibliotecilor necesare
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# 1. Încărcarea setului de date
# Fișierele trebuie descărcate manual și plasate în același director cu acest script
day_data = pd.read_csv('day.csv')
hour_data = pd.read_csv('hour.csv')

# 2. Preprocesarea datelor
# a) Vizualizarea datelor
print(day_data.head())
print(hour_data.head())

# b) Vizualizarea statisticilor descriptive
print(day_data.describe())
print(hour_data.describe())

# c) Tratarea valorilor lipsă
print(day_data.isnull().sum())
print(hour_data.isnull().sum())
# Nu sunt valori lipsă în setul de date conform documentației

# d) Detectarea și eliminarea outlierilor
# Folosind metoda IQR pentru a elimina outlierii
numeric_columns = day_data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    Q1 = day_data[col].quantile(0.25)
    Q3 = day_data[col].quantile(0.75)
    IQR = Q3 - Q1
    day_data = day_data[~((day_data[col] < (Q1 - 1.5 * IQR)) | (day_data[col] > (Q3 + 1.5 * IQR)))]

# 3. Elaborarea modelelor de regresie
# Pregătirea datelor pentru antrenare
X = day_data[['temp', 'hum', 'windspeed']]
y = day_data['cnt']

# Împărțirea setului în antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a) Modelul de regresie liniară
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)
y_pred_lin = lin_reg.predict(X_test_scaled)

# b) Modelul k-NN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# 4. Compararea performanțelor modelelor
mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print(f'Regresie Liniară - MSE: {mse_lin}, R2: {r2_lin}')
print(f'k-NN - MSE: {mse_knn}, R2: {r2_knn}')

# 5. Interfață Streamlit
st.title('Bike Sharing Prediction')
st.write('### Comparația modelelor de regresie')

# Afișarea rezultatelor în interfață
st.write(f'Regresie Liniară - MSE: {mse_lin:.2f}, R2: {r2_lin:.2f}')
st.write(f'k-NN - MSE: {mse_knn:.2f}, R2: {r2_knn:.2f}')

# Graficul predicțiilor
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(y_test, y_pred_lin, color='blue')
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[0].set_title('Regresie Liniară')
ax[0].set_xlabel('Valori reale')
ax[0].set_ylabel('Predicții')

ax[1].scatter(y_test, y_pred_knn, color='green')
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[1].set_title('k-NN')
ax[1].set_xlabel('Valori reale')
ax[1].set_ylabel('Predicții')

st.pyplot(fig)
