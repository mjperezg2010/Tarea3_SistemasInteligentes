import sys
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():
    #Cargar Datasets
    dataset1 = sys.argv[1]
    dataset2 = sys.argv[2]

    # Datos de Entrenamiento

    # Regresion Lineal: minimos cuadrados
    # datos originales
    datos = pd.read_csv(dataset1)
    datosTest = pd.read_csv(dataset2)

    x = datos.loc[:, datos.columns != 'score']
    y = datos['score']
    y_true = datosTest['score']
    linearRegression = LinearRegression()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest.loc[:, datosTest.columns != 'score'])
    print("Regresion Lineal minimos cuadrados sin normalizar: ", metrics.mean_squared_error(y_true, y_pred))

    # datos normalizados
    scaler = StandardScaler()
    scaler.fit(datos)
    datos = scaler.transform(datos)

    scaler2 = StandardScaler()
    scaler2.fit(datosTest)
    datosTest = scaler2.transform(datosTest)

    x = datos[:, :5]
    y = datos[:, -1]
    y_true = datosTest[:, -1]
    linearRegression = LinearRegression()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest[:, :5])
    print("Regresion Lineal minimos cuadrados normalizado: ", metrics.mean_squared_error(y_true, y_pred))

    # Regresion Lineal: Lasso con regularizacion L1
    # datos originales
    datos = pd.read_csv(dataset1)
    datosTest = pd.read_csv(dataset2)
    x = datos.loc[:, datos.columns != 'score']
    y = datos['score']
    y_true = datosTest['score']
    linearRegression = Lasso()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest.loc[:, datosTest.columns != 'score'])
    print("Regresion Lineal Lasso L1 sin normalizar:", metrics.mean_squared_error(y_true, y_pred))

    # datos normalizados
    datos = pd.read_csv(dataset1)
    datosTest = pd.read_csv(dataset2)

    scaler = StandardScaler()
    scaler.fit(datos)
    datos = scaler.transform(datos)

    scaler2 = StandardScaler()
    scaler2.fit(datosTest)
    datosTest = scaler2.transform(datosTest)

    x = datos[:, :5]
    y = datos[:, -1]
    y_true = datosTest[:, -1]
    linearRegression = Lasso()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest[:, :5])
    print("Regresion Lineal Lasso L1 normalizado:", metrics.mean_squared_error(y_true, y_pred))

if __name__ == '__main__':
    main()