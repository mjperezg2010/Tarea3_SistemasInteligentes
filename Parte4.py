import sys
import time
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():
    datos = pd.read_csv(sys.argv[1])
    attribute = list(datos)
    columns = len(attribute)
    rows = len(datos)

    # Guardar tabla
    for i in range(rows):
        for j in range(columns):
            if datos.iloc[i][j] == "Si":
                datos.iloc[i][j] = "1"
            elif datos.iloc[i][j] == "No":
                datos.iloc[i][j] = "0"
    logisticregresion = LogisticRegression()
    x = datos.loc[:, datos.columns != 'class']
    y = datos['class']
    logisticregresion.fit(x, y)
    prediccion = logisticregresion.predict(x)
    print("Reporte de clasificacion")
    print(classification_report(prediccion, y, labels=['windows']))
    print("Accuracy: ", metrics.accuracy_score(prediccion, y))

    # Tiempo total de prediccion
    datosTest = pd.read_csv(sys.argv[2])
    attribute = list(datosTest)
    columns = len(attribute)
    rows = len(datosTest)

    # Guardar tabla
    for i in range(rows):
        for j in range(columns):
            if datosTest.iloc[i][j] == "Si":
                datosTest.iloc[i][j] = "1"
            elif datosTest.iloc[i][j] == "No":
                datosTest.iloc[i][j] = "0"

    logisticregresion = LogisticRegression()
    x = datosTest.loc[:, datosTest.columns != 'class']
    y = datosTest['class']
    logisticregresion.fit(x, y)
    inicio = time.time()
    logisticregresion.predict(x)
    fin = time.time()
    print("Tiempo total predicción sobre datos de prueba", (fin - inicio) * 1000, "milisegundos")

if __name__ == '__main__':
    main()