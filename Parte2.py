import sys
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import time
import pandas as pd

#K-Nearest Neighbors Algorithm
def knn(datos,datosTest):
    k = [1, 3, 5, 7, 9, 11, 13, 15]
    for i in k:
        knn = KNeighborsClassifier(n_neighbors=i)
        x = datos.loc[:,datos.columns!='class']
        y = datos['class']
        knn.fit(x,y)
        predicted= knn.predict(x)
        print("Reporte de clasificacion k=",i)
        print(classification_report(predicted, y, labels=['windows']))
        print("Accuracy: ", metrics.accuracy_score(predicted, y))

        #Tiempo total de predicción sobre datos de prueba
        x = datosTest.loc[:, datos.columns != 'class']
        y = datosTest['class']
        knn.fit(x, y)
        inicio = time.time()
        knn.predict(x)
        fin = time.time()
        print("Tiempo total predicción sobre datos de prueba", (fin - inicio) * 1000, "milisegundos")


def main():
    # Datos originales
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

    # Datos Test
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

    knn(datos, datosTest)

if __name__ == "__main__":
    main()