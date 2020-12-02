import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.linear_model import LinearRegression,Lasso,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time





#Funcion con la que se realiza el kmean
def kmeans(datos):

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(datos)
    labels = [0, 1, 2, 3, 4]
    scatter=plt.scatter(datos['x'], datos['y'], c=kmeans.labels_, cmap='autumn')
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.title("K-Means")
    plt.show()

#Agglomerative Clustering(“Single Linkage”)
def agglomerative(datos):

    #k = 1,2,3,4,5
    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative.fit(datos)
    scatter=plt.scatter(datos['x'], datos['y'], c=agglomerative.labels_, cmap='rainbow')
    labels=[0,1,2,3,4]
    plt.legend(handles=scatter.legend_elements()[0],labels=labels)
    plt.title("Agglomerative")
    plt.show()

    # umbral de distancia = 0.25, 0.50, 0.75, 1.0 y 1.5
    agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    agglomerative.fit(datos)
    plt.scatter(datos['x'], datos['y'], c=agglomerative.labels_, cmap='rainbow')
    plt.legend()
    plt.title("Agglomerative")
    plt.show()

#DB Scan
def dbscan(datos):
    #valores eps=0.25,0.35 y 0.5; min_samples= 5,10 y 15
    # 0.25 con 10 , 0.25 con 15, 0.35 con 15,
    # 0.5 con 5 sale 2 clases
    # El resto salen muchas
    dbscan = DBSCAN(eps=0.35, min_samples=5)
    dbscan.fit(datos)
    scatter=plt.scatter(datos['x'], datos['y'], c=dbscan.labels_, cmap='rainbow')
    labels = [0, 1, 2, 3, 4,5,6,7,8,9]
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.title("DB Scan")
    plt.show()


#K-Nearest Neighbors Algorithm
def knn(datos,datosTest):
    total=0
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
        inicio=""
        fin=""
        inicio = time.time()
        knn.predict(x)
        fin = time.time()
        total=total+(fin - inicio) * 1000
        print("Tiempo total predicción sobre datos de prueba", (fin - inicio) * 1000, "milisegundos")

    print("Total:",total)

def Parte1():
    # Cargar Datos
    datos = pd.read_csv("datos_1.csv")
    datos1 = pd.read_csv("datos_2.csv")
    datos2 = pd.read_csv("datos_3.csv")
    datos3 = [datos2]

    # Parte 1
    for i in datos3:
        #kmeans(i)
        agglomerative(i)
        #dbscan(i)

def Parte2():
    # Datos originales
    datos = pd.read_csv("os_training_data.csv")
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

    #Datos Test
    datosTest = pd.read_csv("os_testing_data.csv")
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


    knn(datos,datosTest)

def Parte3(dataset1,dataset2):

    #Datos de Entrenamiento
    #Regresion Lineal: minimos cuadrados

    #datos originales
    datos = pd.read_csv(dataset1)
    datosTest = pd.read_csv(dataset2)

    x = datos.loc[:, datos.columns != 'score']
    y = datos['score']
    y_true = datosTest['score']
    linearRegression = LinearRegression()
    linearRegression.fit(x,y)
    y_pred = linearRegression.predict(datosTest.loc[:, datosTest.columns != 'score'])
    print("Regresion Lineal minimos cuadrados sin normalizar: ",metrics.mean_squared_error(y_true,y_pred))

    #datos normalizados
    # Datos
    scaler = StandardScaler()
    scaler.fit(datos)
    datos=scaler.transform(datos)

    scaler2 = StandardScaler()
    scaler2.fit(datosTest)
    datosTest=scaler2.transform(datosTest)

    x = datos[:,:5]
    y = datos[:, -1]
    y_true = datosTest[:,-1]
    linearRegression = LinearRegression()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest[:, :5])
    print("Regresion Lineal minimos cuadrados normalizado: ",metrics.mean_squared_error(y_true, y_pred))

    #Regresion Lineal: Lasso con regularizacion L1
    #datos originales
    datos = pd.read_csv(dataset1)
    datosTest = pd.read_csv(dataset2)
    x = datos.loc[:, datos.columns != 'score']
    y = datos['score']
    y_true = datosTest['score']
    linearRegression = Lasso()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest.loc[:, datosTest.columns != 'score'])
    print("Regresion Lineal Lasso L1 sin normalizar:",metrics.mean_squared_error(y_true, y_pred))

    #datos normalizados
    datos = pd.read_csv(dataset1)
    datosTest = pd.read_csv(dataset2)

    scaler = StandardScaler()
    scaler.fit(datos)
    datos = scaler.transform(datos)

    scaler2 = StandardScaler()
    scaler2.fit(datosTest)
    datosTest = scaler2.transform(datosTest)

    x = datos[:, :5]
    y = datos[:,-1]
    y_true = datosTest[:,-1]
    linearRegression = Lasso()
    linearRegression.fit(x, y)
    y_pred = linearRegression.predict(datosTest[:, :5])
    print("Regresion Lineal Lasso L1 normalizado:",metrics.mean_squared_error(y_true, y_pred))


def Parte4():
    datos = pd.read_csv("os_training_data.csv")
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
    print("Accuracy: ",metrics.accuracy_score(prediccion,y))

    #Tiempo total de prediccion
    datosTest= pd.read_csv("os_testing_data.csv")
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
    logisticregresion.fit(x,y)
    inicio = time.time()
    logisticregresion.predict(x)
    fin = time.time()
    print("Tiempo total predicción sobre datos de prueba",(fin-inicio)*1000,"milisegundos")





def main():

    #Parte1
    #Parte1()

    #Parte 2
    #Parte2()

    #Parte3
    #Parte3("datos_4_train.csv","datos_4_train.csv")

    #Parte4
    Parte4()



if __name__ == "__main__":
    main()
