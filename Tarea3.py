import pandas as pd
#import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


#Funcion con la que se realiza el kmean
def kmeans(datos):

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(datos)
    plt.scatter(datos['x'], datos['y'], c=kmeans.labels_, cmap='rainbow')
    plt.show()

#Agglomerative Clustering(“Single Linkage”)
def agglomerative(datos):

    #k = 1,2,3,4,5
    agglomerative = AgglomerativeClustering(n_clusters=5)
    agglomerative.fit(datos)
    scatter=plt.scatter(datos['x'], datos['y'], c=agglomerative.labels_, cmap='rainbow')
    print(len(np.unique(agglomerative.labels_)))

    labels=[0,1,2,3,4]
    plt.legend(handles=scatter.legend_elements()[0],labels=labels)
    plt.show()

    # umbral de distancia = 0.25, 0.50, 0.75, 1.0 y 1.5
    agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=0.25)
    agglomerative.fit(datos)
    plt.scatter(datos['x'], datos['y'], c=agglomerative.labels_, cmap='rainbow')
    plt.legend()
    plt.show()

#DB Scan
def dbscan(datos):
    #valores eps=0.25,0.35 y 0.5; min_samples= 5,10 y 15
    # 0.25 con 10 , 0.25 con 15, 0.35 con 15,
    # 0.5 con 5 sale 2 clases
    # El resto salen muchas
    dbscan = DBSCAN(eps=0.5, min_samples=15)
    dbscan.fit(datos)
    scatter=plt.scatter(datos['x'], datos['y'], c=dbscan.labels_, cmap='rainbow')
    labels = [0, 1, 2, 3, 4,5,6,7,8]
    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.show()


def main():
    #Cargar Datos
    datos = pd.read_csv("datos_3.csv")
    print(datos.head())
    #kmeans(datos)
    #agglomerative(datos)
    dbscan(datos)


if __name__ == "__main__":
    main()
