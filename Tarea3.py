import pandas as pd





def main():

    datos = pd.read_csv("datos_1.csv")
    attribute = list(datos)
    tabla = []
    columns = len(attribute)
    rows = len(datos)

    # Guardar tabla
    for i in range(rows):
        temp = []
        for j in range(columns):
            temp.append(datos.iloc[i][j])
            tabla.append(temp)