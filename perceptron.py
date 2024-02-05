import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import perceptron as pl
import os

numeroEpocas = 0
pesosFinales = None
evolucionPeso = []
errorPermisible = 0
errorEpocas = []
pesosIniciales = None


def obtener_pesos():
    return pesosIniciales, pesosFinales, numeroEpocas, errorPermisible 

def trainPerceptron(tasaDeAprendizaje, epocas, rutaArchivo):
    #variables globales
    global errorEpocas, evolucionPeso, pesosIniciales,numeroEpocas, pesosFinales
    evolucionPeso.clear()
    errorEpocas.clear()

    frameData = pd.read_csv(rutaArchivo, delimiter = ';', header=None)
    numeroCaracteristicas = len(frameData.columns) - 1

    pesos = np.random.uniform(low=0, high=1, size=(numeroCaracteristicas + 1, 1)).round(4)
    columnX = np.hstack([frameData.iloc[:, :-1].values, np.ones((frameData.shape[0], 1))])
    columnY = np.array(frameData.iloc[:, -1])
    pesosIniciales = pesos.copy()
    numeroEpocas = epocas

    print(f"Pesos iniciales {pesos}")
    np.set_printoptions(precision=4, suppress=True)
    pesosIniciales, pesosFinales, epocas, error = pl.obtener_pesos()
    print(f"Épocas: {epocas}")
    print(f"Error permisible: {error}")
    print(f"Pesos iniciales: {pesosIniciales}")
    print(f"Pesos finales: {pesosFinales}")

    for i in range(numeroCaracteristicas + 1):
        evolucionPeso.append([])

    for epoch in range(epocas):
        u = np.dot(columnX, pesos)
        Yc = np.where(u >= 0, 1, 0).reshape(-1, 1)
        errors = columnY.reshape(-1, 1) - Yc
        errorNorma = np.linalg.norm(errors)
        errorEpocas.append(errorNorma)

        for i in range(numeroCaracteristicas + 1):
            evolucionPeso[i].append(pesos[i, 0])

        productErrors = np.dot(columnX.T, errors)
        deltaW = tasaDeAprendizaje * productErrors
        pesos += np.round(deltaW, 4)
        
    pesosFinales = pesos

def showResults():
    plt.figure(figsize=(12, 4)) 
    plt.plot(range(1, len(errorEpocas) + 1), errorEpocas)
    plt.title('Evolución de norma del error')
    plt.xlabel('Época')
    plt.ylabel('Norma del error')

    folder_path_error = "GraficasError"
    os.makedirs(folder_path_error, exist_ok=True)

    plt.savefig(os.path.join(folder_path_error, "graficaError.png"))
    plt.close()

    plt.figure(figsize=(12, 4))
    for i, pesos_epoca in enumerate(evolucionPeso):
        plt.plot(range(1, len(pesos_epoca) + 1), pesos_epoca, label=f'Peso {i + 1}')

    plt.title('Evolución de los pesos')
    plt.xlabel('Época')
    plt.ylabel('Valor del Peso')
    plt.legend()

    folder_path_pesos = "GraficasPesos"
    os.makedirs(folder_path_pesos, exist_ok=True)

    plt.savefig(os.path.join(folder_path_pesos, "graficaPesos.png"))
    plt.close()