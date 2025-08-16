import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ejecutar_analisis():

        # - RUTA PARA GUARDAR DATOS -
    # Ruta absoluta de carpeta raíz del proyecto
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ruta_base apunta a .../Proyecto
    ruta_datos = os.path.join(ruta_base, "datos") # Carpeta datos dentro de la raíz Proyecto
    ruta_csv = os.path.join(ruta_datos, 'datos_seguimiento.csv')

        # - OBTENER DATOS -
    df = pd.read_csv(ruta_csv) # df = DataFrame

    # Extraer columnas
    t = df["tiempo"].to_numpy() # Convier columna a array NumPy.
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

        # - DISTANCIA VS TIEMPO -
    # Punto inicial
    x0, y0 = x[0], y[0]
    # Calcular distancia al primer punto
    distancia = ((x - x0)**2 + (y - y0)**2)**0.5

        # - GRÁFICAS -
    plt.subplot(3, 1, 1)
    plt.plot(t, x, label="x(t)", color='blue')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("x")
    plt.title("Distancia (v) vs tiempo")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, y, label="y(t)", color='red')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("y")
    plt.title("Distancia (y) vs tiempo")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, distancia, label="distancia(t)", color='green')
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Distancia (d)")
    plt.title("Distancia total al punto inicial vs tiempo")
    plt.grid(True)

        # - AJUSTE -
    plt.tight_layout() # Ajustar automáticamente espacios entre gráficas (no se sobrepongan o queden juntas)

        # - GUARDAR IMAGEN -
    ruta_imagen = os.path.join(ruta_datos, "grafica_seguimiento.png")
    plt.savefig(ruta_imagen)

        # - RENDER -
    plt.show()

    print("Analisis ejecutado. Datos guardados.")


# Evitar que se ejecute al importar
if __name__ == "__main__":
    ejecutar_analisis()