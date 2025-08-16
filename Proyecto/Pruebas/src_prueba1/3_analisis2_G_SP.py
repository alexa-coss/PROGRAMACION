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
    ruta_graficas = os.path.join(ruta_datos, "graficas_xyd.png")
    plt.savefig(ruta_graficas)

        # - RENDER -
    plt.show()

        # - SELECCIONAR SEÑAL -
    senal_principal, nombre_senal = seleccionar_senal_principal(x, y, distancia)

        # - GRÁFICA SEÑAL SELECCIONADA -
    plt.figure()
    plt.plot(t, senal_principal)
    plt.xlabel("Tiempo [s]")
    plt.ylabel(nombre_senal)
    plt.title(f"Señal principal seleccionada: {nombre_senal} vs tiempo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        # - GUARDAR IMAGEN -
    ruta_senalPrincipal = os.path.join(ruta_datos, "grafica_senalPrincipal.png")
    plt.savefig(ruta_senalPrincipal)

    print("Analisis ejecutado. Datos guardados.")


    # - SELECCIONAR SEÑAL -
def seleccionar_senal_principal(x, y, distancia):
    # Rango de movimiento (variación) en cada eje
    rango_x = np.max(x) - np.min(x)
    rango_y = np.max(y) - np.min(y)

    # Si un eje domina, seleccionarlo
    if rango_x > 1.5 * rango_y:
        print("📈 Se seleccionó el eje X como señal principal.")
        return x, "x"
    elif rango_y > 1.5 * rango_x:
        print("📈 Se seleccionó el eje Y como señal principal.")
        return y, "y"
    else:
        print("📊 Movimiento en ambos ejes → se usará la distancia al punto inicial.")
        return distancia, "distancia"


# Evitar que se ejecute al importar
if __name__ == "__main__":
    ejecutar_analisis()