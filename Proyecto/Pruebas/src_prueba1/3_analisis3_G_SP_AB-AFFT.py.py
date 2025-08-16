import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

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

    # - ¿MOVIMIENTO RÍTMICO? -
    print("\n--- Análisis básico de ritmicidad ---")
    ritmico_basico = analizar_ritmicidad_basica(senal_principal, t)

    print("\n--- Análisis con FFT ---")
    ritmico_fft = analizar_ritmicidad_fft(senal_principal, t, nombre_senal, ruta_datos)

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


    # - ¿RÍTMICO? -

    # - Analisis básico (con picos) -
def analizar_ritmicidad_basica(senal_principal, t):
    # Detectar picos
    picos, _ = find_peaks(senal_principal, distance=10)  # Ajustarl parámetro si es muy sensible

    if len(picos) < 2:
        print("❌ No hay suficientes picos para analizar el ritmo.")
        return False

    # Calcular intervalos de tiempo entre picos
    tiempos_picos = t[picos]
    intervalos = np.diff(tiempos_picos)

    # Desviación estándar y media de intervalos
    std_intervalos = np.std(intervalos)
    media_intervalos = np.mean(intervalos)

    # Variabilidad de intervalos
    coef_var = std_intervalos / media_intervalos

    print(f"Coeficiente de variación: {coef_var:.3f}")

    if coef_var < 0.2:
        print("✅ Movimiento rítmico (básico).")
        return True
    else:
        print("❌ Movimiento no rítmico (básico).")
        return False


        # - Analisis FFT (Transformada de Fourier)  -
def analizar_ritmicidad_fft(senal_principal, t, nombre_senal, ruta_datos):
    N = len(senal_principal) # Número de datos
    T = t[1] - t[0]  # Paso temporal: intervalo entre dos puntos.

    # Aplicar FFT
    yf = fft(senal_principal) # Aplicar Transformada Rápida de Fourier → pasa de tiempo a frecuencia
    xf = fftfreq(N, T) # Obtener las frecuencias correspondientes a cada punto de yf

    # Considerar solo frecuencias positivas
    idx = xf > 0
    xf = xf[idx]
    yf = np.abs(yf[idx]) # abs() - obtener magnitud de número complejo

    # Detectar frecuencia dominante
    frecuencia_dominante = xf[np.argmax(yf)] # Buscar frecuencia con mayor amplitud (frecuencia principal)
    print(f"🎵 Frecuencia dominante: {frecuencia_dominante:.2f} Hz")

    # Graficar espectro
    plt.figure()
    plt.plot(xf, yf) # Espectro de frecuencias
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Amplitud")
    plt.title(f"FFT de {nombre_senal}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

        # - GUARDAR IMAGEN -
    ruta_espectroFrecuencias = os.path.join(ruta_datos, "grafica_espectroFrecuencias.png")
    plt.savefig(ruta_espectroFrecuencias)

    # Criterio simple: amplitud dominante debe ser clara
    if max(yf) > 5 * np.mean(yf): # Si pico principal es más alto que el promedio
        print("✅ Movimiento rítmico (FFT).")
        return True
    else:
        print("❌ Movimiento no rítmico (FFT).")
        return False


# Evitar que se ejecute al importar
if __name__ == "__main__":
    ejecutar_analisis()