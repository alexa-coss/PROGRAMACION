import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

def analisis():

        # - RUTA PARA GUARDAR DATOS -
    # Ruta absoluta de carpeta raíz del proyecto
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ruta_base apunta a .../Proyecto
    ruta_datos = os.path.join(ruta_base, "datos") # Carpeta datos dentro de la raíz Proyecto
    ruta_csv_legacy = os.path.join(ruta_datos, 'datos_seguimiento.csv') # Determinar CSV legado
    ruta_csv_L = os.path.join(ruta_datos, 'datos_seguimiento_L.csv') # Determinar CSV ojo izquierdo
    ruta_csv_R = os.path.join(ruta_datos, 'datos_seguimiento_R.csv') # Determinar CSV ojo derecho

        # - OBTENER DATOS -
    datasets = {}  # Usar contenedor de series disponibles
    if os.path.exists(ruta_csv_L):  # Validar existencia L
        datasets["L"] = pd.read_csv(ruta_csv_L)
    if os.path.exists(ruta_csv_R):  # Validar existencia R
        datasets["R"] = pd.read_csv(ruta_csv_R)
    if not datasets and os.path.exists(ruta_csv_legacy):  # Delegar al CSV legado si no hay L/R
        datasets["LEGACY"] = pd.read_csv(ruta_csv_legacy)

    if not datasets:  # Validar que exista al menos un dataset
        print("No se encontraron archivos de datos de seguimiento")
        return

        # - ANALIZAR SERIES DISPONIBLES -
    for etiqueta, df in datasets.items():  # Iterar por cada serie disponible
        # Extraer columnas
        t = df["tiempo"].to_numpy() # Convertir columna a array NumPy
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()

            # - DISTANCIA VS TIEMPO -
        # Punto inicial
        x0, y0 = x[0], y[0]
        # Calcular distancia al primer punto
        distancia = ((x - x0)**2 + (y - y0)**2)**0.5

            # - GRÁFICAS -
        plt.figure(figsize=(9, 7))  # Determinar tamaño de figura
        plt.subplot(3, 1, 1)
        plt.plot(t, x, label="x(t)", color='blue')
        plt.xlabel("Tiempo [s]")
        plt.ylabel("x")
        plt.title(f"Distancia (x) vs tiempo - {etiqueta}")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(t, y, label="y(t)", color='red')
        plt.xlabel("Tiempo [s]")
        plt.ylabel("y")
        plt.title(f"Distancia (y) vs tiempo - {etiqueta}")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(t, distancia, label="distancia(t)", color='green')
        plt.xlabel("Tiempo [s]")
        plt.ylabel("Distancia (d)")
        plt.title(f"Distancia total al punto inicial vs tiempo - {etiqueta}")
        plt.grid(True)

            # - AJUSTE -
        plt.tight_layout() # Ajustar automáticamente espacios entre gráficas

            # - GUARDAR IMAGEN -
        ruta_graficas = os.path.join(ruta_datos, f"graficas_xyd_{etiqueta}.png") # Guardar por ojo
        plt.savefig(ruta_graficas)
        plt.close()  # Liberar figura para memoria

            # - SELECCIONAR SEÑAL -
        senal_principal, nombre_senal = seleccionar_senal_principal(x, y, distancia)  # Determinar señal principal

            # - GRÁFICA SEÑAL SELECCIONADA -
        plt.figure()
        plt.plot(t, senal_principal)
        plt.xlabel("Tiempo [s]")
        plt.ylabel(nombre_senal)
        plt.title(f"Señal principal seleccionada: {nombre_senal} vs tiempo - {etiqueta}")
        plt.grid(True)
        plt.tight_layout()

            # - GUARDAR IMAGEN -
        ruta_senalPrincipal = os.path.join(ruta_datos, f"grafica_senalPrincipal_{etiqueta}.png") # Guardar por ojo
        plt.savefig(ruta_senalPrincipal)
        plt.close()

            # - ¿MOVIMIENTO RÍTMICO? -
        print(f"\n--- Análisis básico de ritmicidad ({etiqueta}) ---")
        ritmico_basico = analizar_ritmicidad_basica(senal_principal, t, ruta_datos, etiqueta)  # Delegar análisis

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
def analizar_ritmicidad_basica(senal_principal, t, ruta_datos, etiqueta):
    # Detectar picos
    picos, _ = find_peaks(senal_principal, distance=10, prominence=0.5) # Ajustar parámetros si es sensible

    # Visualizar señal y picos detectados
    plt.figure()  # Crear figura para picos
    plt.plot(t, senal_principal)
    if len(picos) > 0:  # Validar picos encontrados
        plt.plot(t[picos], senal_principal[picos], "rx")  # Marcar picos
    plt.title(f"Señal y picos detectados - {etiqueta}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Valor de la señal")
    plt.tight_layout()
    ruta_picos = os.path.join(ruta_datos, f"senal_y_picos_{etiqueta}.png") # Guardar figura de picos
    plt.savefig(ruta_picos)
    # plt.show()  # Usar si se requiere ver la figura en pantalla
    plt.close()

    if len(picos) < 5:
        print("❌ No hay suficientes picos para analizar el ritmo.")
        return False

    # Calcular intervalos de tiempo entre picos y amplitud
    tiempos_picos = t[picos]
    intervalos = np.diff(tiempos_picos)
    alturas = senal_principal[picos]

        # Intervalos
    # Desviación estándar y media
    std_intervalos = np.std(intervalos)
    media_intervalos = np.mean(intervalos)
    # Variabilidad de intervalos
    cv_int = std_intervalos / media_intervalos if media_intervalos != 0 else np.inf

        # Amplitud
    # Desviación estándar y media de alturas
    std_alturas = np.std(alturas)
    media_alturas = np.mean(alturas)
    # Variabilidad de alturas
    cv_amp = std_alturas / media_alturas if media_alturas != 0 else np.inf

    print(f"CV intervalos: {cv_int:.3f} | CV amplitud: {cv_amp:.3f}")

    if cv_int < 0.1 and cv_amp < 0.3:
        print("✅ Movimiento rítmico (básico).")
        return True
    else:
        print("❌ Movimiento no rítmico (básico).")
        return False


# Solo ejecuta si este archivo se corre directamente (bloque de prueba individual)
if __name__ == "__main__":
    analisis()
