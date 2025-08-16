import os
import cv2  # Importar OpenCV
import time
import numpy as np
import tkinter as tk
from video.display import show_letterboxed
from video.roi import detectar_ojos_e_iris, reset_roi_state

def seguimiento(ruta_video, lado_fijo=None):
        # - INICIALIZAR VÍDEO -
    video = cv2.VideoCapture(ruta_video) # Abrir video
    fps_video = video.get(cv2.CAP_PROP_FPS) # Obtener fps

        # - RUTA PARA GUARDAR DATOS -
    # Ruta absoluta de carpeta raíz del proyecto
    ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # ruta_base apunta a .../Proyecto
    ruta_datos = os.path.join(ruta_base, "datos") # Carpeta datos dentro de la raíz Proyecto
    os.makedirs(ruta_datos, exist_ok=True) # Asegurar que existe carpeta, si no crear

    trayectoria = []

    ok, frame = video.read() # Leer primer frame
    if not ok:
        print("No se pudo leer el video.")
        guardar_datos(trayectoria, ruta_datos)
        return

        # - OBTENER ROI IRIS -
    datos_ojo = detectar_ojos_e_iris(frame, lado_fijo=lado_fijo) # Intentar detectar en el primer frame
    intentos = 0
    while datos_ojo is None and intentos < 10: # Si falla, hay 10 intentos
        ok, frame = video.read()
        if not ok:
            print("No hay más frames para intentar detección.")
            guardar_datos(trayectoria, ruta_datos)
            return
        datos_ojo = detectar_ojos_e_iris(frame, lado_fijo=lado_fijo)
        intentos += 1

    if datos_ojo is None:
        print("No se pudo detectar el ojo/iris.")
        guardar_datos(trayectoria, ruta_datos)
        return
    
    bbox = datos_ojo["bbox_iris"]
    if lado_fijo is None:
        lado_fijo = datos_ojo["lado"]  # Guardar ojo si no vino de fuera

    # Validación rápida del bbox
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        print("BBox del iris inválido.")
        guardar_datos(trayectoria, ruta_datos)
        return

        # - TRACKER -
    # Inicializar tracker
    tracker = cv2.TrackerCSRT_create() # Crear objeto de seguimiento (tracker) usando algoritmo CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability).
        # 'TrackerCSRT' tipo de tracker preciso y robusto, ideal para objetos con movimiento lento. Útil para objetos que cambian de escala, rotación o iluminación.
        # 'create()' método que instancia (crea) el tracker.
    tracker.init(frame, bbox) # Inicializar tracker con el primer cuadro (frame) y la región seleccionada (bbox).
        # 'init' vincular tracker al objeto a seguir desde el cuadro inicial.

        # - PARA SEGUIMIENTO -
    # Guardar
    trayectoria = [] # Lista de trayectoria
    t0 = time.time()  # Tiempo inicial
    cx_prev, cy_prev = None, None  # Posición anterior

        # - GUARDAR VÍDEO -
    # Configurar vídeo de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
    ruta_video = os.path.join(ruta_datos, "video_seguimiento.mp4")
    video_salida = cv2.VideoWriter(ruta_video, fourcc, fps_video, (frame.shape[1], frame.shape[0]))

        # - SEGUIMIENTO -
    while True:
            # - INICIO -
        # Leer nuevo frame
        ok, frame = video.read()
        if not ok:
            break

            # - SEGUIMIENTO -
        # Actualizar posición del objeto en nuevo cuadro (frame).
        ok, bbox = tracker.update(frame)
            # 'update' método que analiza el nuevo frame para encontrar el objeto.
                # 'ok' booleano (True o False) indica si el objeto fue encontrado exitosamente.
                # 'bbox' tupla (x, y, w, h) con nueva posición y tamaño del objeto en el frame actual.

        cx = cy = None
        
        if ok: # si el objeto fue encontrado (true).
            # Convertir coordenadas a enteros
            x, y, w, h = [int(v) for v in bbox]
                # Extrae valores 'bbox', convetirr a enteros.
                # 'bbox' tiene la forma (x, y, w, h) → posición y tamaño del rectángulo.
                # 'int()' float a enteros. (recorrer cada valor 'v' en 'bbox')
            # Calcula coordenadas del centro del rectángulo.
            cx = x + w // 2  # centro x (Sumar mitad del alto (h) a la coordenada y)
            cy = y + h // 2  # centro y (Sumar mitad del ancho (w) a la coordenada x)

            # Dibujar recuadro y centro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # rectangle(frame, esquina superior izquierda, esquina inferior derecha, verde, thickness)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                # circle(frame, centro, radio, azul, relleno (-1))
            cv2.putText(frame, f"Posicion: ({cx}, {cy})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tiempo transcurrido
            tiempo = time.time() - t0 


            if ok:
                print(f"[DEBUG] Punto guardado: t={tiempo:.2f}, cx={cx}, cy={cy}")
            else:
                print("[DEBUG] Tracker perdió el objeto")

            
            # Guardar tiempo y posición
            trayectoria.append((tiempo, cx, cy)) # 'append()' agregar al final de la lista.
        else: # si el objeto no fue encontrado (false).
            cv2.putText(frame, "Ojo perdido, reintentando búsqueda...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            datos_ojo = detectar_ojos_e_iris(frame, lado_fijo=lado_fijo)
            if datos_ojo is not None:
                bbox = datos_ojo["bbox_iris"]
                x, y, w, h = [int(v) for v in bbox]
                if w > 0 and h > 0:
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    cv2.putText(frame, "Ojo recuperado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # - ESTADO MOVIMIENTO -
        # Verificar movimiento
        if cx is not None and cx_prev is not None:
            delta = ((cx - cx_prev)**2 + (cy - cy_prev)**2)**0.5 # Distancia entre posición actual y anterior.
            if delta > 1:  # Umbral de movimiento
                texto_estado = "Estado: Alerta, movimiento detectado!"
                color = (0, 255, 0)
            else:
                texto_estado = "Estado: No se ha detectado movimiento"
                color = (0, 0, 255)
        else: # Si no hay coordenadas previas (None).
            texto_estado = "Estado: No se ha detectado movimiento"
            color = (0, 0, 255)

        # Actualizar posición anterior
        cx_prev, cy_prev = cx, cy

        # Mostrar estado en pantalla
        cv2.putText(frame, texto_estado, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # - GUARDAR VÍDEO -
        video_salida.write(frame)

        # Mostrar video en una ventana
        show_letterboxed(frame, win="Seguimiento de iris")

            # - SALIR -
        # Salir al presionar 'Esc'
        if cv2.waitKey(int(1000 / fps_video)) & 0xFF == 27:
            # 'waitKey()' esperar por una entrada del teclado → 1000[ms]/FPS = 1000/20 = 50.
            # '27' Esc en código ASCII.
            print("ESC presionado, guardando datos y saliendo...")
            break

            # - CERRAR TODO -
    video.release() # Liberar cámara o video.
    video_salida.release() # Liberar video de salida (guardado).
    cv2.destroyAllWindows()
    reset_roi_state()

    # - GUARDAR SEGUIMIENTO -
        # Tiempo y coordenadas
    guardar_datos(trayectoria, ruta_datos)

    print("Seguimiento ejecutado. Datos guardados.")


# - GUARDAR SEGUIMIENTO -
# Tiempo y coordenadas
def guardar_datos(trayectoria, ruta_datos):
    ruta_csv = os.path.join(ruta_datos, "datos_seguimiento.csv")
    with open(ruta_csv, "w") as archivo: # Abrir o crea archivo en modo escritura 'w'. (Si existe, sobrescribir)
        archivo.write("tiempo,x,y\n")  # Encabezado
        for t, cx, cy in trayectoria: # Recorrer cada par de coordenadas (cx, cy) en 'trayectoria'.
            archivo.write(f"{t:.3f},{cx},{cy}\n")


# Solo ejecuta si este archivo se corre directamente (bloque de prueba individual)
if __name__ == "__main__":
    seguimiento()
