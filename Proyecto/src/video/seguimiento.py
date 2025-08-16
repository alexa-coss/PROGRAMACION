import os
import cv2  # Importar OpenCV
import time
import numpy as np
import tkinter as tk
from video.display import show_letterboxed
from video.roi import detectar_ojos_e_iris_multi, reset_roi_state  # Usar detección múltiple de ojos

def seguimiento(ruta_video, modo_ojos="auto", lado_fijo=None):
        # - INICIALIZAR VÍDEO -
    video = cv2.VideoCapture(ruta_video) # Abrir video
    fps_video = video.get(cv2.CAP_PROP_FPS) # Obtener fps

        # - RUTA PARA GUARDAR DATOS -
    ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Determinar ruta base del proyecto
    ruta_datos = os.path.join(ruta_base, "datos") # Determinar carpeta de datos
    os.makedirs(ruta_datos, exist_ok=True) # Crear carpeta de datos si no existe

    trayectoria_L = []  # Lista de trayectoria ojo izquierdo
    trayectoria_R = []  # Lista de trayectoria ojo derecho

    ok, frame = video.read() # Leer primer frame
    if not ok:
        print("No se pudo leer el video.")
        guardar_datos_multi(trayectoria_L, trayectoria_R, ruta_datos) # Guardar vacío
        return

        # - OBTENER ROI IRIS -
    datos_multi = detectar_ojos_e_iris_multi(frame, lado_fijo=lado_fijo) # Detectar ambos ojos
    intentos = 0
    while (not datos_multi["left"] and not datos_multi["right"]) and intentos < 10: # Reintentar si no detecta
        ok, frame = video.read()
        if not ok:
            print("No hay más frames para intentar detección.")
            guardar_datos_multi(trayectoria_L, trayectoria_R, ruta_datos)
            return
        datos_multi = detectar_ojos_e_iris_multi(frame, lado_fijo=lado_fijo)
        intentos += 1

    if not datos_multi["left"] and not datos_multi["right"]:
        print("No se pudo detectar ningún ojo/iris.")
        guardar_datos_multi(trayectoria_L, trayectoria_R, ruta_datos)
        return
    
    # Crear trackers según el modo solicitado
    tracker_L, tracker_R = None, None
    if modo_ojos in ("auto", "both"):
        if datos_multi["left"]:
            bbox_L = datos_multi["left"]["iris_bbox"]
            tracker_L = cv2.TrackerCSRT_create() # Crear tracker para ojo izquierdo
            tracker_L.init(frame, bbox_L) # Inicializar tracker con ROI izquierdo
        if datos_multi["right"]:
            bbox_R = datos_multi["right"]["iris_bbox"]
            tracker_R = cv2.TrackerCSRT_create() # Crear tracker para ojo derecho
            tracker_R.init(frame, bbox_R) # Inicializar tracker con ROI derecho
    elif modo_ojos == "mono":
        if datos_multi["left"]:
            bbox_L = datos_multi["left"]["iris_bbox"]
            tracker_L = cv2.TrackerCSRT_create()
            tracker_L.init(frame, bbox_L)
        elif datos_multi["right"]:
            bbox_R = datos_multi["right"]["iris_bbox"]
            tracker_R = cv2.TrackerCSRT_create()
            tracker_R.init(frame, bbox_R)

    t0 = time.time()  # Tiempo inicial
    cx_prev_L, cy_prev_L = None, None  # Posición anterior ojo izquierdo
    cx_prev_R, cy_prev_R = None, None  # Posición anterior ojo derecho

        # - GUARDAR VÍDEO -
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
    ruta_video_out = os.path.join(ruta_datos, "video_seguimiento.mp4") # Ruta salida video
    video_salida = cv2.VideoWriter(ruta_video_out, fourcc, fps_video, (frame.shape[1], frame.shape[0]))

        # - SEGUIMIENTO -
    while True:
            # - INICIO -
        ok, frame = video.read()
        if not ok:
            break

        # Actualizar ojo izquierdo si existe tracker
        cx_L = cy_L = None
        if tracker_L:
            ok_L, bbox_L = tracker_L.update(frame) # Actualizar tracker L
            if ok_L:
                x, y, w, h = [int(v) for v in bbox_L]
                cx_L, cy_L = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Dibujar ROI
                cv2.circle(frame, (cx_L, cy_L), 4, (255, 0, 0), -1) # Dibujar centro
                tiempo = time.time() - t0 # Calcular tiempo
                trayectoria_L.append((tiempo, cx_L, cy_L)) # Guardar trayectoria L

        # Actualizar ojo derecho si existe tracker
        cx_R = cy_R = None
        if tracker_R:
            ok_R, bbox_R = tracker_R.update(frame) # Actualizar tracker R
            if ok_R:
                x, y, w, h = [int(v) for v in bbox_R]
                cx_R, cy_R = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # Dibujar ROI
                cv2.circle(frame, (cx_R, cy_R), 4, (0, 0, 255), -1) # Dibujar centro
                tiempo = time.time() - t0 # Calcular tiempo
                trayectoria_R.append((tiempo, cx_R, cy_R)) # Guardar trayectoria R

        # Mostrar video en ventana
        show_letterboxed(frame, win="Seguimiento de iris") # Mostrar frame

        # Guardar frame en salida
        video_salida.write(frame)

        # Salir con ESC
        if cv2.waitKey(int(1000 / fps_video)) & 0xFF == 27:
            print("ESC presionado, guardando datos y saliendo...")
            break

        # Actualizar posiciones previas
        cx_prev_L, cy_prev_L = cx_L, cy_L
        cx_prev_R, cy_prev_R = cx_R, cy_R

    video.release()
    video_salida.release()
    cv2.destroyAllWindows()
    reset_roi_state()

    # Guardar trayectorias en CSV
    guardar_datos_multi(trayectoria_L, trayectoria_R, ruta_datos)

    print("Seguimiento ejecutado. Datos guardados.")

# Guardar datos de seguimiento múltiple
def guardar_datos_multi(tray_L, tray_R, ruta_datos):
    ruta_csv_L = os.path.join(ruta_datos, "datos_seguimiento_L.csv") # Ruta CSV L
    ruta_csv_R = os.path.join(ruta_datos, "datos_seguimiento_R.csv") # Ruta CSV R
    with open(ruta_csv_L, "w") as archivo:
        archivo.write("tiempo,x,y\n")
        for t, cx, cy in tray_L:
            archivo.write(f"{t:.3f},{cx},{cy}\n")
    with open(ruta_csv_R, "w") as archivo:
        archivo.write("tiempo,x,y\n")
        for t, cx, cy in tray_R:
            archivo.write(f"{t:.3f},{cx},{cy}\n")

if __name__ == "__main__":
    seguimiento("video.mp4", modo_ojos="both") # Ejemplo de prueba
