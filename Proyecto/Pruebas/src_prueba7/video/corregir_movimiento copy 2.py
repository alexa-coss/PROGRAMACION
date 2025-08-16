import cv2
import numpy as np
import os
import tkinter as tk
from video.display import show_letterboxed
from video.roi import detectar_ojos_e_iris

def corregir_movimiento(ruta_video, mostrar_roi_en_pantalla=True):
    # Corrige el movimiento del vídeo usando seguimiento de un recuadro (ROI) con meanShift.

    cap = cv2.VideoCapture(ruta_video) # Abrir vídeo desde ruta
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener cuadros por segundo
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del vídeo en píxeles (convertir a entero)
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Alto del vídeo en píxeles (convertir a entero)

    ok, prev_frame = cap.read() # Leer primer frame del vídeo
    if not ok:
        print("No se pudo leer el vídeo.")
        return None

    # 1. Obtener ROI automático de ojo
    # Buscar ojo en varios frames (máx. 30 intentos)
    # 1. 🔥 Seleccionar recuadro inicial ROI AUTOMÁTICO (x, y, w, h)
    #    Buscamos ojo en los primeros frames y usamos su bbox como ROI, igual que si lo hubieras marcado a mano.
    roi = None
    datos_ojo = detectar_ojos_e_iris(prev_frame)
    intentos = 0
    while datos_ojo is None and intentos < 60:   # 🔥 hasta ~2–3 s si el video es 20–30 fps
        ok, frm = cap.read()
        if not ok:
            break
        datos_ojo = detectar_ojos_e_iris(frm)
        if datos_ojo is not None:
            prev_frame = frm  # 🔥 si detectamos más adelante, actualizamos frame base
            break
        intentos += 1

    if datos_ojo is None:
        print("No se pudo detectar el ojo en los primeros 200 frames.")
        return None

    # 🔥 ROI base = ojo completo detectado
    x, y, w, h = datos_ojo["bbox_ojo"]

    # 🔥 Recortar un poco para evitar ceja/piel (mismo espíritu que cuando lo dibujabas tú)
    recorte_margen = 0.12  # 12% por lado; ajusta 0.08–0.18 si hace falta
    x += int(w * recorte_margen)
    y += int(h * recorte_margen)
    w = max(10, int(w * (1 - 2*recorte_margen)))
    h = max(10, int(h * (1 - 2*recorte_margen)))
    # Limitar a bordes
    x = max(0, min(x, ancho-1)); y = max(0, min(y, alto-1))
    w = min(w, ancho - x);       h = min(h, alto - y)
    roi = (x, y, w, h)  # 🔥 ROI automático final (equivalente a tu selección manual)

    # 2. Preparar histograma HSV para seguimiento con meanShift (igual que antes)
    hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    mask     = cv2.inRange(hsv_prev, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hsv  = hsv_prev[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_roi = mask    [roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    roi_hist = cv2.calcHist([roi_hsv], [0], mask_roi, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Preparar vídeo salida
    ruta_salida = os.path.splitext(ruta_video)[0] + "_estabilizado.mp4" # Nombre vídeo corregido
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Códec para vídeo mp4
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto)) # Crear archivo vídeo salida

    out.write(prev_frame)  # Escribir primer frame sin corregir

    rect = roi  # Variable para almacenar posición actual del ROI

    while True:  # Leer cada frame
        ok, curr_frame = cap.read()  # Leer frame actual
        if not ok:  # Si no hay más frames
            break

        # 3. Convertir frame actual a HSV y calcular backprojection del histograma ROI
        hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 4. 🔥 Usar rect directamente
        ret, new_rect = cv2.meanShift(back_proj, rect,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

        # 5. 🔥 Calcular desplazamiento usando el rect anterior
        dx = new_rect[0] - rect[0]
        dy = new_rect[1] - rect[1]

        # 6. 🔥 Corregir frame desplazando en dirección opuesta
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        frame_corregido = cv2.warpAffine(curr_frame, M, (ancho, alto))

        # 🔥 Actualizar rect para el siguiente ciclo
        rect = new_rect # Actualizar posición del ROI para el siguiente frame

        # 🔥 Dibujar recuadro y centro solo en la VENTANA (no se guarda en el archivo)
        if mostrar_roi_en_pantalla:
            frame_mostrar = frame_corregido.copy()
            x, y, w, h = rect
            cv2.rectangle(frame_mostrar, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.circle(frame_mostrar, (cx, cy), 4, (255, 0, 0), -1)
            show_letterboxed(frame_mostrar, win="Video estabilizado")
        else:
            show_letterboxed(frame_corregido, win="Video estabilizado")

        if cv2.waitKey(1) & 0xFF == 27:
            print("Salida anticipada por ESC")
            break

        out.write(frame_corregido) # Guardar frame corregido en archivo de video de salida
        prev_frame = frame_corregido # Actualizar prev_frame para comparar con el siguiente frame en la próxima iteración

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Estabilización ejecutada. Datos guardados.")

    return ruta_salida
