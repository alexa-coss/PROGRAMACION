import cv2
import numpy as np
import os
import tkinter as tk
from video.display import show_letterboxed
from video.roi import detectar_ojos_e_iris

def corregir_movimiento(ruta_video):
    # Corrige el movimiento del vídeo usando seguimiento de un recuadro (ROI) con meanShift.

    cap = cv2.VideoCapture(ruta_video) # Abrir vídeo desde ruta
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener cuadros por segundo
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del vídeo en píxeles (convertir a entero)
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Alto del vídeo en píxeles (convertir a entero)


    print("[DEBUG] isOpened?:", cap.isOpened())
    ok, frame = cap.read()
    print("[DEBUG] Primer frame leído?:", ok)
    if ok:
        print("[DEBUG] Tamaño del frame:", frame.shape)


    ok, prev_frame = cap.read() # Leer primer frame del vídeo
    if not ok:
        print("No se pudo leer el vídeo.")
        return None

    # 1. Obtener ROI automático de ojo
    # Buscar ojo en varios frames (máx. 30 intentos)
    datos_ojo = None
    for i in range(200):  # intenta con los primeros 30 frames
        ok, frame = cap.read()
        if not ok:
            break
        datos_ojo = detectar_ojos_e_iris(frame)
        if datos_ojo is not None:
            print(f"[DEBUG] Ojo detectado en frame {i+1}")
            prev_frame = frame  # guardar este frame para seguir
            break

    if datos_ojo is None:
        print("No se pudo detectar el ojo en los primeros 200 frames.")
        return None

    # 🔥 NUEVO: ROI para estabilizar = ojo completo, ROI para seguimiento = iris
    roi_ojo = datos_ojo["bbox_ojo"]       # (x, y, w, h) ojo entero
    roi_iris = datos_ojo["bbox_iris"]     # (x, y, w, h) solo iris

    # 🔥 NUEVO: Ajustar ROI del ojo para evitar incluir ceja/piel excesiva
    x, y, w, h = roi_ojo
    recorte_margen = 0.15  # Reducir 15% de ancho y alto
    x += int(w * recorte_margen)
    y += int(h * recorte_margen)
    w = int(w * (1 - 2 * recorte_margen))
    h = int(h * (1 - 2 * recorte_margen))
    roi_ojo_ajustado = (x, y, w, h)

    # 2. Preparar histograma HSV para seguimiento con meanShift
    # 🔥 Preparar histograma HSV solo con el iris (evita piel/párpado)
    hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_prev, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hsv = hsv_prev[roi_iris[1]:roi_iris[1]+roi_iris[3], roi_iris[0]:roi_iris[0]+roi_iris[2]]
    mask_roi = mask[roi_iris[1]:roi_iris[1]+roi_iris[3], roi_iris[0]:roi_iris[0]+roi_iris[2]]
    roi_hist = cv2.calcHist([roi_hsv], [0], mask_roi, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # 🔥 rect_estabilizar = ojo, rect_seguimiento = iris
    rect_estabilizar = roi_ojo_ajustado
    rect_seguimiento = roi_iris

    # Preparar vídeo salida
    ruta_salida = os.path.splitext(ruta_video)[0] + "_estabilizado.mp4" # Nombre vídeo corregido
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Códec para vídeo mp4
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto)) # Crear archivo vídeo salida

    out.write(prev_frame)  # Escribir primer frame sin corregir

    ### rect = roi  # Variable para almacenar posición actual del ROI

    while True: # Leer cada frame
        ok, curr_frame = cap.read() # Leer frame actual
        if not ok: # Si no hay más frames
            break

        # 3. Convertir frame actual a HSV y calcular backprojection del histograma ROI
        hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 4. Aplicar meanShift para encontrar nuevo recuadro del ROI
        ret, new_rect = cv2.meanShift(back_proj, rect, 
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

        # 5. Calcular desplazamiento para corrección
        dx = new_rect[0] - rect[0]
        dy = new_rect[1] - rect[1]

        # 6. Corregir movimiento del frame actual moviéndolo en dirección opuesta al desplazamiento
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        frame_corregido = cv2.warpAffine(curr_frame, M, (ancho, alto))

        rect = new_rect  # Actualizar posición del ROI para el siguiente frame

        # 7. Dibujar ROI y centro en el frame corregido (solo visual)
        frame_mostrar = frame_corregido.copy() # Copia para mostrar
        x, y, w, h = rect
        cv2.rectangle(frame_mostrar, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(frame_mostrar, (cx, cy), 4, (255, 0, 0), -1)

        # Mostrar frame corregido en ventana
        show_letterboxed(frame_mostrar, win="Video estabilizado") # Mostrar frame corregido en ventana

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
