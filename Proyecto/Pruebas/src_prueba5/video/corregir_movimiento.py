import cv2
import numpy as np
import os
import tkinter as tk

def corregir_movimiento(ruta_video):
    # Corrige el movimiento del vídeo usando seguimiento de un recuadro (ROI) con meanShift.

    cap = cv2.VideoCapture(ruta_video) # Abrir vídeo desde ruta
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener cuadros por segundo
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del vídeo en píxeles (convertir a entero)
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Alto del vídeo en píxeles (convertir a entero)

    ok, prev_frame = cap.read() # Leer primer frame del vídeo
    if not ok:
        print("No se pudo leer el vídeo.")
        return None

    # 1. Seleccionar recuadro inicial ROI (x, y, w, h)
    roi = fijarRecuadro(prev_frame)  # Devuelva rectángulo completo

    # 2. Preparar histograma HSV para seguimiento con meanShift
    hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_prev, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hsv = hsv_prev[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    mask_roi = mask[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    roi_hist = cv2.calcHist([roi_hsv], [0], mask_roi, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Preparar vídeo salida
    ruta_salida = os.path.splitext(ruta_video)[0] + "_estabilizado.mp4" # Nombre vídeo corregido
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Códec para vídeo mp4
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto)) # Crear archivo vídeo salida

    out.write(prev_frame)  # Escribir primer frame sin corregir

    rect = roi  # Variable para almacenar posición actual del ROI

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

        cv2.imshow("Video estabilizado", frame_corregido)  # Mostrar frame corregido en ventana
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


def fijarRecuadro(frame):
        # - INSTRUCCIONES -
    # Mostrar instrucciones antes de seleccionar ROI
    frame_instrucciones = 255 * np.ones_like(frame)  # Fondo blanco
    instrucciones = [
        "Instrucciones",
        "Selecciona el recuadro fijo alrededor del ojo, desde el centro hasta el extremo del recuadro.",
        "Presiona ENTER para continuar.",
        "ESC para cancelar."
    ]
    y = 30
    for linea in instrucciones:
        cv2.putText(frame_instrucciones, linea, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y += 30

    nombre_ventanaInt = "Instrucciones"
    cv2.imshow(nombre_ventanaInt, frame_instrucciones)
    cv2.waitKey(5000)  # Mostrar 5 segundos
    cv2.destroyWindow(nombre_ventanaInt)

    # Obtener resolución de pantalla
    def obtener_resolucion_pantalla():
        root = tk.Tk()
        root.withdraw()
        return root.winfo_screenwidth(), root.winfo_screenheight()

    ancho_pantalla, alto_pantalla = obtener_resolucion_pantalla()

    # Calcular escala para que el frame quepa en la pantalla
    alto_frame, ancho_frame = frame.shape[:2]
    escala = min(ancho_pantalla / ancho_frame, alto_pantalla / alto_frame)
    nuevo_ancho = int(ancho_frame * escala)
    nuevo_alto = int(alto_frame * escala)

    # Redimensionar frame
    frame_redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

        # - SELECCIONAR ROI (RECUADRO) -
    cv2.namedWindow("Seleccionar recuadro fijo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seleccionar recuadro fijo", nuevo_ancho, nuevo_alto)
    roi_redimensionado = cv2.selectROI("Seleccionar recuadro fijo", frame_redimensionado, fromCenter=True, showCrosshair=True)
    cv2.destroyWindow("Seleccionar recuadro fijo")

    # Convertir ROI redimensionado al tamaño original del video
    roi = tuple(int(v / escala) for v in roi_redimensionado)

    return roi  # Devuelve el rectángulo (x, y, w, h)
