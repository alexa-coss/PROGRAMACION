import cv2
import numpy as np
import os

# Punto fijo inicial (guardar al seleccionar)
punto_inicial = None
punto_actual = None


def corregir_movimiento(ruta_video):
    global punto_inicial, punto_actual

    cap = cv2.VideoCapture(ruta_video) # Abrir vídeo desde ruta
    fps = cap.get(cv2.CAP_PROP_FPS) # Obtener cuadros por segundo
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Ancho del vídeo en píxeles (convertir a entero)
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Alto del vídeo en píxeles (convertir a entero)

    ok, prev_frame = cap.read() # Leer primer frame del vídeo
    if not ok:
        print("No se pudo leer el vídeo.")
        return None

    punto_inicial = fijarPunto(prev_frame) # Seleccionar punto fijo en primer frame
    punto_actual = punto_inicial.copy() # Copiar para seguimiento continuo

    # Preparar vídeo salida
    ruta_salida = os.path.splitext(ruta_video)[0] + "_corregido.mp4" # Nombre vídeo corregido
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Códec para vídeo mp4
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (ancho, alto)) # Crear archivo vídeo salida

    out.write(prev_frame)  # Escribir primer frame sin corregir

    while True: # Leer cada frame
        ok, curr_frame = cap.read() # Leer frame actual
        if not ok: # Si no hay más frames
            break

        frame_corr, punto_actual = corregir_video(prev_frame, curr_frame, punto_actual) # Corregir movimiento del frame actual respecto al anterior y actualizar posición del punto fijo

        cv2.imshow("Video corregido", frame_corr)  # Mostrar frame corregido en ventana
        if cv2.waitKey(1) & 0xFF == 27:
            print("Salida anticipada por ESC")
            break

        out.write(frame_corr) # Guardar frame corregido en archivo de video de salida
        prev_frame = frame_corr # Actualizar prev_frame para comparar con el siguiente frame en la próxima iteración

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video corregido guardado en: {ruta_salida}")
    return ruta_salida


def fijarPunto(frame):
        # - INSTRUCCIONES -
    # Mostrar instrucciones antes de seleccionar ROI
    frame_instrucciones = 255 * np.ones_like(frame)  # Fondo blanco
    instrucciones = [
        "Instrucciones",
        "Selecciona un punto fijo del rostro.",
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

    # - SELECCIONAR PUNTO -
    # Muestrar primer frame para seleccionar un punto fijo
    nombre_ventana = "Seleccionar el punto fijo"
    global punto_inicial # Coordenadas del punto fijo seleccionado (se usa como referencia)
    punto_inicial = cv2.selectROI(nombre_ventana, frame, fromCenter=True, showCrosshair=True) # Usuario selecciona punto fijo en el frame (se muestra un cursor centrado)
    cv2.destroyWindow(nombre_ventana)

    # Convertimos el ROI (x, y, w, h) a punto central
    x, y, w, h = punto_inicial # Extraer coordenadas de ROI seleccionado
    punto_inicial = np.array([[x + w // 2, y + h // 2]], dtype=np.float32) # Calcular punto central del ROI y conviertir a formato float32
    return punto_inicial # Devolver punto fijo inicial como coordenada (x, y)


def corregir_video(prev_frame, curr_frame, prev_punto):
    # Usar optical flow para calcular movimiento del punto fijo y corregir frame actual.
    global punto_actual # Guardar posición actual del punto fijo en cada frame

    # Optical flow para seguir punto fijo
    nuevo_punto, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_punto, None) # Calcular nuevo punto (posición actual) usando flujo óptico entre dos frames

    if status[0][0] == 1: # Verificar que punto fijo fue encontrado correctamente en el nuevo frame
        # Calcular desplazamiento del punto fijo en x y y entre los dos frames
        dx = nuevo_punto[0][0] - prev_punto[0][0]
        dy = nuevo_punto[0][1] - prev_punto[0][1]

        # Mover el frame al revés del movimiento
        h, w = curr_frame.shape[:2] # Obtener alto y ancho de frame actual
        M = np.float32([[1, 0, -dx], [0, 1, -dy]]) # Matriz de transformación para mover frame en dirección opuesta al desplazamiento
        frame_corregido = cv2.warpAffine(curr_frame, M, (w, h)) # Aplicar corrección de movimiento al frame actual

        punto_actual = nuevo_punto # Actualizar posición actual del punto fijo para el siguiente frame
        return frame_corregido, nuevo_punto # Devolver frame corregido y nueva posición del punto fijo
    else:
        print("⚠️ No se pudo rastrear el punto fijo")
        return curr_frame, prev_punto # Devolver frame sin corregir
