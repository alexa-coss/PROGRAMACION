# Leer video

import cv2
import numpy as np

# Activar cámara o leer vídeo.
video = cv2.VideoCapture('Video.mp4') # cv2.VideoCapture(0) → Crea un objeto VideoCapture que abre el archivo.

# Leer primer frame
ret, prev_frame = video.read()
if not ret:
    print("Error al abrir el video.")
    exit()

#Convertir a escala de grises (primer frame)
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:

    ret, frame = video.read() # Capturar un fotograma de un video o de una cámara en tiempo real.
    if ret == False: # Si la captura falla, salir.
        print("Error al reproducir vídeo.")
        break

    #Convertir a escala de grises (frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Restar el frame con primer frame (prev_frame)
    diff = cv2.absdiff(prev_gray, gray)

    # Aplicar umbral para resaltar el movimiento
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    # Mostrar resultados
    cv2.imshow('Vídeo',frame) # Nombre de la ventana.
    cv2.imshow('Detector de movimiento',thresh) # Nombre de la ventana.

    # Actualizar frame
    prev_gray = gray.copy() # Copiar frame actual para usarlo en la siguiente iteración
    
    # Cerrar al presionar 'Esc'
    k = cv2.waitKey(50) & 0xFF # 50 → 1000[ms]/FPS = 1000/20 =50
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()