# Sustracción video

import cv2

# Activar cámara o leer vídeo.
cap = cv2.VideoCapture(0) # cv2.VideoCapture(0) → Crea un objeto VideoCapture que accede a la cámara.

while True:

    ret, frame = cap.read() # Capturar un fotograma de un video o de una cámara en tiempo real.
    # 'cap' objeto de la clase 'cv2.VideoCapture', representa la cámara o el archivo de video.
    # '.read()' intenta capturar un fotograma del video.)
        # 'ret' booleano (True o False), indica si el fotograma fue leído correctamente.
        # 'frame' array de NumPy, contiene la imagen del fotograma capturado.
    ret, frame2 = cap.read() # →
    diff = cv2.absdiff(frame, frame2) # →

    if ret == False: # Si la captura falla, salir.
        print("Error al reproducir vídeo.")
        break
    
    cv2.imshow('Detector de movimiento',diff) # Nombre de la ventana. →
    
    # Cerrar al presionar 'Esc'.
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()