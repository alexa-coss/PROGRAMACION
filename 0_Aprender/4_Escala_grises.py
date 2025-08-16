# Convertir a escala de grises video

import cv2

# Activar cámara o leer vídeo.
cap = cv2.VideoCapture(0) # cv2.VideoCapture(0) → Crea un objeto VideoCapture que accede a la cámara.

while True:

    ret, frame = cap.read() # Capturar un fotograma de un video o de una cámara en tiempo real.

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # →

    if ret == False: # Si la captura falla, salir.
        print("Error al reproducir vídeo.")
        break
    
    cv2.imshow('Detector de movimiento',gray) # Nombre de la ventana. →
    
    # Cerrar al presionar 'Esc'.
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()