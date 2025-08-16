# Guardar video

import cv2

# Activar cámara o leer vídeo.
cap = cv2.VideoCapture(0) # cv2.VideoCapture(0) → Crea un objeto VideoCapture que accede a la cámara.
save = cv2.VideoWriter('Video.mp4',cv2.VideoWriter_fourcc(*'XVID'),20.0,(640,480))
    # cv2.VideoWriter('Video.avi',...) → Guardar con nombre.
        # cv2.VideoWriter_fourcc(*'XVID') → Especifica el códec de compresión del vídeo.
            # 'XVID', formato popular para archivos AVI.
        # 20.0 → Tasa de fotogramas (FPS, frames per second), 20 cuadros por segundo.
        # (640,480) → Establece la resolución del video en 640x480 píxeles.

while True:

    ret, frame = cap.read() # Capturar un fotograma de un video o de una cámara en tiempo real.
    # cap → objeto de la clase 'cv2.VideoCapture', representa la cámara o el archivo de video.
    # .read() → intenta capturar un fotograma del video.)
        # ret → booleano (True o False), indica si el fotograma fue leído correctamente.
        # frame → array de NumPy, contiene la imagen del fotograma capturado.
    
    if ret == False: # Si la captura falla, salir.
        print("Error al reproducir vídeo.")
        break
    
    cv2.imshow('Detector de movimiento',frame) # Nombre de la ventana.
    save.write(frame)
    
    # Cerrar al presionar 'Esc'.
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
save.release()
cv2.destroyAllWindows()