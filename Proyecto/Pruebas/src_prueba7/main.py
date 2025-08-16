from utils_silencio import suppress_stdout_stderr
from procesar_video import procesar_video
from analisis import analisis
import os
import cv2  # Importar OpenCV

def main():
    # Vídeo
    ruta_base = os.path.dirname(os.path.abspath(__file__))  # Carpeta src/
    ruta_video = os.path.join(ruta_base, '..', 'videos', 'nistagmo', 'Prueba2.mp4')
    ruta_video = os.path.normpath(ruta_video)  # Normalizar la ruta

    procesar_video(ruta_video)

    analisis()

    # - NO EJECUTAR SI ES IMPORTADO -
if __name__ == "__main__":
    main()
