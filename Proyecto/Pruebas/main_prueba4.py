from src.procesar_video import procesar_video
from src.analisis import analisis
import os
import cv2  # Importar OpenCV

def main():
    # Vídeo
    ruta_base = os.path.dirname(os.path.abspath(__file__))  # Carpeta src/
    ruta_video = os.path.join(ruta_base, 'videos', 'nistagmo', 'Prueba2.mp4')

    procesar_video(ruta_video)

    analisis()

    # - NO EJECUTAR SI ES IMPORTADO -
if __name__ == "__main__":
    main()
