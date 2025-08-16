from src.video.video import video
from src_prueba3.analisis import analisis
import os
import cv2  # Importar OpenCV

def main():
    # Vídeo
    ruta_base = os.path.dirname(os.path.abspath(__file__))  # Carpeta src/
    ruta_video = os.path.join(ruta_base, 'videos', 'nistagmo', 'Prueba2.mp4')

    video(ruta_video)

    analisis()

    # - NO EJECUTAR SI ES IMPORTADO -
if __name__ == "__main__":
    main()
