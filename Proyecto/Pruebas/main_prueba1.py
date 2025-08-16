from src.seguimiento import seguimiento
from src_prueba3.analisis import analisis
from src.preprocesar import preprocesar
import os
import cv2  # Importar OpenCV

def main():
    # Vídeo
    ruta_base = os.path.dirname(os.path.abspath(__file__))  # Carpeta src/
    ruta_video = os.path.join(ruta_base, 'videos', 'pelota_fondo', 'video_pelota.mov')

    ruta_video_preprocesado = preprocesar(ruta_video)

    seguimiento(ruta_video_preprocesado)

    analisis()

    # - NO EJECUTAR SI ES IMPORTADO -
if __name__ == "__main__":
    main()
