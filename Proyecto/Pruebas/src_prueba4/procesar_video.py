from .video.rotacion import rotacion  # Función que retorna ruta video rotado
from .video.corregir_movimiento import corregir_movimiento  # Función que selecciona punto, corrige movimiento y guarda
from .video.seguimiento import seguimiento  # Función que hace tracking final
import os

def procesar_video(ruta_video_original):
    # Paso 1: corregir rotación
    ruta_video_rotado = rotacion(ruta_video_original)

    # Paso 2: corregir movimiento y guardar video corregido
    ruta_video_corregido = corregir_movimiento(ruta_video_rotado)

    # Paso 3: seguimiento con video corregido
    seguimiento(ruta_video_corregido)

# Solo ejecuta si este archivo se corre directamente (bloque de prueba individual)
if __name__ == "__main__":
    # Vídeo
    ruta_base = os.path.dirname(os.path.abspath(__file__))  # Carpeta src/
    ruta_video = os.path.join(ruta_base, 'videos', 'nistagmo', 'Prueba2.mp4')
    procesar_video(ruta_video)
