import subprocess  # Permite ejecutar comandos del sistema (como ffprobe)
import json        # Leer datos que ffprobe devuelve (formato JSON)
import cv2
import os
from pymediainfo import MediaInfo

def rotacion(ruta_video):            
    # Crear ruta para guardar vídeo corregido
    base, ext = os.path.splitext(ruta_video)
    ruta_salida = base + '_rotado' + ext

        # SI YA ESTA PREPROCESADO
    if os.path.exists(ruta_salida):
        print("⚠️ Video ya corregido encontrado. Usando vídeo existente.")
        return ruta_salida

    rotacion = obtener_rotacion(ruta_video)

    cap = cv2.VideoCapture(ruta_video) # Abrir video original y leerlo frame por frame

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Cuadros por segundo 
    
    if rotacion == 0:
        print("Rotación ejecutada. No fue necesario corregir rotación.")
        return ruta_video # Sin rotación, usar original

    # Invertir ancho y alto si es 90° o 270°
    if rotacion in [90, 270]:
        width, height = height, width

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Definir códec (formato) para guardar video
    out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height)) # Preparar objeto VideoWriter para guardar nuevo video corregido

    while True:
        ret, frame = cap.read() # Leer cada frame del video original
        if not ret:
            break
        frame_rotado = corregir_orientacion(frame, rotacion)
        out.write(frame_rotado) # Escribir frame rotado en el nuevo video

    cap.release() # Cerrar vídeo original (cap)
    out.release() # Cerrar vídeo rotado (out)

    print("Rotación ejecutada. Datos guardados.")

    return ruta_salida # devolver ruta del vídeo

    
def obtener_rotacion(ruta_video):
    # pymediainfo para detectar rotación del video desde metadatos
    media_info = MediaInfo.parse(ruta_video)

    for track in media_info.tracks: # Recorrer todas las pistas encontradas en el archivo de video
        if track.track_type == "Video" and track.rotation is not None: # Verifica si la pista es tipo "Video" y contiene información de rotación
            rotacion = int(float(track.rotation)) #  Convertir rotación a entero
            print("Rotación detectada (MediaInfo):", rotacion)
            return rotacion # Devolver rotación
    return 0 # Sin rotación detectada


def corregir_orientacion(frame, rotacion):
    # Aplicar rotación al frame si es necesario
    if rotacion == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # Girar 90° en sentido horario
    elif rotacion == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotacion == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # Girar 90° en sentido antihorario
    return frame # Si no hay rotación, devuelve el frame tal cual


# Solo ejecuta si este archivo se corre directamente (bloque de prueba individual)
if __name__ == "__main__":
    rotacion()
