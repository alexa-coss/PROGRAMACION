import subprocess  # Permite ejecutar comandos del sistema (como ffprobe)
import json        # Leer datos que ffprobe devuelve (formato JSON)
import cv2
import os

def preprocesar(ruta_video):
    # Crear ruta para guardar vídeo corregido
    base, ext = os.path.splitext(ruta_video)
    ruta_salida = base + '_corregido' + ext

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
        # Asumir rotación si es .mov (iPhone típico)
        if ruta_video.lower().endswith(".mov"):
            print("⚠️ Video .mov detectado, rotando 90 grados manualmente.")
            rotacion = 90
        else:
            print("Preprocesamiento ejecutado. No fue necesario corregir rotación.")
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

    print("Preprocesamiento ejecutado. Datos guardados.")

    return ruta_salida # devolver ruta del vídeo

    
def obtener_rotacion(ruta_video):
    # ffprobe para detectar rotación del video desde metadatos
    cmd = [
        r"C:\Users\alexa\Downloads\Programas\ffmpeg-7.1.1-essentials_build\bin\ffprobe.exe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate",
        "-of", "json",
        ruta_video
    ]
    """ cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0", # Selecciona primer stream de video
        "-show_entries", "stream_tags=rotate", # Pedir solo etiqueta 'rotate'
        "-of", "json", # Salida en formato JSON (más fácil procesar)
        ruta_video
    ] """
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) # Ejecutar comando y guardar salida (stdout) y errores (stderr) en result.
    salida = result.stdout.decode('utf-8') # Convetir bytes en texto legible (string)
    try:
        datos = json.loads(salida) # Convetir JSON en un diccionario de Python
        rotacion = int(datos['streams'][0]['tags']['rotate']) # Extraer valor de rotación
        return rotacion # Si existe, devolver rotación (90, 180, 270)
    except (KeyError, IndexError, ValueError):  # Si no hay metadatos o no se puede leer
        return 0  # Sin rotación detectada


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
    preprocesar()
