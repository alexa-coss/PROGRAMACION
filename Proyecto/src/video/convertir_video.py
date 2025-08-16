import cv2, os, subprocess, sys

def asegurar_h264(ruta_video):
    """
    Verifica si un video se puede leer con OpenCV.
    Si no, lo convierte a H.264 y devuelve la ruta del archivo convertido.
    """
    ruta_abs = os.path.abspath(ruta_video)
    # print("[DEBUG] Verificando video:", ruta_abs)

    # 1. Comprobar si existe
    if not os.path.exists(ruta_abs):
        raise FileNotFoundError(f"El archivo no existe: {ruta_abs}")

    # 2. Probar lectura
    cap = cv2.VideoCapture(ruta_abs)
    ok, _ = cap.read()
    cap.release()

    if ok:
        # print("[DEBUG] El video se puede leer, no es necesario convertir.")
        return ruta_abs  # No necesita conversión

    # 3. Convertir a H.264 con ffmpeg
    base, _ = os.path.splitext(ruta_abs)
    nuevo_archivo = base + "_h264.mp4"

    print("[DEBUG] El video no se puede leer, convirtiendo a H.264...")
    comando = [
        "ffmpeg", "-y",
        "-i", ruta_abs,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        nuevo_archivo
    ]

    try:
        subprocess.run(comando, check=True)
    except FileNotFoundError:
        raise RuntimeError("No se encontró ffmpeg. Instálalo y asegúrate de que esté en el PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error al convertir el video: {e}")

    # 4. Comprobar que el convertido sí se lee
    cap = cv2.VideoCapture(nuevo_archivo)
    ok, _ = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"No se pudo leer el video convertido: {nuevo_archivo}")

    print("[DEBUG] Conversión exitosa:", nuevo_archivo)
    return nuevo_archivo


if __name__ == "__main__":
    # Uso rápido desde terminal
    if len(sys.argv) < 2:
        print("Uso: python convertir_video.py ruta_al_video")
        sys.exit(1)
    nueva_ruta = asegurar_h264(sys.argv[1])
    print("Video listo:", nueva_ruta)
