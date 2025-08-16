import os
import cv2  # Importar OpenCV
import time
import numpy as np
import tkinter as tk

def seguimiento(ruta_video):

        # - RESOLUCIÓN DE PANTALLA -

    # Obtener resolución de pantalla
    def obtener_resolucion_pantalla():
        root = tk.Tk()
        root.withdraw()  # Oculta la ventana
        ancho = root.winfo_screenwidth()
        alto = root.winfo_screenheight()
        return ancho, alto

    # Redimensionar frame con bordes negros si sobra espacio
    def redimensionar_con_bordes(frame, ancho_pantalla, alto_pantalla):
        alto_frame, ancho_frame = frame.shape[:2]
        
        escala = min(ancho_pantalla / ancho_frame, alto_pantalla / alto_frame)
        nuevo_ancho = int(ancho_frame * escala)
        nuevo_alto = int(alto_frame * escala)

        frame_redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

        canvas = np.zeros((alto_pantalla, ancho_pantalla, 3), dtype=np.uint8)

        x_offset = (ancho_pantalla - nuevo_ancho) // 2
        y_offset = (alto_pantalla - nuevo_alto) // 2

        canvas[y_offset:y_offset+nuevo_alto, x_offset:x_offset+nuevo_ancho] = frame_redimensionado

        return canvas

        # - RESOLUCIÓN DE PANTALLA -

        # - INICIALIZAR VÍDEO -

    # Abrir video
    video = cv2.VideoCapture(ruta_video)

    # Obtener fps
    fps_video = video.get(cv2.CAP_PROP_FPS)

    # Leer primer frame
    ok, frame = video.read()
    if not ok:
        print("No se pudo leer el video.")
        exit()

        # - PANTALLA -
    ancho_pantalla, alto_pantalla = obtener_resolucion_pantalla()

        # - INSTRUCCIONES -
    # Mostrar instrucciones antes de seleccionar ROI
    frame_instrucciones = 255 * np.ones_like(frame)  # Fondo blanco
    instrucciones = [
        "Instrucciones",
        "Selecciona la pupila o iris a seguir en un recuadro.",
        "Presiona ENTER para continuar.",
        "ESC para cancelar."
    ]
    y = 30
    for linea in instrucciones:
        cv2.putText(frame_instrucciones, linea, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        y += 30

    nombre_ventanaInt = "Instrucciones"
    cv2.imshow(nombre_ventanaInt, frame_instrucciones)
    cv2.waitKey(5000)  # Mostrar 5 segundos
    cv2.destroyWindow(nombre_ventanaInt)

        # - RESOLUCIÓN DE PANTALLA -
    # Calcular escala para ajustar el frame a la pantalla
    alto_frame, ancho_frame = frame.shape[:2]
    escala_roi = min(ancho_pantalla / ancho_frame, alto_pantalla / alto_frame)
    nuevo_ancho = int(ancho_frame * escala_roi)
    nuevo_alto = int(alto_frame * escala_roi)

    # Redimensionar frame para mostrarlo en pantalla completa sin salirse
    frame_para_roi = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

    # Crear ventana para seleccionar ROI y adaptarla al tamaño
    nombre_ventana = "Seleccionar el objeto a seguir"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, nuevo_ancho, nuevo_alto)
    bbox_redimensionado = cv2.selectROI(nombre_ventana, frame_para_roi, False)
    cv2.destroyWindow(nombre_ventana)

    # Convertir las coordenadas del ROI redimensionado al tamaño original
    bbox = tuple(int(v / escala_roi) for v in bbox_redimensionado)
        # - RESOLUCIÓN DE PANTALLA -

        # - TRACKER -
    # Inicializar tracker
    tracker = cv2.TrackerCSRT_create() # Crear objeto de seguimiento (tracker) usando algoritmo CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability).
        # 'TrackerCSRT' tipo de tracker preciso y robusto, ideal para objetos con movimiento lento. Útil para objetos que cambian de escala, rotación o iluminación.
        # 'create()' método que instancia (crea) el tracker.
    tracker.init(frame, bbox) # Inicializar tracker con el primer cuadro (frame) y la región seleccionada (bbox).
        # 'init' vincular tracker al objeto a seguir desde el cuadro inicial.

        # - PARA SEGUIMIENTO -
    # Guardar
    trayectoria = [] # Lista de trayectoria
    t0 = time.time()  # Tiempo inicial
    cx_prev, cy_prev = None, None  # Posición anterior

        # - PANTALLA -
    cv2.namedWindow("Seguimiento de objeto", cv2.WINDOW_NORMAL) # Manejar mi tamaño
    cv2.resizeWindow("Seguimiento de objeto", ancho_pantalla, alto_pantalla)

        # - RUTA PARA GUARDAR DATOS -
    # Ruta absoluta de carpeta raíz del proyecto
    ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # ruta_base apunta a .../Proyecto
    ruta_datos = os.path.join(ruta_base, "datos") # Carpeta datos dentro de la raíz Proyecto
    os.makedirs(ruta_datos, exist_ok=True) # Asegurar que existe carpeta, si no crear

        # - GUARDAR VÍDEO -
    # Configurar vídeo de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
    ruta_video = os.path.join(ruta_datos, "video_seguimiento.mp4")
    video_salida = cv2.VideoWriter(ruta_video, fourcc, fps_video, (ancho_pantalla, alto_pantalla))

        # - SEGUIMIENTO -
    while True:
            # - INICIO -
        # Leer nuevo frame
        ok, frame = video.read()
        if not ok:
            break

            # - SEGUIMIENTO -
        # Actualizar posición del objeto en nuevo cuadro (frame).
        ok, bbox = tracker.update(frame)
            # 'update' método que analiza el nuevo frame para encontrar el objeto.
                # 'ok' booleano (True o False) indica si el objeto fue encontrado exitosamente.
                # 'bbox' tupla (x, y, w, h) con nueva posición y tamaño del objeto en el frame actual.

        cx = cy = None
        
        if ok: # si el objeto fue encontrado (true).
            # Convertir coordenadas a enteros
            x, y, w, h = [int(v) for v in bbox]
                # Extrae valores 'bbox', convetirr a enteros.
                # 'bbox' tiene la forma (x, y, w, h) → posición y tamaño del rectángulo.
                # 'int()' float a enteros. (recorrer cada valor 'v' en 'bbox')
            # Calcula coordenadas del centro del rectángulo.
            cx = x + w // 2  # centro x (Sumar mitad del alto (h) a la coordenada y)
            cy = y + h // 2  # centro y (Sumar mitad del ancho (w) a la coordenada x)

            # Dibujar recuadro y centro
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # rectangle(frame, esquina superior izquierda, esquina inferior derecha, verde, thickness)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                # circle(frame, centro, radio, azul, relleno (-1))
            cv2.putText(frame, f"Posicion: ({cx}, {cy})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tiempo transcurrido
            tiempo = time.time() - t0 

            # Guardar tiempo y posición
            trayectoria.append((tiempo, cx, cy)) # 'append()' agregar al final de la lista.
        else: # si el objeto no fue encontrado (false).
            cv2.putText(frame, "Objeto perdido", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # - ESTADO MOVIMIENTO -
        # Verificar movimiento
        if cx is not None and cx_prev is not None:
            delta = ((cx - cx_prev)**2 + (cy - cy_prev)**2)**0.5 # Distancia entre posición actual y anterior.
            if delta > 2:  # Umbral de movimiento
                texto_estado = "Estado: Alerta, movimiento detectado!"
                color = (0, 255, 0)
            else:
                texto_estado = "Estado: No se ha detectado movimiento"
                color = (0, 0, 255)
        else: # Si no hay coordenadas previas (None).
            texto_estado = "Estado: No se ha detectado movimiento"
            color = (0, 0, 255)

        # Actualizar posición anterior
        cx_prev, cy_prev = cx, cy

        # Mostrar estado en pantalla
        cv2.putText(frame, texto_estado, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Ajustar tamaño de frame a la pantalla
        frame_ajustado = redimensionar_con_bordes(frame, ancho_pantalla, alto_pantalla)

            # - GUARDAR VÍDEO -
        video_salida.write(frame_ajustado)

        # Mostrar video en una ventana
        cv2.imshow("Seguimiento de objeto", frame_ajustado)

            # - SALIR -
        # Salir al presionar 'Esc'
        if cv2.waitKey(int(1000 / fps_video)) & 0xFF == 27:
            # 'waitKey()' esperar por una entrada del teclado → 1000[ms]/FPS = 1000/20 = 50.
            # '27' Esc en código ASCII.
            print("ESC presionado, guardando datos y saliendo...")
            break

            # - CERRAR TODO -
    video.release() # Liberar cámara o video.
    video_salida.release() # Liberar video de salida (guardado).
    cv2.destroyAllWindows()

    # - GUARDAR SEGUIMIENTO -
        # Tiempo y coordenadas
    guardar_datos(trayectoria, ruta_datos)

    print("Seguimiento ejecutado. Datos guardados.")


# - GUARDAR SEGUIMIENTO -
# Tiempo y coordenadas
def guardar_datos(trayectoria, ruta_datos):
    ruta_csv = os.path.join(ruta_datos, "datos_seguimiento.csv")
    with open(ruta_csv, "w") as archivo: # Abrir o crea archivo en modo escritura 'w'. (Si existe, sobrescribir)
        archivo.write("tiempo,x,y\n")  # Encabezado
        for t, cx, cy in trayectoria: # Recorrer cada par de coordenadas (cx, cy) en'trayectoria'.
            archivo.write(f"{t:.3f},{cx},{cy}\n")


# Solo ejecuta si este archivo se corre directamente (bloque de prueba individual)
if __name__ == "__main__":
    seguimiento()
