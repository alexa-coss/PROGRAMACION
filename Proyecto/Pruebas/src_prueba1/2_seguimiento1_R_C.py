import cv2  # Importar OpenCV
import time
t0 = time.time()  # tiempo de inicio

# Abrir video
video = cv2.VideoCapture('../videos/VID-20220609-WA0022.mp4')

# Leer primer frame
ok, frame = video.read()
if not ok:
    print("No se pudo leer el video.")
    exit()

# Seleccionar región (ROI) manualmente
bbox = cv2.selectROI("Selecciona el objeto a seguir", frame, False) # Abrir ventana para que el usuario seleccione el objeto a seguir (dibujando un rectángulo).
    # 'selectROI' seleccionar Región de Interés (ROI) con mouse.
    # 'False' no permite selección múltiple (una ROI a la vez).
    # 'bbox' tupla (x, y, w, h) con posición y tamaño del rectángulo.
cv2.destroyWindow("Selecciona el objeto a seguir") # Cerrar ventana

# Inicializar tracker
tracker = cv2.TrackerCSRT_create() # Crear objeto de seguimiento (tracker) usando algoritmo CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability).
    # 'TrackerCSRT' tipo de tracker preciso y robusto, ideal para objetos con movimiento lento. Útil para objetos que cambian de escala, rotación o iluminación.
    # 'create()' método que instancia (crea) el tracker.
tracker.init(frame, bbox) # Inicializar tracker con el primer cuadro (frame) y la región seleccionada (bbox).
    # 'init' vincular tracker al objeto a seguir desde el cuadro inicial.

# Lista para guardar la trayectoria
trayectoria = []

while True:
    # Leer nuevo frame
    ok, frame = video.read()
    if not ok:
        break

        # Actualizar posición del objeto en nuevo cuadro (frame).
    ok, bbox = tracker.update(frame)
        # 'update' método que analiza el nuevo frame para encontrar el objeto.
            # 'ok' booleano (True o False) indica si el objeto fue encontrado exitosamente.
            # 'bbox' tupla (x, y, w, h) con nueva posición y tamaño del objeto en el frame actual.

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
        cv2.putText(frame, f"Posición: ({cx}, {cy})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Guardar posición
        trayectoria.append((cx, cy))
            # Agregar coordenadas actuales del centro (cx, cy) a 'trayectoria'.
    else: # si el objeto no fue encontrado (false).
        cv2.putText(frame, "Objeto perdido", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Mostrar video en una ventana
    cv2.imshow("Seguimiento de objeto", frame)

    # Salir al presionar 'Esc'
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Cerrar todo
video.release() # Liberar cámara o video.
cv2.destroyAllWindows()

# Guardar coordenadas para análisis
with open("2_seguimiento1_R_C.txt", "w") as archivo: # Abrir o crea archivo 'trayectoria.txt' en modo escritura 'w'. (Si ya existe, sobrescribir)
    for cx, cy in trayectoria: # Recorrer cada par de coordenadas (cx, cy) en'trayectoria'.
        archivo.write(f"{cx},{cy}\n") # Escribir cada par en una línea, separadas por coma y salto de línea al final.

print("Seguimiento terminado. Coordenadas guardadas en trayectoria.txt.")
