import cv2  # Importar OpenCV
import time


    # - SELECCIONAR PUNTO -

punto_seleccionado = None

def seleccionar_punto(event, x, y, flags, param):
    global punto_seleccionado
    if event == cv2.EVENT_LBUTTONDOWN:
        punto_seleccionado = (x, y)


    # - INICIO -

# Abrir vídeo
video = cv2.VideoCapture('../videos/VID-20220609-WA0022.mp4')

# Leer primer frame
ok, frame = video.read()
if not ok:
    print("No se pudo leer el video.")
    exit()

nombre_ventana = "Seleccionar punto a seguir (click izquierdo)"

cv2.namedWindow(nombre_ventana)
cv2.setMouseCallback(nombre_ventana, seleccionar_punto)

# Mostrar frame y esperar selección
while True:
    frame_temp = frame.copy()

    instrucciones = [
        "Selecciona el punto a seguir dando click izquierdo",
        "Presiona ENTER para continuar"
    ]

    y = 30
    for linea in instrucciones:
        cv2.putText(frame, linea, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += 30  # separa las líneas

    if punto_seleccionado:
        cv2.circle(frame_temp, punto_seleccionado, 5, (0, 255, 0), -1)
    cv2.imshow(nombre_ventana, frame_temp)
    if cv2.waitKey(1) & 0xFF == 13 and punto_seleccionado:  # Enter para continuar
        break

cv2.destroyWindow(nombre_ventana) # Cerrar ventana

# Crear caja pequeña alrededor del punto para inicializar tracker
x, y = punto_seleccionado
w = h = 30  # Tamaño mínimo para que tracker funcione
bbox = (x - w//2, y - h//2, w, h)

# Inicializar tracker
tracker = cv2.TrackerCSRT_create() # Crear objeto de seguimiento (tracker) usando algoritmo CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability).
    # 'TrackerCSRT' tipo de tracker preciso y robusto, ideal para objetos con movimiento lento. Útil para objetos que cambian de escala, rotación o iluminación.
    # 'create()' método que instancia (crea) el tracker.
tracker.init(frame, bbox) # Inicializar tracker con el primer cuadro (frame) y la región seleccionada (bbox).
    # 'init' vincular tracker al objeto a seguir desde el cuadro inicial.


    # - SEGUIMIENTO -

# Lista para guardar la trayectoria
trayectoria = []
t0 = time.time()  # Tiempo inicial

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
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
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

    # Mostrar video en una ventana
    cv2.imshow("Seguimiento de punto", frame)

    # Salir al presionar 'Esc'
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Cerrar todo
video.release() # Liberar cámara o video.
cv2.destroyAllWindows()


    # - GUARDAR -

# Guardar tiempo y coordenadas para análisis
with open("2_seguimiento3_P_CT.txt", "w") as archivo: # Abrir o crea archivo 'trayectoria.txt' en modo escritura 'w'. (Si ya existe, sobrescribir)
    for t, cx, cy in trayectoria: # Recorrer cada par de coordenadas (cx, cy) en'trayectoria'.
        archivo.write(f"{t:.3f},{cx},{cy}\n") # Cada tiempo y posición en una línea, separados por coma y salto de línea al final | 't:.3f' tiempo → 3 decimales).

print("Seguimiento terminado. Coordenadas guardadas en trayectoria.txt.")
