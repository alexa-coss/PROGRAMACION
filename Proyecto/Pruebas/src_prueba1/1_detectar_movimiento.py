import cv2  # Importar OpenCV
from screeninfo import get_monitors

# Abrir vídeo (archivo o cámara)
captura = cv2.VideoCapture('../videos/VID-20220609-WA0022.mp4') # → 0 para la cámara web (tiempo real) | 'nombre_archivo.mp4'

# Leer primer fotograma (imagen) como frame base
ret, frame = captura.read() # Capturar fotograma.
    # 'captura' objeto de la clase 'cv2.VideoCapture', representa la cámara o archivo de vídeo.
    # '.read()' intenta capturar un fotograma del video.)
        # 'ret' booleano (True o False), indica si el fotograma fue leído correctamente.
        # 'frame' array de NumPy, contiene la imagen del fotograma capturado.

# Convertir a escala de grises (comparar más fácil)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 'cvtColor' Color Convert. → cvtColor(imagen_original, tipo_conversion)
    # 'COLOR_BGR2GRAY' tipo de conversión (BGR to GRAY).

# Aplicar desenfoque para reducir ruido
frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    # 'GaussianBlur' desenfoque gaussiano a una imagen
    # '(21, 21)'  tamaño del kernel (o máscara de convolución), ventana de 21x21 px | números impares | al aumentar, aumenta el desenfoque.
    # '0' desviación estándar (sigma) en X. | 0 → calcular automáticamente según tamaño de kernel.

# No se pudo inicializar vídeo
if not captura.isOpened():
    print("Error al abrir la cámara o reproducir video.")
    exit()

    #  Ventana

# Obtener resolución pantalla automáticamente
monitor = get_monitors()[0]
pantalla_ancho = monitor.width
pantalla_alto = monitor.height

# Obtener dimensiones del video
video_ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
video_alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear ventana fullscreen
cv2.namedWindow("Video Fullscreen", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Video Fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Leer fotograma nuevo
    ret, frame = captura.read()
    if not ret:  # Si la captura falla
        print("Vídeo terminado.")
        break  # salir.

    # Convertir a escala de grises y aplicar desenfoque
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Restar imagen actual con el frame base (ver diferencias)
    diferencia = cv2.absdiff(frame_gray, gray)
        # 'absdiff' calcular diferencia absoluta entre dos imágenes pixel por pixel.
    frame_gray = gray  # ¡Actualizar comparación del siguiente ciclo!

    # Aplicar umbral para detectar áreas con cambios (movimiento)
    _, umbral = cv2.threshold(diferencia, 35, 255, cv2.THRESH_BINARY)
        # 'threshold' aplica umbral a una imagen en escala de grises | Convertir imagen en blanco → movimiento y negro.
        # '25' umbral mínimo. | px > 25 se converte en 255 (blanco); px <= 25 se converte en 0 (negro).
        # '255' valor máximo (blanco).
        # 'THRESH_BINARY' tipo de umbral | Imagen binaria: negro (0) o blanco (255).
            # '_' umbral usado | se ignora.
            # 'umbral' imagen binaria resultante.

    # Agrandamos las zonas blancas para que se vean mejor
    umbral = cv2.dilate(umbral, None, iterations=2)
        # 'dilate' dilatación sobre una imagen binaria.
        # 'None' kernel (forma del filtro) | None → kernel cuadrado 3x3 por defecto.
        # 'iterations=2' aplicar dilatación dos veces → expansión más fuerte (zonas blancas crecen más).

    # Eliminar manchas pequeñas por cambios de luz.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    umbral = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)

    texto_estado = "Estado: No se ha detectado movimiento"
    color = (0, 255, 0)

    # Buscamos contornos (bordes) de objetos en movimiento
    contornos, _ = cv2.findContours(umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 'findContours' detectar contornos (bordes) de regiones blancas en imagen binaria.
        # 'umbral.copy()' se pasa una copia | evitar cambiar original.
        # 'RETR_EXTERNAL' solo contornos externos, ignorar internos dentro de otros.
        # 'CHAIN_APPROX_SIMPLE' reducir cantidad de puntos en el contorno guardando, solo los necesarios.
            # 'contornos' lista de arrays de puntos, cada uno representa un contorno detectado.
            # '_' jerarquía (estructura de relaciones entre contornos) | ignora.

        # 
    # Calcular área total de movimiento
    area_total_movimiento = sum(cv2.contourArea(c) for c in contornos) # Suma área de todos los contornos detectados.
    area_total_frame = frame.shape[0] * frame.shape[1]
    porcentaje_movimiento = (area_total_movimiento / area_total_frame) * 100

    # Si el cambio representa más del 20% del frame, ignorar.
    if porcentaje_movimiento > 20:
        contornos = []  # Vaciar la lista para no dibujar rectángulos
        #

    for contorno in contornos: # recorrer cada contorno del array.
        area = cv2.contourArea(contorno)
        # print(area)  # Mostrar tamaño del área
        if area < 1000 or area > 50000:  # Si el área es muy pequeña o muy grande
            continue  # Ignorar.

        texto_estado = "Estado: Alerta, movimiento detectado!"
        color = (0, 0, 255)

        # Dibujar rectángulo alrededor del objeto en movimiento
        (x, y, w, h) = cv2.boundingRect(contorno)
            # 'boundingRect' calcular rectángulo más pequeño que contiene al contorno.
                # x, y: coordenadas de la esquina superior izquierda | w, h: ancho y alto del rectángulo.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 'rectangle' dibujar rectángulo.
            # (0, 255, 0), 2 → (verde), grosor.

        # Texto

    # Parámetros de fuente
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Posición donde mostrar texto (coordenadas)
    x, y = 10, 30  # posición base (línea base)

    # Calcular tamaño del texto
    (text_width, text_height), baseline = cv2.getTextSize(texto_estado, font, font_scale, thickness)

    # Dibujar rectángulo de fondo que se ajusta al tamaño del texto
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 2, y + baseline + 0),
        (0, 0, 0),  # fondo negro
        -1  # relleno
    )

    # Mostrar texto en la parte superior izquierda
    #cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, texto_estado, (x, y), font, font_scale, color, thickness)


        # Ventana
    
    relacion_video = video_ancho / video_alto
    relacion_pantalla = pantalla_ancho / pantalla_alto

    if relacion_video > relacion_pantalla:
        nuevo_ancho = pantalla_ancho
        nuevo_alto = int(pantalla_ancho / relacion_video)
    else:
        nuevo_alto = pantalla_alto
        nuevo_ancho = int(pantalla_alto * relacion_video)

    frame_redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

    frame_final = cv2.copyMakeBorder(
        frame_redimensionado,
        top=(pantalla_alto - nuevo_alto) // 2,
        bottom=(pantalla_alto - nuevo_alto + 1) // 2,
        left=(pantalla_ancho - nuevo_ancho) // 2,
        right=(pantalla_ancho - nuevo_ancho + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    # Mostrar video en una ventana
    cv2.imshow("Detector de movimiento", frame_final) # Nombre de la ventana.

    # Salir al presionar 'Esc'
    k = cv2.waitKey(50) & 0xFF
        # 'waitKey()' esperar por una entrada del teclado → 1000[ms]/FPS = 1000/20 =50.
        # obtener solo valor del carácter presionado.
    if k == 27: # '27' Esc en código ASCII.
        break # Cerrar

# Cerrar todo
captura.release() # Liberar cámara o video.
cv2.destroyAllWindows()
