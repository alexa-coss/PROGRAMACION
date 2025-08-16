import cv2
import numpy as np

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Inicializar el primer frame para la detección de movimiento
_, first_frame = cap.read()
first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)

movement_detected = False
movement_timestamps = []  # Guardamos los tiempos de los movimientos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a escala de grises y aplicar un desenfoque
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Calcular la diferencia entre el primer frame y el frame actual
    frame_delta = cv2.absdiff(first_frame, gray_frame)
    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)

    # Encontrar los contornos del movimiento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movement_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Ajusta este valor según el tamaño del objeto
            movement_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if movement_detected:
        movement_timestamps.append(cv2.getTickCount())  # Guardamos el tiempo del movimiento

    # Mostrar el video en la ventana
    cv2.imshow("Frame", frame)

    # Si el movimiento es rítmico, lo determinamos por los intervalos de tiempo entre movimientos
    if len(movement_timestamps) > 1:
        intervals = np.diff(movement_timestamps) / cv2.getTickFrequency()
        if len(intervals) > 1:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            if std_interval < 0.1 * avg_interval:
                print("Movimiento rítmico detectado")
            else:
                print("Movimiento no rítmico detectado")

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
