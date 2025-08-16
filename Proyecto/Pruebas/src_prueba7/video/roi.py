from utils_silencio import suppress_stdout_stderr

with suppress_stdout_stderr():
    import mediapipe as mp

import cv2, numpy as np  # Importa OpenCV, NumPy y MediaPipe

mp_face = mp.solutions.face_mesh  # Crea acceso rápido al módulo FaceMesh de MediaPipe

# Índices de puntos clave para ojo e iris en malla facial
LEFT_EYE_IDX = [33, 133, 159, 145, 153, 154, 155, 246] # contorno ojo izq
LEFT_IRIS_IDX = [468, 469, 470, 471] # iris izq
RIGHT_EYE_IDX = [362, 263, 386, 374, 380, 381, 382, 466] # contorno ojo der
RIGHT_IRIS_IDX = [473, 474, 475, 476] # iris der

# Guardar lado fijo y última caja detectada
_lado_fijo = None # Guardar elegido
_ultimo_bbox_ojo = None
_ultimo_bbox_iris = None
_ultimo_iris_data = None

    # - CREAR RECUADRO ALREDEDOR DEL OJO -
def _bbox_from_points(points, img_w, img_h, margin=0.3):
    # Calcular recuadro alrededor del ojo a partir de puntos, 
    # ampliar un margen y ajustar para que no se salga de la imagen

    pts = np.array(points, dtype=np.float32)  # Lista → array NumPy
    x1, y1 = pts.min(axis=0)  # Punto mínimo (esquina sup izq)
    x2, y2 = pts.max(axis=0)  # Punto máximo (esquina inf der)
    w, h = x2-x1, y2-y1  # Ancho y alto
    # Ampliar caja con margen
    x1 -= w*margin; y1 -= h*margin # Restar sup izq
    x2 += w*margin; y2 += h*margin # Restar inf der
    # Limitar caja a bordes de imagen
    x1 = int(max(0, x1)); y1 = int(max(0, y1))
    x2 = int(min(img_w-1, x2)); y2 = int(min(img_h-1, y2))
    return (x1, y1, x2-x1, y2-y1) # (x, y, ancho, alto)


    # - CALCULAR CENTRO Y RADIO -
def _center_radius(points):
    pts = np.array(points, dtype=np.float32) # Lista → array
    cx, cy = pts.mean(axis=0) # Centro promedio
    r = np.mean(np.linalg.norm(pts - [cx, cy], axis=1)) # Radio medio
    return float(cx), float(cy), float(r) # Centro y radio


    # - PARA UN OJO -
# Helper para crear un bbox a partir del centro del iris y su radio.
def _bbox_from_center(cx, cy, r, m=2.5, w=0, h=0):
    # Generar rectángulo (x,y,w,h) alrededor de iris con margen m*r, recortado a la imagen.
    x1 = int(max(0, cx - m*r)); y1 = int(max(0, cy - m*r))
    x2 = int(min(w-1, cx + m*r)); y2 = int(min(h-1, cy + m*r))
    return (x1, y1, x2 - x1, y2 - y1)


    # - PARA UN OJO -
# Fallback con HoughCircles para detectar pupila cuando MediaPipe no da landmarks (close-up de un solo ojo).
def _fallback_pupil_hough(frame_bgr, prev=None):
    # Normalizar contraste y suavizar para resaltar pupila
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.medianBlur(g, 5)
    # Buscar círculos oscuros (pupila). Ajustar min/maxRadius video cambia de escala
    circles = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=80, param2=18, minRadius=5, maxRadius=120
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles[0]))
    if prev is not None:
        px, py = prev
        # El círculo más cercano al anterior para dar estabilidad temporal
        c = min(circles, key=lambda c: (c[0]-px)**2 + (c[1]-py)**2)
    else:
        # Al iniciar, el más grande suele ser más estable
        c = max(circles, key=lambda c: c[2])
    return int(c[0]), int(c[1]), int(c[2])


    # - DETECTAR OJO E IRIS -
def detectar_ojos_e_iris(frame_bgr, lado_fijo=None):
    global _lado_fijo, _ultimo_bbox_ojo, _ultimo_bbox_iris, _ultimo_iris_data

    # Detectar cara con MediaPipe FaceMesh, obtiener puntos clave (landmarks) y convertirlos a coordenadas (píxeles).
    h, w = frame_bgr.shape[:2] # Obtener dimensiones de imagen frame_bgr. | .shape devuelve (alto, ancho, canales), [:2] toma alto y ancho.
    with mp_face.FaceMesh(static_image_mode=False,
                          refine_landmarks=True, # Necesario para iris
                          max_num_faces=1, # Solo 1 cara
                          min_detection_confidence=0.3,
                          min_tracking_confidence=0.3) as mesh:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # BGR → RGB
        res = mesh.process(rgb) # Procesar cara
        if not res.multi_face_landmarks: # Si no detecta
            # Si no detecta cara, usar último ojo detectado
            if _ultimo_bbox_ojo is not None:
                return {
                    "lado": _lado_fijo,
                    "bbox_ojo": _ultimo_bbox_ojo,
                    "bbox_iris": _ultimo_bbox_iris,
                    "iris": _ultimo_iris_data
                }
            # Si no hay historial, intentar fallback por pupila (un ojo / close-up).
            circ = _fallback_pupil_hough(frame_bgr, prev=(_ultimo_iris_data[:2] if _ultimo_iris_data else None))
            if circ is not None:
                cx, cy, r = circ
                bbox_iris = _bbox_from_center(cx, cy, r, m=2.0, w=w, h=h)  #
                bbox_ojo  = _bbox_from_center(cx, cy, max(r*2.2, 20), m=1.4, w=w, h=h)  #
                if _lado_fijo is None:  # Etiquetar por consistencia si no se había lado elegido
                    _lado_fijo = "right"
                _ultimo_bbox_ojo  = bbox_ojo  # cache
                _ultimo_bbox_iris = bbox_iris
                _ultimo_iris_data = (int(cx), int(cy), float(r))
                return {
                    "lado": _lado_fijo,
                    "bbox_ojo": bbox_ojo,
                    "bbox_iris": bbox_iris,
                    "iris": _ultimo_iris_data
                }
            return None
        lm = res.multi_face_landmarks[0].landmark # Landmarks cara
        def xy(i): return (lm[i].x*w, lm[i].y*h) # Escalar coord a pixeles

        # Obtener puntos ojo coordenadas (píxeles)
        left_eye  = [xy(i) for i in LEFT_EYE_IDX]
        right_eye = [xy(i) for i in RIGHT_EYE_IDX]
        left_iris  = [xy(i) for i in LEFT_IRIS_IDX]
        right_iris = [xy(i) for i in RIGHT_IRIS_IDX]

        # Calcular área
        # Convertir lista de puntos en array de NumPy de enteros (int32). OpenCV trabajará con datos en formato estándar.
        area_L = cv2.contourArea(np.array(left_eye, dtype=np.int32))
        area_R = cv2.contourArea(np.array(right_eye, dtype=np.int32))

        # Si recibe lado_fijo como argumento, usarlo
        if lado_fijo is not None:
            _lado_fijo = lado_fijo

        if _lado_fijo is None: # Si no hay lado elegido
            if area_L > 0 and area_R > 0: # Si ambos se detectaron
                _lado_fijo = "left" if area_L >= area_R else "right" # Si ojo izquierdo es más grande o igual qeu derecho
            elif area_L > 0: # Si solo detecta ojo izquierdo
                _lado_fijo = "left"
            elif area_R > 0: # Si solo detecta ojo derecho
                _lado_fijo = "right"
            else: # Si no detecta ojo
                # Si no hay contornos de ojo, intentar con iris directamente vía fallback.
                circ = _fallback_pupil_hough(frame_bgr, prev=(_ultimo_iris_data[:2] if _ultimo_iris_data else None))
                if circ is not None:
                    cx, cy, r = circ
                    bbox_iris = _bbox_from_center(cx, cy, r, m=2.0, w=w, h=h)
                    bbox_ojo  = _bbox_from_center(cx, cy, max(r*2.2, 20), m=1.4, w=w, h=h)
                    _lado_fijo = _lado_fijo or "right"
                    _ultimo_bbox_ojo  = bbox_ojo
                    _ultimo_bbox_iris = bbox_iris
                    _ultimo_iris_data = (int(cx), int(cy), float(r))
                    return {
                        "lado": _lado_fijo,
                        "bbox_ojo": bbox_ojo,
                        "bbox_iris": bbox_iris,
                        "iris": _ultimo_iris_data
                    }
                return None

        # Intentar obtener siempre el mismo lado
        if _lado_fijo == "left" and len(left_iris) > 0:
            cx, cy, r = _center_radius(left_iris) # Centro y radio de iris
            # Forzar que el ROI siempre se centre en el iris (mejor para close-up de un solo ojo)
            bbox_ojo = _bbox_from_center(cx, cy, r, m=2.5, w=w, h=h)
            bbox_iris = _bbox_from_center(cx, cy, r, m=2.2, w=w, h=h)

        elif _lado_fijo == "right" and len(right_iris) > 0:
            cx, cy, r = _center_radius(right_iris) # Centro y radio de iris
            bbox_ojo = _bbox_from_center(cx, cy, r, m=2.5, w=w, h=h)
            bbox_iris = _bbox_from_center(cx, cy, r, m=2.2, w=w, h=h)

        else:
            # Si no detecta el ojo fijo ni el iris, usar último guardado
            if _ultimo_bbox_ojo is not None:
                return {
                    "lado": _lado_fijo,
                    "bbox_ojo": _ultimo_bbox_ojo,
                    "bbox_iris": _ultimo_bbox_iris,
                    "iris": _ultimo_iris_data
                }
            # Último intento con fallback (puede pasar que FaceMesh no dé iris aunque haya cara)
            circ = _fallback_pupil_hough(frame_bgr, prev=(_ultimo_iris_data[:2] if _ultimo_iris_data else None))
            if circ is not None:
                cx, cy, r = circ
                bbox_iris = _bbox_from_center(cx, cy, r, m=2.0, w=w, h=h)
                bbox_ojo  = _bbox_from_center(cx, cy, max(r*2.2, 20), m=1.4, w=w, h=h)
                _lado_fijo = _lado_fijo or "right"
                _ultimo_bbox_ojo  = bbox_ojo
                _ultimo_bbox_iris = bbox_iris
                _ultimo_iris_data = (int(cx), int(cy), float(r))
                return {
                    "lado": _lado_fijo,
                    "bbox_ojo": bbox_ojo,
                    "bbox_iris": bbox_iris,
                    "iris": _ultimo_iris_data
                }
            return None

        # Guardar última detección para usar si se pierde
        _ultimo_bbox_ojo = bbox_ojo
        _ultimo_bbox_iris = bbox_iris
        _ultimo_iris_data = (cx, cy, r)

        # Devolver datos
        return {"lado": _lado_fijo, "bbox_ojo": bbox_ojo, "bbox_iris": bbox_iris, "iris": (cx, cy, r)}

def reset_roi_state():
    global _lado_fijo, _ultimo_bbox_ojo, _ultimo_bbox_iris, _ultimo_iris_data
    _lado_fijo = None
    _ultimo_bbox_ojo = None 
    _ultimo_bbox_iris = None
    _ultimo_iris_data = None
