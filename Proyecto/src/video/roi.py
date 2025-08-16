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

# [NUEVO] Cache por ojo (para modo binocular y/o cuando falte un lado)
_ultimo_bbox_ojo_L = None  # [NUEVO] última caja de ojo izquierdo
_ultimo_bbox_iris_L = None  # [NUEVO] última caja de iris izquierdo
_ultimo_iris_data_L = None  # [NUEVO] último (cx,cy,r) izquierdo

_ultimo_bbox_ojo_R = None  # [NUEVO] última caja de ojo derecho
_ultimo_bbox_iris_R = None  # [NUEVO] última caja de iris derecho
_ultimo_iris_data_R = None  # [NUEVO] último (cx,cy,r) derecho

UMBRAL_AREA = 3000 # ojo chico => close-up (ajusta según tu resolución)
MODO_ANALISIS = "auto" # "auto" | "binocular" | "monocular"

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


# [NUEVO] Construir estructura de datos por ojo de forma consistente
def _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag):
    # [NUEVO] Devuelve dict estándar para un ojo
    return {
        "eye_bbox": bbox_ojo,            # [NUEVO] (x,y,w,h) del ojo
        "iris_bbox": bbox_iris,          # [NUEVO] (x,y,w,h) del iris
        "center": (float(cx), float(cy)),# [NUEVO] centro (cx,cy) del iris
        "radius": float(r),              # [NUEVO] radio del iris
        "monocular": bool(monocular_flag)# [NUEVO] flag de modo monocular
    }


    # - DETECTAR OJO E IRIS (NUEVO: MULTI) -
def detectar_ojos_e_iris_multi(frame_bgr, lado_fijo=None):
    # [NUEVO] Detecta ambos ojos cuando existan. Si solo hay uno, devuelve solo ese.
    # [NUEVO] Mantiene caché por lado para tolerar pérdidas breves.
    global _lado_fijo
    global _ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, _ultimo_iris_data_L
    global _ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, _ultimo_iris_data_R

    h, w = frame_bgr.shape[:2]  # [NUEVO] dimensiones del frame

    # [NUEVO] Respetar lado forzado si se pasa
    if lado_fijo is not None:
        _lado_fijo = lado_fijo  # [NUEVO] guardar elección externa

    # [NUEVO] Atajo: modo monocular forzado → usar Hough si es posible
    if MODO_ANALISIS == "monocular":
        # [NUEVO] Elegir "lado lógico" para etiquetar el monocular (persistente)
        prefer = _lado_fijo if _lado_fijo in ("left", "right") else "right"
        prev = _ultimo_iris_data_L[:2] if prefer == "left" and _ultimo_iris_data_L else (
               _ultimo_iris_data_R[:2] if prefer == "right" and _ultimo_iris_data_R else None)
        circ = _fallback_pupil_hough(frame_bgr, prev=prev)  # [NUEVO] buscar pupila
        L = R = None  # [NUEVO] inicializar respuestas
        if circ is not None:
            cx, cy, r = circ
            bbox_iris = _bbox_from_center(cx, cy, r, m=2.0, w=w, h=h)
            bbox_ojo  = _bbox_from_center(cx, cy, r, m=2.5, w=w, h=h)
            if prefer == "left":
                _ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, _ultimo_iris_data_L = bbox_ojo, bbox_iris, (cx, cy, r)
                L = _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag=True)
            else:
                _ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, _ultimo_iris_data_R = bbox_ojo, bbox_iris, (cx, cy, r)
                R = _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag=True)
            return {
                "left": L, "right": R,              # [NUEVO] devuelve solo el ojo detectado
                "mode": {"requested": "monocular", "auto_decision": "monocular"},
                "has_face": False                   # [NUEVO] no dependió de cara
            }
        # [NUEVO] si Hough falla, continuar con FaceMesh abajo (por si hay cara entera)

    # [NUEVO] FaceMesh para detectar landmarks
    with mp_face.FaceMesh(static_image_mode=False,
                          refine_landmarks=True, # Necesario para iris
                          max_num_faces=1,       # Solo 1 cara
                          min_detection_confidence=0.3,
                          min_tracking_confidence=0.3) as mesh:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # BGR → RGB
        res = mesh.process(rgb) # Procesar cara
        if not res.multi_face_landmarks: # Si no detecta
            # [NUEVO] Devolver desde caché si hay algo
            L = _make_eye_entry(*_ultimo_bbox_ojo_L, *_ultimo_bbox_iris_L, *_ultimo_iris_data_L, True) if False else None  # [NUEVO] plantilla (no usar así)
            # [NUEVO] Mejor: construir desde caché correctamente por lado
            L = (_make_eye_entry(_ultimo_bbox_ojo_L, _ultimo_bbox_iris_L,
                                 _ultimo_iris_data_L[0], _ultimo_iris_data_L[1], _ultimo_iris_data_L[2],
                                 monocular_flag=True)
                 if _ultimo_bbox_ojo_L and _ultimo_bbox_iris_L and _ultimo_iris_data_L else None)
            R = (_make_eye_entry(_ultimo_bbox_ojo_R, _ultimo_bbox_iris_R,
                                 _ultimo_iris_data_R[0], _ultimo_iris_data_R[1], _ultimo_iris_data_R[2],
                                 monocular_flag=True)
                 if _ultimo_bbox_ojo_R and _ultimo_bbox_iris_R and _ultimo_iris_data_R else None)

            # [NUEVO] Si no hay caché, último intento con Hough (un ojo)
            if L is None and R is None:
                prev = None
                circ = _fallback_pupil_hough(frame_bgr, prev=prev)
                if circ is not None:
                    cx, cy, r = circ
                    bbox_iris = _bbox_from_center(cx, cy, r, m=2.0, w=w, h=h)
                    bbox_ojo  = _bbox_from_center(cx, cy, r, m=2.5, w=w, h=h)
                    if (_lado_fijo or "right") == "left":
                        _ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, _ultimo_iris_data_L = bbox_ojo, bbox_iris, (cx, cy, r)
                        L = _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag=True)
                    else:
                        _ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, _ultimo_iris_data_R = bbox_ojo, bbox_iris, (cx, cy, r)
                        R = _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag=True)

            return {
                "left": L, "right": R,              # [NUEVO] puede venir uno, el otro o ninguno
                "mode": {"requested": MODO_ANALISIS, "auto_decision": "monocular"},
                "has_face": False
            }

        # [NUEVO] Sí hay landmarks → extraer coordenadas
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

        # [NUEVO] Decisión automática de modo según tamaño aparente
        modo_auto = "binocular"  # [NUEVO] por defecto binocular
        if MODO_ANALISIS == "auto":
            if area_L < UMBRAL_AREA and area_R < UMBRAL_AREA:
                modo_auto = "monocular"  # [NUEVO] close-up (ambos chicos) → tratar como monocular

        # [NUEVO] Construcción por ojo
        L = None; R = None  # [NUEVO] inicializar

        # [NUEVO] OJO IZQUIERDO
        if len(left_iris) > 0 and area_L > 0:
            cxL, cyL, rL = _center_radius(left_iris)  # [NUEVO] centro/radio iris izq
            if MODO_ANALISIS == "monocular" or modo_auto == "monocular":
                bbox_ojo_L  = _bbox_from_center(cxL, cyL, rL, m=2.5, w=w, h=h)  # [NUEVO] close-up: ROI por iris
                bbox_iris_L = _bbox_from_center(cxL, cyL, rL, m=2.2, w=w, h=h)  # [NUEVO] iris box
                monoL = True  # [NUEVO]
            else:
                bbox_ojo_L  = _bbox_from_points(left_eye, w, h, margin=0.35)  # [NUEVO] cara completa: contorno ojo
                bbox_iris_L = _bbox_from_center(cxL, cyL, rL, m=2.2, w=w, h=h) # [NUEVO]
                monoL = False  # [NUEVO]
            _ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, _ultimo_iris_data_L = bbox_ojo_L, bbox_iris_L, (cxL, cyL, rL)  # [NUEVO] cache
            L = _make_eye_entry(bbox_ojo_L, bbox_iris_L, cxL, cyL, rL, monocular_flag=monoL)  # [NUEVO]
        else:
            # [NUEVO] usar caché si existe
            if _ultimo_bbox_ojo_L and _ultimo_bbox_iris_L and _ultimo_iris_data_L:
                cxL, cyL, rL = _ultimo_iris_data_L
                L = _make_eye_entry(_ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, cxL, cyL, rL, monocular_flag=True)

        # [NUEVO] OJO DERECHO
        if len(right_iris) > 0 and area_R > 0:
            cxR, cyR, rR = _center_radius(right_iris)  # [NUEVO] centro/radio iris der
            if MODO_ANALISIS == "monocular" or modo_auto == "monocular":
                bbox_ojo_R  = _bbox_from_center(cxR, cyR, rR, m=2.5, w=w, h=h)  # [NUEVO]
                bbox_iris_R = _bbox_from_center(cxR, cyR, rR, m=2.2, w=w, h=h)  # [NUEVO]
                monoR = True  # [NUEVO]
            else:
                bbox_ojo_R  = _bbox_from_points(right_eye, w, h, margin=0.35)  # [NUEVO]
                bbox_iris_R = _bbox_from_center(cxR, cyR, rR, m=2.2, w=w, h=h) # [NUEVO]
                monoR = False  # [NUEVO]
            _ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, _ultimo_iris_data_R = bbox_ojo_R, bbox_iris_R, (cxR, cyR, rR)  # [NUEVO]
            R = _make_eye_entry(bbox_ojo_R, bbox_iris_R, cxR, cyR, rR, monocular_flag=monoR)  # [NUEVO]
        else:
            # [NUEVO] usar caché si existe
            if _ultimo_bbox_ojo_R and _ultimo_bbox_iris_R and _ultimo_iris_data_R:
                cxR, cyR, rR = _ultimo_iris_data_R
                R = _make_eye_entry(_ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, cxR, cyR, rR, monocular_flag=True)

        # [NUEVO] Si sigue sin haber nada (caso raro), último intento con Hough global
        if L is None and R is None:
            circ = _fallback_pupil_hough(frame_bgr, prev=None)
            if circ is not None:
                cx, cy, r = circ
                bbox_iris = _bbox_from_center(cx, cy, r, m=2.0, w=w, h=h)
                bbox_ojo  = _bbox_from_center(cx, cy, r, m=2.5, w=w, h=h)
                # [NUEVO] Etiquetar al lado "fijo" si existe, sino elegir por X (derecha)
                side = _lado_fijo if _lado_fijo in ("left","right") else "right"
                if side == "left":
                    _ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, _ultimo_iris_data_L = bbox_ojo, bbox_iris, (cx, cy, r)
                    L = _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag=True)
                else:
                    _ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, _ultimo_iris_data_R = bbox_ojo, bbox_iris, (cx, cy, r)
                    R = _make_eye_entry(bbox_ojo, bbox_iris, cx, cy, r, monocular_flag=True)

        # [NUEVO] Decidir _lado_fijo si no está y hay ambos
        if _lado_fijo is None:
            if L and R:
                # [NUEVO] elegir por área mayor si ambas presentes
                _lado_fijo = "left" if area_L >= area_R else "right"
            elif L:
                _lado_fijo = "left"
            elif R:
                _lado_fijo = "right"

        return {
            "left": L, "right": R,  # [NUEVO] puede venir uno o ambos
            "mode": {"requested": MODO_ANALISIS, "auto_decision": modo_auto},  # [NUEVO]
            "has_face": True  # [NUEVO]
        }


    # - DETECTAR OJO E IRIS -
def detectar_ojos_e_iris(frame_bgr, lado_fijo=None):
    global _lado_fijo, _ultimo_bbox_ojo, _ultimo_bbox_iris, _ultimo_iris_data

    # Detectar cara con MediaPipe FaceMesh, obtiener puntos clave (landmarks) y convertirlos a coordenadas (píxeles).
    h, w = frame_bgr.shape[:2] # Obtener dimensiones de imagen frame_bgr. | .shape devuelve (alto, ancho, canales), [:2] toma alto y ancho.

    # [NUEVO] Usar la versión multi y adaptar al contrato antiguo para no romper al llamador actual.
    multi = detectar_ojos_e_iris_multi(frame_bgr, lado_fijo=lado_fijo)  # [NUEVO] obtener ambos

    # [NUEVO] Elegir qué lado devolver manteniendo API anterior
    if lado_fijo is not None:
        _lado_fijo = lado_fijo  # [NUEVO] respetar orden externo

    # [NUEVO] decidir lado a reportar
    L = multi.get("left")
    R = multi.get("right")
    side = _lado_fijo
    if side is None:
        # [NUEVO] si no hay fijo, elegir el que exista o el más grande (si ambos)
        if L and R:
            side = "left" if L["eye_bbox"][2]*L["eye_bbox"][3] >= R["eye_bbox"][2]*R["eye_bbox"][3] else "right"
        elif L:
            side = "left"
        elif R:
            side = "right"

    # [NUEVO] preparar retorno compatible
    entry = L if side == "left" else (R if side == "right" else None)

    if entry is None:
        # [NUEVO] mantener tus caídas de respaldo previas (historial simple)
        if _ultimo_bbox_ojo is not None:
            return {
                "lado": _lado_fijo,
                "bbox_ojo": _ultimo_bbox_ojo,
                "bbox_iris": _ultimo_bbox_iris,
                "iris": _ultimo_iris_data
            }
        # [NUEVO] sin historial → None
        return None

    # [NUEVO] actualizar historial simple (compat)
    _ultimo_bbox_ojo = entry["eye_bbox"]
    _ultimo_bbox_iris = entry["iris_bbox"]
    _ultimo_iris_data = (entry["center"][0], entry["center"][1], entry["radius"])

    # Devolver datos
    return {
        "lado": side if side else _lado_fijo,               # [NUEVO] etiqueta lado reportado
        "bbox_ojo": entry["eye_bbox"],                      # [NUEVO]
        "bbox_iris": entry["iris_bbox"],                    # [NUEVO]
        "iris": (entry["center"][0], entry["center"][1], entry["radius"]),  # [NUEVO]
        "monocular": entry["monocular"]                     # [NUEVO]
    }

def reset_roi_state():
    global _lado_fijo, _ultimo_bbox_ojo, _ultimo_bbox_iris, _ultimo_iris_data
    _lado_fijo = None
    _ultimo_bbox_ojo = None 
    _ultimo_bbox_iris = None
    _ultimo_iris_data = None

    # [NUEVO] limpiar caches por ojo
    global _ultimo_bbox_ojo_L, _ultimo_bbox_iris_L, _ultimo_iris_data_L
    global _ultimo_bbox_ojo_R, _ultimo_bbox_iris_R, _ultimo_iris_data_R
    _ultimo_bbox_ojo_L = None
    _ultimo_bbox_iris_L = None
    _ultimo_iris_data_L = None
    _ultimo_bbox_ojo_R = None
    _ultimo_bbox_iris_R = None
    _ultimo_iris_data_R = None
