# -*- coding: utf-8 -*-
import cv2, numpy as np, os

# Inicializar detección con MediaPipe y definir índices de ojos/iris si está disponible
_USA_MEDIAPIPE = True # Variable bandera: indica si se intentará usar MediaPipe para detección facial.
try:
    import mediapipe as mp # MediaPipe (detección rostro, ojos, iris, etc.)
    mp_face = mp.solutions.face_mesh # Cargar módulo malla facial (Face Mesh) de MediaPipe.
    
    # Índices de vértices del modelo de malla facial correspondientes a las zonas de ojos e iris
    LEFT_EYE_IDX  = [33,133,159,145,153,154,155,246] # Puntos delimitan contorno ojo izq
    RIGHT_EYE_IDX = [362,263,386,374,380,381,382,466] # Puntos delimitan contorno ojo der
    LEFT_IRIS_IDX  = [468,469,470,471] # Puntos forman el iris izq
    RIGHT_IRIS_IDX = [473,474,475,476] # Puntos forman el iris der
except Exception:
    _USA_MEDIAPIPE = False # Si falla importación o algo del bloque, se desactiva el uso de MediaPipe


# -------- Fallback: Haar Cascades (incluido en OpenCV) --------
# Cargar clasificadores Haar para detección de rostro y ojos (método alternativo a MediaPipe)
_CASCADE_FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") # Cargar clasificador Haar para detección de rostros frontales
_CASCADE_EYE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml") # Cargar clasificador Haar para detección de ojos


# -------- Calcular recuadro delimitador (bounding box) --------
# Calcular bounding box a partir de puntos, aplicando margen y límites de imagen
def _bbox_from_points(points, img_w, img_h, margin=0.35):
    pts = np.array(points, dtype=np.float32)             # Convertir lista de puntos a arreglo NumPy (float32)
    x1, y1 = pts.min(axis=0); x2, y2 = pts.max(axis=0)   # Calcular coordenadas mínimas y máximas (extremos)
    w, h = x2 - x1, y2 - y1                              # Calcular ancho y alto iniciales del box
    
    x1 -= w * margin; y1 -= h * margin # Expandir box hacia afuera (izq/arriba) según margen
    x2 += w * margin; y2 += h * margin # Expandir box hacia afuera (der/abajo) según margen
    
    x1 = int(max(0, x1)); y1 = int(max(0, y1))            # Ajustar coordenadas mínimas para no salir de la imagen
    x2 = int(min(img_w - 1, x2)); y2 = int(min(img_h - 1, y2))  # Ajustar coordenadas máximas al límite de imagen
    
    w = max(1, x2 - x1); h = max(1, y2 - y1) # Recalcular ancho y alto asegurando mínimo 1 píxel
    return (x1, y1, w, h)                    # Retornar bounding box formato (x, y, ancho, alto)


# -------- Calcular centro y radio --------
# Calcular centroide y radio promedio a partir de un conjunto de puntos
def _center_radius(points):
    pts = np.array(points, dtype=np.float32)                    # Convertir lista de puntos a arreglo NumPy (float32)
    cx, cy = pts.mean(axis=0)                                   # Calcular coordenadas centroide (promedio X, Y)
    r = max(2.0, np.linalg.norm(pts - [cx, cy], axis=1).mean()) # Calcular radio promedio desde centroide; mínimo 2.0 px
    return float(cx), float(cy), float(r)                       # Retornar centro (cx, cy) y radio (r) como floats


# -------- Ajustar bounding box a los límites de la imagen --------
# Recortar coordenadas y tamaño de bounding box para que no salga del área válida
def _clip_bbox(x, y, w, h, W, H):
    x = max(0, min(x, W - 1)) # Limitar X para que esté dentro de la imagen (0 a W-1)
    y = max(0, min(y, H - 1)) # Limitar Y para que esté dentro de la imagen (0 a H-1)
    w = max(1, min(w, W - x)) # Ajustar ancho: al menos 1 px y sin salir del límite derecho
    h = max(1, min(h, H - y)) # Ajustar alto: al menos 1 px y sin salir del límite inferior
    return int(x), int(y), int(w), int(h) # Devolver coordenadas y tamaño como enteros


# -------- Convertir a escala de grises y ecualizar --------
# Convertir imagen BGR a gris y aplicar ecualización adaptativa (CLAHE) para mejorar contraste

def _gray_eq(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)               # Convertir imagen de BGR a escala de grises
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Crear objeto CLAHE (Contraste Limitado Adaptativo)
    return clahe.apply(g)                                       # Aplicar CLAHE y devolver imagen mejorada


# ----------------- Detección automática de ojo -----------------
def _auto_roi_mediapipe(cap, max_busqueda=90, prefer="auto"):
    # -------- Detectar ROI automáticamente con MediaPipe --------
    # Buscar y fijar un ROI de ojo/iris en primeros frames
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtener ancho video en píxeles
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Obtener alto video en píxeles
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)     # Guardar posición inicial de frame para restaurar después
    side_fijo = None                            # Variable para almacenar lado fijo a seguir ("left" o "right")
    bbox_final = None                           # Variable para almacenar bounding box definitivo del ojo


    # -------- Inicializar modelo Face Mesh de MediaPipe --------
    # Configurar y crear detector de malla facial para seguimiento de un rostro
    with mp_face.FaceMesh(
            static_image_mode=False,      # Procesar como video (detección + seguimiento en frames consecutivos)
            max_num_faces=1,              # Detectar máximo 1 rostro
            refine_landmarks=True,        # Activar puntos adicionales para iris y detalles de ojos
            min_detection_confidence=0.5, # Umbral mínimo de confianza para detección inicial
            min_tracking_confidence=0.5   # Umbral mínimo de confianza para seguimiento posterior
        ) as face_mesh:                   # Crear y usar contexto del detector face_mesh
       # -------- Iterar sobre frames iniciales para buscar ROI --------
        for _ in range(max_busqueda):  # Repetir hasta 'max_busqueda' intentos para encontrar ojo válido
            # -------- Leer frame y procesar con Face Mesh --------
            ok, frame = cap.read()                       # Leer un frame del video
            if not ok: break                             # Si no se puede, salir del bucle
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir imagen de BGR (OpenCV) a RGB (MediaPipe)
            res = face_mesh.process(rgb)                 # Procesar frame con Face Mesh para detectar landmarks faciales
            if not res.multi_face_landmarks: continue    # Si no se detecta rostro, pasar al siguiente frame
            lm = res.multi_face_landmarks[0].landmark    # Obtener la lista de landmarks (puntos faciales) del primer rostro

            # -------- Obtener coordenadas absolutas --------
            def pts(indices): # Convertir índices de landmarks en coordenadas (x, y) en píxeles
                return [(lm[i].x * W, lm[i].y * H) for i in indices] # Escalar coordenadas normalizadas a tamaño real de imagen

            # -------- Obtener coordenadas de ojos e iris --------
            left_eye  = pts(LEFT_EYE_IDX);  right_eye  = pts(RIGHT_EYE_IDX) # Coordenadas de contorno ojo
            left_iris = pts(LEFT_IRIS_IDX); right_iris = pts(RIGHT_IRIS_IDX) # Coordenadas de iris

            # -------- Calcular área de un polígono --------
            def area(poly): # Áreas aproximadas (para saber si "existe" en cuadro)
                p = np.array(poly, dtype=np.float32) # Convertir lista de puntos a arreglo NumPy (float32)
                x = p[:,0]; y = p[:,1] # Separar coordenadas X e Y
                return float(abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1))) * 0.5) # Calcular área usando fórmula del polígono (shoelace)

            area_L = area(left_eye); area_R = area(right_eye) # Calcular área de cada ojo

            # -------- Fijar ojo a seguir --------
            if side_fijo is None:
                if prefer in ("left","right"): # Si se indicó preferencia de lado
                    side_fijo = prefer if ((prefer=="left" and area_L>0) or (prefer=="right" and area_R>0)) else None
                if side_fijo is None: # Si no hay preferencia o no fue posible usarla
                    if area_L>0 and area_R>0:
                        # Toma el "más grande" (más visible) o izquierdo por consistencia
                        side_fijo = "left" if area_L >= area_R else "right"
                    elif area_L>0:
                        side_fijo = "left"
                    elif area_R>0:
                        side_fijo = "right"
                    else:
                        continue  # Aún no se detecta ojo válido

            # -------- Calcular ROI final según ojo elegido --------
            # Construye bbox alrededor del ojo elegido (centrado en iris, margen ~2.5r)
            if side_fijo == "left" and area_L>0 and len(left_iris)==4: # Si se eligió izquierdo y es válido
                cx, cy, r = _center_radius(left_iris)    # Calcular centro y radio de iris
                m = 2.5                                  # Factor de ampliación para bounding box
                x1 = int(cx - m*r); y1 = int(cy - m*r)   # Calcular esquina superior izquierda de ROI
                w  = int(2*m*r);    h  = int(2*m*r)      # Calcular ancho y alto de ROI
                x1,y1,w,h = _clip_bbox(x1,y1,w,h,W,H)    # Ajustar ROI a límites de la imagen
                bbox_final = (x1,y1,w,h)                 # Guardar bounding box final
                break                                    # Salir del bucle (ROI encontrado)
            elif side_fijo == "right" and area_R>0 and len(right_iris)==4: # Si se eligió ojo derecho y es válido
                cx, cy, r = _center_radius(right_iris)   # Calcular centro y radio de iris
                m = 2.5                                  # Factor de ampliación para bounding box
                x1 = int(cx - m*r); y1 = int(cy - m*r)   # Calcular esquina superior izquierda de ROI
                w  = int(2*m*r);    h  = int(2*m*r)      # Calcular ancho y alto de ROI
                x1,y1,w,h = _clip_bbox(x1,y1,w,h,W,H)    # Ajustar ROI a límites de la imagen
                bbox_final = (x1,y1,w,h)                 # Guardar bounding box final
                break                                    # Salir del bucle (ROI encontrado)

    # -------- Restaurar posición y devolver resultado --------
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0) # Reiniciar video a posición inicial
    return bbox_final, (side_fijo or "unknown") # Retornar bounding box final y lado elegido (o "unknown" si no se fijó)


# -------- Detectar ROI automáticamente con Haar Cascades --------
# Fallback sin MediaPipe: detectar rostro, luego ojos, y devolver el ojo más grande encontrado
def _auto_roi_haar(cap, max_busqueda=120):
    # Fallback sin MediaPipe: cara->ojos con Haar. Devuelve bbox del ojo más grande.
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtener ancho video
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Obtener alto video
    pos0 = cap.get(cv2.CAP_PROP_POS_FRAMES)     # Guardar posición actual frame
    mejor = None; mejor_area = 0                # Inicializar mejor ROI y su área

    for _ in range(max_busqueda):                    # Revisar hasta 'max_busqueda' frames
        ok, frame = cap.read()                       # Leer frame video
        if not ok: break                             # Si no se pudo leer, salir del bucle
        g = _gray_eq(frame)                          # Convertir a gris y ecualizar contraste
        faces = _CASCADE_FACE.detectMultiScale(g, 1.2, 5) # Detectar rostros
        for (fx,fy,fw,fh) in faces:                  # Iterar sobre rostros detectados
            roi_face = g[fy:fy+fh, fx:fx+fw]         # Extraer región de rostro
            eyes = _CASCADE_EYE.detectMultiScale(roi_face, 1.15, 5) # Detectar ojos en el rostro
            for (ex,ey,ew,eh) in eyes:               # Iterar sobre ojos detectados
                x = fx+ex; y = fy+ey; w = ew; h = eh # Calcular posición del ojo en coordenadas de imagen completa
                # margen
                x,y,w,h = _clip_bbox(int(x-0.2*w), int(y-0.2*h), int(1.4*w), int(1.4*h), W,H) # Aplicar margen y recortar a límites
                area = w*h                          # Calcular área ROI
                if area > mejor_area:               # Si este ojo es más grande que el mejor encontrado
                    mejor_area = area; mejor = (x,y,w,h) # Guardar como nuevo mejor ROI
        if mejor is not None:                       # Si ya hay un ojo detectado
            break                                   # Salir del bucle

    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0) # Restaurar posición inicial video
    return mejor, "unknown"                # Retornar ROI encontrado y lado desconocido


# -------- Seleccionar método de detección de ROI --------
def _auto_roi(cap, prefer="auto"): # Usar MediaPipe si está disponible, si no, Haar Cascades
    if _USA_MEDIAPIPE:                                      # Si MediaPipe está habilitado
        bbox, side = _auto_roi_mediapipe(cap, prefer=prefer) # Detectar ROI con MediaPipe
        if bbox is not None: return bbox, side               # Si ROI válido, lo devuelve
    # Fallback
    return _auto_roi_haar(cap) # Si falla o no hay MediaPipe, usa Haar Cascades


# -------- Estabilización por traslación: coincidencia de plantilla --------
# Encontrar ROI dentro del frame actual buscando la mejor coincidencia
def _match_roi(gray, templ): # Buscar en imagen gris la posición más parecida a la plantilla dada
    res = cv2.matchTemplate(gray, templ, cv2.TM_CCOEFF_NORMED) # Comparar plantilla con toda la imagen usando correlación normalizada
    _, maxVal, _, maxLoc = cv2.minMaxLoc(res)                  # Obtener valor de coincidencia máximo y su posición
    return maxLoc, float(maxVal)                               # Retornar coordenada superior izq. de la mejor coincidencia y su valor de similitud


# -------- Corregir movimiento (estabilizar ROI) --------
def corregir_movimiento(ruta_video, salida=None, prefer_side="auto", debug=True):
    # Abrir video, leer metadatos y preparar parámetros básicos
    import time                                           # Importar time para medición/logs puntuales
    cap = cv2.VideoCapture(ruta_video)                    # Cargar video
    if not cap.isOpened():                                # Verificar que video abrió correctamente
        print("No se pudo abrir el vídeo."); return None  # Salir si falla

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0               # Obtener FPS; usar 30.0 si no está disponible
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))          # Ancho video en píxeles
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))         # Alto video en píxeles

    # -------- Detectar ROI inicial automáticamente --------
    bbox, side = _auto_roi(cap, prefer=prefer_side)              # Detectar ROI (ojo) usando MediaPipe o Haar
    if bbox is None:                                             # Si no se detecta
        print("No se pudo detectar un ojo de forma automática.")
        cap.release()                                            # Liberar recurso de video
        return None                                              # Salir sin continuar
    x0, y0, w0, h0 = bbox                                        # Guardar coordenadas y tamaño inicial del ROI

    # -------- Leer primer frame para iniciar estabilización --------
    ok, frame0 = cap.read()                       # Leer primer frame del video
    if not ok:                                    # Si no se puede
        print("No se pudo leer el primer frame.")
        cap.release()                             # Liberar recurso de video
        return None                               # Salir sin continuar

    # -------- Preparar salida del video estabilizado --------
    if salida is None:                                        # Si no se indicó ruta de salida
        base, _ = os.path.splitext(ruta_video)                # Separar nombre base y extensión de archivo original
        salida = f"{base}_estabilizado.mp4"                           # Crear nombre de salida con sufijo
    writer = cv2.VideoWriter(salida,                          # Crear objeto para escribir video de salida
                             cv2.VideoWriter_fourcc(*"mp4v"), # Codec MP4V
                             fps,
                             (W, H))                          # Resolución igual a video original

    # -------- Inicializar plantilla y referencias para seguimiento --------
    gray0  = _gray_eq(frame0)                 # Convertir primer frame a gris y ecualizar
    templ  = gray0[y0:y0+h0, x0:x0+w0].copy() # Extraer ROI inicial como plantilla para coincidencia
    ref_tl = (x0, y0)                         # Guardar coordenada superior izquierda inicial del ROI
    last_tl = ref_tl                          # Inicializar última posición de ROI
    writer.write(frame0)                      # Guardar primer frame sin modificar en video de salida

    # -------- VISOR EN VIVO --------
    # -------- Configuración de visualización y control --------
    show_side_by_side = False # tecla S -> Mostrar original y estabilizado lado a lado
    show_gray = False         # tecla G -> Mostrar video en escala de grises
    show_template = True      # tecla T -> Mostrar plantilla (ROI) usada para seguimiento
    paused = False            # Pausar/reanudar reproducción con tecla (generalmente barra espaciadora)
    
    # -------- Configurar ventana depuración --------
    if debug:
        cv2.namedWindow("Estabilizado", cv2.WINDOW_NORMAL)            # Crear ventana ajustable para mostrar resultado
        cv2.resizeWindow("Estabilizado", min(1200, 2*W), min(800, H)) # Ajustar tamaño de ventana según resolución
        cv2.moveWindow("Estabilizado", 60, 60)                        # Mover ventana a posición (60,60) en pantalla

    # -------- Inicializar parámetros seguimiento --------
    THRESH_BAD = 0.45 # Umbral mínimo de coincidencia aceptable para considerar tracking válido
    t0 = time.time()  # Tiempo de inicio del proceso (para medir rendimiento)
    n_frames = 0      # Contador de frames procesados

    # -------- Bucle principal de procesamiento de frames --------
    while True:  # Procesar cada frame de video hasta que termine o se interrumpa
        # -------- Leer siguiente frame si no está en pausa --------
        if not paused:
            ok, frame = cap.read() # Leer siguiente frame de video
            if not ok: break       # Salir si no hay más
            n_frames += 1          # Incrementar contador de frames procesados

            # -------- Buscar ROI en el frame actual --------
            gray = _gray_eq(frame)              # Convertir frame a gris y ecualizar
            tl, score = _match_roi(gray, templ) # Buscar plantilla en frame y obtener posición (tl) y puntaje (score)
            if score < THRESH_BAD:              # Si puntaje menor a umbral (mala coincidencia)
                cx, cy = last_tl                # Usar última posición válida
            else:            # Si coincidencia es buena
                cx, cy = tl  # Actualizar posición actual del ROI
                last_tl = tl # Guardar como última posición válida

            # -------- Calcular y aplicar desplazamiento para estabilizar --------
            dx = ref_tl[0] - cx # Desplazamiento X para alinear ROI
            dy = ref_tl[1] - cy # Desplazamiento Y para alinear ROI
            M = np.float32([[1, 0, dx], [0, 1, dy]]) # Matriz de transformación afín (traslación)
            estabilizado = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, # Aplicar traslación al frame
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)) # Rellenar bordes con negro

            # -------- Dibujar elementos de depuración en frame --------
            draw_base = cv2.cvtColor(gray if show_gray else (estabilizado if not show_side_by_side else frame), cv2.COLOR_GRAY2BGR) if show_gray else (estabilizado if not show_side_by_side else frame) 
            # Seleccionar base de dibujo: gris, estabilizado u original, según configuración
            vis = draw_base.copy() # Copiar imagen base para dibujar encima
            # ROI fijo
            cv2.rectangle(vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2) # Dibujar rectángulo verde en ROI de referencia
            # Texto
            elapsed = max(1e-3, time.time() - t0) # Tiempo transcurrido desde inicio
            fps_rt = n_frames / elapsed           # FPS promedio en tiempo real
            cv2.putText(vis, f"score:{score:.2f}  side:{side}  FPS:{fps_rt:.1f}", (20, 30), # Mostrar score, lado y FPS
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 

            # -------- Mostrar plantilla como miniatura --------
            if show_template:                                             # Si está habilitada la visualización de la plantilla
                th, tw = templ.shape                                      # Alto y ancho de la plantilla original
                scale = 120.0 / max(th, tw)                               # Escala para que el lado mayor sea de 120 px
                tmini = cv2.resize(templ, (int(tw * scale), int(th * scale))) # Redimensionar plantilla
                tmini = cv2.cvtColor(tmini, cv2.COLOR_GRAY2BGR)           # Convertir a BGR para mostrar en color
                hM, wM, _ = tmini.shape                                   # Dimensiones de la miniatura
                vis[10:10 + hM, 10:10 + wM] = tmini                       # Colocar miniatura en esquina superior izquierda

            # -------- Mostrar comparación lado a lado --------
            if show_side_by_side:
                # Columna izquierda: original con bbox en posición detectada (coincidencia actual)
                left = frame.copy()
                cv2.rectangle(left, (cx, cy), (cx + w0, cy + h0), (0, 0, 255), 2) # Caja roja en ROI detectado

                right = vis                                                       # Columna derecha: imagen con estabilización y depuración
                canvas = np.zeros((H, W * 2, 3), dtype=np.uint8)                  # Lienzo vacío para juntar ambas columnas
                canvas[:, :W] = left                                              # Poner original a la izquierda
                canvas[:, W:] = right                                             # Poner estabilizado a la derecha
                out_show = canvas                                                 # Definir imagen final a mostrar
            else:
                out_show = vis # Si no es lado a lado, mostrar solo imagen depurada

            # -------- Guardar frame estabilizado --------
            writer.write(estabilizado) # Escribir frame estabilizado en archivo de salida
        else:
            out_show = vis # Mantener última imagen mostrada si está en pausa

        # -------- Controles de depuración en tiempo real --------
        if debug:
            cv2.imshow("Estabilizado", out_show)            # Mostrar frame en ventana de depuración
            k = cv2.waitKey(1) & 0xFF                       # Esperar tecla (1 ms) y obtener código
            if k == 27:  # ESC                              # Si se presiona ESC
                break
            elif k == ord(' '):  # pausa                    # Barra espaciadora -> Pausar/reanudar
                paused = not paused
            elif k == ord('s'):                             # Tecla S -> Alternar vista lado a lado
                show_side_by_side = not show_side_by_side
            elif k == ord('g'):                             # Tecla G -> Alternar escala de grises
                show_gray = not show_gray
            elif k == ord('t'):                             # Tecla T -> Alternar miniatura de plantilla
                show_template = not show_template

    # -------- Finalizar y liberar recursos --------
    cap.release(); writer.release()            # Cerrar video entrada y archivo salida
    if debug: cv2.destroyAllWindows()          # Cerrar ventanas si está en modo depuración
    print(f"Estabilización lista -> {salida}") # Mensaje con ruta video estabilizado
    return salida                              # Devolver ruta video de salida


if __name__ == "__main__":
    # Ejemplo:
    ruta = "ruta/a/tu/video.mp4"
    corregir_movimiento(ruta, prefer_side="auto", debug=True)
