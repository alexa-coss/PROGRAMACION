import cv2, numpy as np, os  # Importa OpenCV y NumPy  # os para rutas
from video.roi import detectar_ojos_e_iris_multi, reset_roi_state  # usar ROI modular

# ----------------------------------------------------------------------
# Utilidades internas (plantillas, búsqueda local y warps)
# ----------------------------------------------------------------------

def _extract_center_from_entry(entry):
    """Devuelve (cx, cy) flotantes del dict de ojo."""
    if entry is None: 
        return None
    cx, cy = entry["center"]
    return float(cx), float(cy)

def _get_mode_centers(multi):
    """A partir de la salida de detectar_ojos_e_iris_multi arma centros por lado y promedio."""
    L = multi.get("left"); R = multi.get("right")
    cL = _extract_center_from_entry(L)
    cR = _extract_center_from_entry(R)
    centers = {"L": cL, "R": cR}
    # Promedio si existen ambos
    if cL is not None and cR is not None:
        centers["AVG"] = ((cL[0] + cR[0]) * 0.5, (cL[1] + cR[1]) * 0.5)
    else:
        centers["AVG"] = None
    return centers, L, R

def _safe_int_rect(x, y, w, h, W, H):
    """Ajusta rect para que no salga de la imagen y devuelve (x,y,w,h) ints."""
    x1 = max(0, int(x)); y1 = max(0, int(y))
    x2 = min(W - 1, int(x + w)); y2 = min(H - 1, int(y + h))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)

def _crop(img, rect):
    """Recorta img con rect (x,y,w,h)."""
    x, y, w, h = rect
    return img[y:y+h, x:x+w].copy()

def _translate(frame, dx, dy):
    """Aplica traslación (dx,dy) al frame con warpAffine."""
    H, W = frame.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ----------------------------------------------------------------------
# Estabilización por plantillas (un ROI o dos)
# ----------------------------------------------------------------------

def _init_templates(first_frame, L, R):
    """
    Crea plantillas de búsqueda basadas en los bbox del iris para seguimiento por correlación.
    - Si hay ambos ojos: devuelve plantillas para L y R.
    - Si hay solo uno: devuelve solo esa.
    """
    H, W = first_frame.shape[:2]
    templates = {}
    if L:
        tL = _crop(first_frame, L["iris_bbox"])  # plantilla del iris izq
        if tL.size > 0:
            templates["L"] = (tL, L["iris_bbox"])
    if R:
        tR = _crop(first_frame, R["iris_bbox"])  # plantilla del iris der
        if tR.size > 0:
            templates["R"] = (tR, R["iris_bbox"])
    return templates  # dict: side -> (template_img, initial_bbox)

def _match_near(frame, tpl, prev_bbox, search_pad=24):
    """
    Busca la plantilla cerca de la posición previa usando matchTemplate.
    - frame: imagen BGR actual
    - tpl: imagen plantilla (BGR o GRAY)
    - prev_bbox: (x,y,w,h) posición anterior de la plantilla
    - search_pad: margen alrededor para búsqueda local
    Devuelve: (best_x, best_y, score)
    """
    H, W = frame.shape[:2]
    x, y, w, h = prev_bbox
    # Región de búsqueda ampliada alrededor de la posición previa
    sx = max(0, x - search_pad); sy = max(0, y - search_pad)
    ex = min(W, x + w + search_pad); ey = min(H, y + h + search_pad)
    search = frame[sy:ey, sx:ex]
    if search.size == 0 or tpl.size == 0:
        return (x, y, -1.0)

    # Convertir a GRAY para robustez
    sG = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    tG = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if (len(tpl.shape) == 3 and tpl.shape[2] == 3) else tpl
    if sG.shape[0] < tG.shape[0] or sG.shape[1] < tG.shape[1]:
        return (x, y, -1.0)

    res = cv2.matchTemplate(sG, tG, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    best_x = sx + max_loc[0]
    best_y = sy + max_loc[1]
    return (best_x, best_y, float(max_val))

# ----------------------------------------------------------------------
# API PRINCIPAL
# ----------------------------------------------------------------------

def corregir_movimiento(ruta_video, prefer_side="auto", ojos="auto", debug=False):
    """
    Corrige el movimiento del vídeo usando seguimiento del iris/ojos (ROI) con plantillas.
    - prefer_side: "auto" | "left" | "right"   → preferencia cuando hay un solo ojo usable.
    - ojos: "auto" | "both" | "mono"           → forzar modo o dejar que decida según detección.
    - debug: True para ventana con visualización.
    Devuelve: (ruta_salida, lado_usado)
      - lado_usado: "both" si estabilizó con dos ojos, "left"/"right" si con uno, o None si no pudo.
    """
    # Abrir vídeo
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        print("No se pudo leer el vídeo.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # fps del video
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, first = cap.read()
    if not ok:
        print("No se pudo leer el vídeo.")
        cap.release()
        return None, None

    # Detectar ojos/iris en primer frame
    multi = detectar_ojos_e_iris_multi(first, lado_fijo=None)  # detección multi-ojos
    centers, L0, R0 = _get_mode_centers(multi)  # extraer centros iniciales

    # Decidir si estabilizar con ambos o uno
    both_available = (centers["L"] is not None) and (centers["R"] is not None)
    use_both = False
    usado = None      # etiqueta de lado usado

    if ojos == "both" and both_available:
        use_both = True
        usado = "both"
    elif ojos == "mono":
        use_both = False
    else:
        # "auto": si hay ambos, usar ambos; si no, uno.
        use_both = both_available

    # Target(s) inicial(es) para estabilizar
    if use_both:
        tgt = centers["AVG"] # promedio de ambos
        usado = "both"
    else:
        # Elegir lado: preferencia del usuario, o automático por disponibilidad/área
        if centers["L"] is not None and centers["R"] is not None:
            # Elige por preferencia si existe, de lo contrario por área (bbox del ojo)
            if prefer_side in ("left", "right"):
                usado = prefer_side
            else:
                # área aprox con eye_bbox inicial si existe, si no, por radio
                areaL = (L0["eye_bbox"][2] * L0["eye_bbox"][3]) if L0 else 0
                areaR = (R0["eye_bbox"][2] * R0["eye_bbox"][3]) if R0 else 0
                usado = "left" if areaL >= areaR else "right"
        elif centers["L"] is not None:
            usado = "left"
        elif centers["R"] is not None:
            usado = "right"
        else:
            # Nada visible → no se puede estabilizar
            cap.release()
            print("No se detectó ojo en el primer frame.")
            return None, None
        tgt = centers["L"] if usado == "left" else centers["R"]

    # Crear escritor de salida
    base, ext = os.path.splitext(ruta_video)
    salida = f"{base}_estabilizado{ext}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(salida, fourcc, fps, (W, H))

    # Inicializar plantillas
    templates = _init_templates(first, L0, R0)

    # Estados previos (bboxes de plantilla)
    prev_bbox_L = L0["iris_bbox"] if L0 else None
    prev_bbox_R = R0["iris_bbox"] if R0 else None

    # Frame inicial ya está alineado por definición
    writer.write(first)

    # Debug UI
    if debug:
        cv2.namedWindow("Estabilizado", cv2.WINDOW_NORMAL)

    # Procesar frames restantes
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Por robustez, intentar actualizar ROIs con FaceMesh de vez en cuando (cada N frames sería ideal).
        # Aquí, para simplicidad, re-detectamos ligero cada frame (si es costoso, puedes espaciar).
        multi = detectar_ojos_e_iris_multi(frame, lado_fijo=None)
        centers, L, R = _get_mode_centers(multi)

        # Si usamos ambos: necesitamos dos centros; si falla uno, caer al otro temporalmente
        if use_both:
            cL = centers["L"]; cR = centers["R"]
            if cL is None and cR is None:
                # No hay ojos detectados → intentar seguir por template matching con posición previa
                # Haremos tracking de plantillas L/R si existen, y calcularemos centro a partir del match
                cL = None; cR = None
                if "L" in templates and prev_bbox_L is not None:
                    xL, yL, sL = _match_near(frame, templates["L"][0], prev_bbox_L, search_pad=24)
                    if sL > 0.25:
                        cL = (xL + templates["L"][0].shape[1] * 0.5, yL + templates["L"][0].shape[0] * 0.5)
                        prev_bbox_L = (xL, yL, templates["L"][0].shape[1], templates["L"][0].shape[0])
                if "R" in templates and prev_bbox_R is not None:
                    xR, yR, sR = _match_near(frame, templates["R"][0], prev_bbox_R, search_pad=24)
                    if sR > 0.25:
                        cR = (xR + templates["R"][0].shape[1] * 0.5, yR + templates["R"][0].shape[0] * 0.5)
                        prev_bbox_R = (xR, yR, templates["R"][0].shape[1], templates["R"][0].shape[0])

            # Centro objetivo (promedio si ambos; si hay uno, usar ese)
            if cL is not None and cR is not None:
                cur_center = ((cL[0] + cR[0]) * 0.5, (cL[1] + cR[1]) * 0.5)
            elif cL is not None:
                cur_center = cL
            elif cR is not None:
                cur_center = cR
            else:
                # Nada que estabilizar en este frame → escribir tal cual
                writer.write(frame)
                if debug:
                    cv2.imshow("Estabilizado", frame)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                continue

            # Desplazamiento necesario para llevar cur_center a tgt (centro de referencia inicial)
            dx = tgt[0] - cur_center[0]
            dy = tgt[1] - cur_center[1]
            stab = _translate(frame, dx, dy)
            writer.write(stab)

            if debug:
                vis = stab.copy()
                cv2.circle(vis, (int(tgt[0]), int(tgt[1])), 4, (0, 255, 0), -1)  # target
                cv2.imshow("Estabilizado", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

        else:
            # Monocular (o auto pero solo hay uno)
            # Determinar centro actual del lado elegido; si falta, intentar por plantilla
            c = None
            if usado == "left":
                c = centers["L"]
                if c is None and "L" in templates and prev_bbox_L is not None:
                    x, y, s = _match_near(frame, templates["L"][0], prev_bbox_L, search_pad=24)
                    if s > 0.25:
                        c = (x + templates["L"][0].shape[1] * 0.5, y + templates["L"][0].shape[0] * 0.5)
                        prev_bbox_L = (x, y, templates["L"][0].shape[1], templates["L"][0].shape[0])
            else:  # usado == "right"
                c = centers["R"]
                if c is None and "R" in templates and prev_bbox_R is not None:
                    x, y, s = _match_near(frame, templates["R"][0], prev_bbox_R, search_pad=24)
                    if s > 0.25:
                        c = (x + templates["R"][0].shape[1] * 0.5, y + templates["R"][0].shape[0] * 0.5)
                        prev_bbox_R = (x, y, templates["R"][0].shape[1], templates["R"][0].shape[0])

            if c is None:
                # No hay centro detectable → escribir tal cual
                writer.write(frame)
                if debug:
                    cv2.imshow("Estabilizado", frame)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                continue

            dx = tgt[0] - c[0]
            dy = tgt[1] - c[1]
            stab = _translate(frame, dx, dy)
            writer.write(stab)

            if debug:
                vis = stab.copy()
                cv2.circle(vis, (int(tgt[0]), int(tgt[1])), 4, (0, 255, 0), -1)
                cv2.imshow("Estabilizado", vis)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

    # Cierre
    cap.release()
    writer.release()
    if debug:
        cv2.destroyAllWindows()

    print(f"Estabilización lista -> {salida}")
    return salida, usado

# ----------------------------------------------------------------------
# Helper de limpieza de estado ROI
# ----------------------------------------------------------------------

def reset_corregir_movimiento_state():
    """Reinicia el estado interno de ROI para una nueva corrida."""
    reset_roi_state()  # Delegar al módulo roi
