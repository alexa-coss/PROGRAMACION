import cv2, numpy as np, ctypes

def show_letterboxed(frame, win='Vista', screen_scale=0.92):
    # Obtener resolución de pantalla (Windows)
    sw = ctypes.windll.user32.GetSystemMetrics(0)
    sh = ctypes.windll.user32.GetSystemMetrics(1)
    max_w, max_h = int(sw*screen_scale), int(sh*screen_scale)

    h, w = frame.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)  # no agrandar más del 100%
    new_w, new_h = int(w*s), int(h*s)

    resized = cv2.resize(frame, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)

    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)  # fondo negro
    y = (max_h - new_h) // 2
    x = (max_w - new_w) // 2
    canvas[y:y+new_h, x:x+new_w] = resized

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, canvas)
