import cv2
import numpy as np



def detectar_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Ubicación (x, y): ({x}, {y})")
        param["puntos"].append([x, y])
        param["clicks"] += 1


def obtener_puntos(imagen, nombre_ventana):
    params = {
        "puntos": [],
        "clicks": 0
    }


    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(nombre_ventana, detectar_click, param=params)


    print(f"Ubica 3 puntos en {nombre_ventana} haz click sobre ellos y presiona 'q'")


    while True:
        cv2.imshow(nombre_ventana, imagen)
        if cv2.waitKey(1) & 0xFF == ord('q') or params["clicks"] >= 3:
            break


    cv2.destroyWindow(nombre_ventana)
    cv2.waitKey(1)


    return np.array(params["puntos"], dtype=np.float32)


imagen_original = cv2.imread("img/cameraman.png")
imagen_procesada = cv2.imread("img/cameraman_processed.png")


h, w = imagen_original.shape[:2]


puntos_origen = obtener_puntos(imagen_original, "Imagen cameraman original")
puntos_destino = obtener_puntos(imagen_procesada, "Imagen cameraman procesada")


M = cv2.getAffineTransform(puntos_origen, puntos_destino)


print("\nMatriz de transformación afín estimada:")
print(M)


imagen_estimada = cv2.warpAffine(imagen_original, M, (w, h))


cv2.imshow("imagen cameraman original", imagen_original)
cv2.imshow("imagen cameraman procesada", imagen_procesada)
cv2.imshow("imagen cameraman estimada", imagen_estimada)


cv2.waitKey(0)
cv2.destroyAllWindows()