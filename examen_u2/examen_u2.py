import cv2
import numpy as np
import os


# PREPROCESAMIENTO

def preprocesar_imagen(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    return morph


# -------------------------------
# DETECCIÓN DE DADOS
# -------------------------------
def detectar_dados(binaria, area_min=3000, area_max=15000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binaria)

    dados = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        aspect_ratio = w / h

        if (area_min < area < area_max and
            0.4 < aspect_ratio < 1.3 and
            w > 20 and h > 20):

            dados.append((x, y, w, h))

    return dados


# CONTEO DE PUNTOS

def contar_puntos(dado_img):
    gray = cv2.cvtColor(dado_img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    puntos = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)

        if perimetro == 0:
            continue

        circularidad = 4 * np.pi * area / (perimetro * perimetro)

        if 25 < area < 300 and 0.6 < circularidad < 1.2:
            puntos += 1

    return puntos



# LEER ANNOTATIONS (YOLO)

def leer_valores_reales(ruta_label):
    valores = []

    with open(ruta_label, "r") as f:
        for linea in f.readlines():
            partes = linea.strip().split()

            if len(partes) > 0:
                valor = int(partes[0])  # clase
                valores.append(valor)

    #  ajuste automático (si usa clases 0–5)
    if len(valores) > 0 and min(valores) == 0:
        valores = [v + 1 for v in valores]

    return valores



# PROCESAR CARPETA

def procesar_carpeta(carpeta_img, carpeta_labels=None):

    error_dados = []
    error_valores = []

    for archivo in os.listdir(carpeta_img):
        if archivo.endswith(".png") or archivo.endswith(".jpg"):

            print("\n==============================")
            print("Imagen:", archivo)

            ruta = os.path.join(carpeta_img, archivo)
            img = cv2.imread(ruta)

            if img is None:
                print("Error cargando imagen")
                continue

            binaria = preprocesar_imagen(img)
            dados = detectar_dados(binaria)

            valores_detectados = []

            for (x, y, w, h) in dados:
                dado_roi = img[y:y+h, x:x+w]
                valor = contar_puntos(dado_roi)
                valores_detectados.append(valor)

            print("Dados detectados:", len(dados))
            print("Valores detectados:", valores_detectados)

            
            # VALORES REALES
          
            if carpeta_labels:
                nombre_txt = os.path.splitext(archivo)[0] + ".txt"
                ruta_label = os.path.join(carpeta_labels, nombre_txt)

                if os.path.exists(ruta_label):
                    valores_reales = leer_valores_reales(ruta_label)

                    print("Dados reales:", len(valores_reales))
                    print("Valores reales:", valores_reales)

                    # ERROR DADOS
                    e_d = abs(len(dados) - len(valores_reales))
                    error_dados.append(e_d)

                    # ERROR VALORES
                    vd = sorted(valores_detectados)
                    vr = sorted(valores_reales)

                    n = min(len(vd), len(vr))

                    if n > 0:
                        diff = sum(abs(vd[i] - vr[i]) for i in range(n)) / n
                        error_valores.append(diff)

    
    # RESULTADOS FINALES
   
    print("\n==============================")
    print("RESULTADOS FINALES")

    if error_dados:
        print("Error medio (dados):", np.mean(error_dados))
    else:
        print("Error medio (dados): No calculado")

    if error_valores:
        print("Error medio (valores):", np.mean(error_valores))
    else:
        print("Error medio (valores): No calculado")



# EJECUCIÓN

carpeta_imagenes = "imgexamenu2"
carpeta_annotations = "Annotations"

procesar_carpeta(carpeta_imagenes, carpeta_annotations)