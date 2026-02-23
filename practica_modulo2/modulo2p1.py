import cv2
import numpy as np
from transformaciones import TransformacionesEuclideanas, TransformacionesAfines


ruta = "img/figura3.1.png"
imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
h,w = imagen.shape

centro_y, centro_x = int(np.floor(h/2)), int(np.floor(w/2))
T1 = TransformacionesEuclideanas.traslado(-centro_x, -centro_y)
T2 = TransformacionesEuclideanas.traslado(centro_x, centro_y)

RN1 = TransformacionesEuclideanas.rotacion(0,1)
TN1 = TransformacionesEuclideanas.traslado(0, 0)
CN1 = TransformacionesAfines.cizallamiento_horizontal(2.9)

T = T2 @ TN1 @ CN1 @ RN1 @ T1

imagen_procesada = TransformacionesAfines.transformacion_opencv(imagen, T[:2,:], (w,h))

print("\nMatriz de transformaciones aplicadas:")
print(T)

cv2.imshow("imagen_procesada",imagen_procesada)
cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.waitKey(1)