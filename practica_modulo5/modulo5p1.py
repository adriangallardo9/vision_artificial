import numpy as np
import matplotlib.pyplot as plt

# Definir un filtro lineal (ejemplo: filtro promedio 3x3)
filtro = gaussiano = np.array ([
    [1,4,6,4,1],
    [4,16,24,16,4],
    [6,24,36,24,6],
    [4,16,24,16,4],
    [1,4,6,4,1]
    ]) * (1/256) 


# Tamaño de la imagen donde se analizará el filtro
size = 256

# Crear una matriz del tamaño de la imagen
H = np.zeros((size,size))

# Insertar el filtro en la esquina superior
H[:filtro.shape[0], :filtro.shape[1]] = filtro

# Calcular la Transformada de Fourier
F = np.fft.fft2(H)

# Centrar frecuencias
F_shift = np.fft.fftshift(F)

# Magnitud
magnitud = np.log(1 + np.abs(F_shift))

# Mostrar resultados
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Filtro espacial")
plt.imshow(filtro, cmap='gray')
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Transformada de Fourier del filtro")
plt.imshow(magnitud, cmap='gray')
plt.colorbar()

plt.show()