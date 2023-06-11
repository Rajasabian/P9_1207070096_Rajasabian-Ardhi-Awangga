# Memanggil Library yang dibutuhkan
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Memanggil gambar yang disimpan dan mengubah citra menjadi grayscale (0)
img = cv.imread('pragos.png', 0)

img_canny = cv.Canny(img,100,200)

#Plot Ouput
fig, axes = plt.subplots(2, 2, figsize=(8, 8))#untuk membuat sebuah figure (gambar) yang terdiri dari matriks subplot 4x2 dengan ukuran (lebar, tinggi) sebesar (8, 8) inci.
ax = axes.ravel()#untuk mengubah matriks axes menjadi array satu dimensi dengan menggunakan metode ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny, cmap='gray')
ax[2].set_title("Citra Output canny")
ax[3].hist(img_canny.ravel(), bins=256)
ax[3].set_title("Histogram Citra Output canny")

fig.tight_layout()# untuk secara otomatis mengatur jarak antara subplot dalam sebuah gambar agar lebih rapi dan tidak tumpang tindih.
plt.show()# Menampilkan Output