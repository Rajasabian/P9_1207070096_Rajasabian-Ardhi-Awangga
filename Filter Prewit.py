# Memanggil Library yang dibutuhkan
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Memanggil gambar yang disimpan dan mengubah citra menjadi grayscale (0)
img = cv.imread('pragos.png', 0)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv.filter2D(img, -1, kernelx)
img_prewitty = cv.filter2D(img, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

#Plot Ouput
fig, axes = plt.subplots(4, 2, figsize=(8, 8))#untuk membuat sebuah figure (gambar) yang terdiri dari matriks subplot 4x2 dengan ukuran (lebar, tinggi) sebesar (8, 8) inci.
ax = axes.ravel()#untuk mengubah matriks axes menjadi array satu dimensi dengan menggunakan metode ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_prewittx, cmap='gray')
ax[2].set_title("Citra Output prewitt X")
ax[3].hist(img_prewittx.ravel(), bins=256)
ax[3].set_title("Histogram Citra Output prewitt X")

ax[4].imshow(img_prewitty, cmap='gray')
ax[4].set_title("Citra Output prewitt Y")
ax[5].hist(img_prewitty.ravel(), bins=256)
ax[5].set_title("Histogram Citra Output prewitt Y")

ax[6].imshow(img_prewitt, cmap='gray')
ax[6].set_title("Citra Output prewitt")
ax[7].hist(img_prewitt.ravel(), bins=256)
ax[7].set_title("Histogram Citra Output prewitt")

fig.tight_layout()# untuk secara otomatis mengatur jarak antara subplot dalam sebuah gambar agar lebih rapi dan tidak tumpang tindih.
plt.show()# Menampilkan Output