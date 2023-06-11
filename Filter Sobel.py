# Memanggil Library yang dibutuhkan
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Memanggil gambar yang disimpan dan mengubah citra menjadi grayscale (0)
img = cv.imread('pragos.png', 0)

# Memanggil proses Sobel dengan ukuran kernel 5
img_sobelx = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)#untuk menerapkan operator Sobel pada citra img untuk mendeteksi gradien tepi horizontal (gradien dalam sumbu x)
img_sobely = cv.Sobel(img, cv.CV_8U, 0, 1, ksize=5)#untuk menerapkan operator Sobel pada citra img untuk mendeteksi gradien tepi horizontal (gradien dalam sumbu y)
img_sobel = img_sobelx + img_sobely#untuk menggabungkan hasil deteksi tepi menggunakan operator Sobel pada sumbu x (img_sobelx) dan sumbu y (img_sobely)

#Plot Ouput
fig, axes = plt.subplots(4, 2, figsize=(8, 8))#untuk membuat sebuah figure (gambar) yang terdiri dari matriks subplot 4x2 dengan ukuran (lebar, tinggi) sebesar (8, 8) inci.
ax = axes.ravel()#untuk mengubah matriks axes menjadi array satu dimensi dengan menggunakan metode ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_sobelx, cmap='gray')
ax[2].set_title("Citra Output Sobel X")
ax[3].hist(img_sobelx.ravel(), bins=256)
ax[3].set_title("Histogram Citra Output Sobel X")

ax[4].imshow(img_sobely, cmap='gray')
ax[4].set_title("Citra Output Sobel Y")
ax[5].hist(img_sobely.ravel(), bins=256)
ax[5].set_title("Histogram Citra Output Sobel Y")

ax[6].imshow(img_sobel, cmap='gray')
ax[6].set_title("Citra Output Sobel")
ax[7].hist(img_sobel.ravel(), bins=256)
ax[7].set_title("Histogram Citra Output Sobel")

fig.tight_layout()# untuk secara otomatis mengatur jarak antara subplot dalam sebuah gambar agar lebih rapi dan tidak tumpang tindih.
plt.show()# Menampilkan Output
