import cv2 as cv
from matplotlib import pyplot as plt

# Membaca gambar
img0 = cv.imread('pragos.png')

# Konversi ke grayscale
gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

# Hilangkan noise dengan filter Gaussian
img = cv.GaussianBlur(gray, (3, 3), 0)

# Aplikasikan filter Laplacian
laplacian = cv.Laplacian(img, cv.CV_64F)
laplacian = laplacian / laplacian.max()

# Tampilkan dengan matplotlib
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.show()

# Menampilkan gambar dengan OpenCV
cv.imshow('laplacian-gaussian', laplacian)
cv.waitKey(0)
cv.destroyAllWindows()
