import cv2
import matplotlib.pyplot as plt

# Membaca gambar
image = cv2.imread("image_191.jpg")

# OpenCV baca format BGR → ubah ke RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize menjadi 224x224
resized_image = cv2.resize(image_rgb, (224,224))

# Tampilkan sebelum dan sesudah
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(image_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(resized_image)
plt.title("Resize 224×224")
plt.axis("off")

plt.tight_layout()

# Simpan gambar untuk dimasukkan ke skripsi
plt.savefig("resize_output.png", dpi=300, bbox_inches='tight')

plt.show()