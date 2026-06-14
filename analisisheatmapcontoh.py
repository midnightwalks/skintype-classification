import cv2
import matplotlib.pyplot as plt

overlay = cv2.imread("hasiloverlay.jpg")
overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

img = overlay.copy()

# lingkaran kecil area fokus
cv2.circle(
    img,
    (390,180),  # sesuaikan posisi
    60,
    (255,0,0),
    3
)

fig, ax = plt.subplots(1,2, figsize=(10,4))

ax[0].imshow(overlay)
ax[0].set_title("Superimposed")

ax[1].imshow(img)
ax[1].set_title("Area Fokus Model")

for a in ax:
    a.axis("off")

plt.tight_layout()

plt.savefig(
    "Gambar_4_5_AnalisisHeatmap.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()