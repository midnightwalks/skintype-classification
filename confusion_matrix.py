# confusion_matrix.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ==================== KONFIGURASI ====================

CLASS_NAMES = [
    'dry',
    'normal',
    'oily',
    'sensitive',
    'combination'
]

MODEL_PATH='models/mobilenetv2_5classes_final.h5'

# ==================== LOAD MODEL ====================

print("Loading model...")
model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded")

# ==================== LOAD TEST DATA ====================

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    classes=CLASS_NAMES,     # PENTING
    shuffle=False
)

print("\nClass mapping:")
print(
    test_generator.class_indices
)

# ==================== PREDIKSI ====================

print("\nPredicting...")

predictions = model.predict(
    test_generator
)

y_pred=np.argmax(
    predictions,
    axis=1
)

y_true=test_generator.classes

# ==================== CLASSIFICATION REPORT ====================

print("\n"+"="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES
    )
)

# ==================== CONFUSION MATRIX ====================

cm=confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(10,8)
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    'Confusion Matrix - Skin Type Classification'
)

plt.xlabel(
    'Predicted Label'
)

plt.ylabel(
    'True Label'
)

plt.tight_layout()

plt.savefig(
    'results/confusion_matrix.png',
    dpi=300
)

plt.show()

# ==================== AKURASI PER KELAS ====================

print("\n"+"="*60)
print("AKURASI PER KELAS")
print("="*60)

for i,class_name in enumerate(
    CLASS_NAMES
):

    class_mask=(
        y_true==i
    )

    class_acc=np.mean(
        y_pred[class_mask]==i
    )

    print(
        f"{class_name:12}: {class_acc:.2%}"
    )