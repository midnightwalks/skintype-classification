# evaluate_new.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

print("="*60)
print("EVALUASI MODEL - 5 KELAS")
print("="*60)

# ==================== KONFIGURASI ====================
CLASS_NAMES = ['dry', 'normal', 'oily', 'sensitive', 'combination']
MODEL_PATH='models/mobilenetv2_5classes_final.h5'

# ==================== LOAD MODEL ====================
print("\n[1] Loading model...")
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
else:
    print(f"❌ Model not found: {MODEL_PATH}")
    exit()

# ==================== LOAD DATA ====================
print("\n[2] Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

# ==================== PREDIKSI ====================
print("\n[3] Making predictions...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# ==================== CLASSIFICATION REPORT ====================
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ==================== CONFUSION MATRIX ====================
print("\n[4] Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - 5 Classes', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix_5classes.png', dpi=300)
plt.show()

# ==================== AKURASI PER KELAS ====================
print("\n" + "="*60)
print("AKURASI PER KELAS")
print("="*60)
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = (y_true == i)
    if sum(class_mask) > 0:
        class_acc = np.mean(y_pred[class_mask] == i)
        print(f"{class_name:12}: {class_acc:.2%} ({sum(class_mask)} samples)")

# ==================== EVALUASI ====================
print("\n" + "="*60)
print("HASIL EVALUASI")
print("="*60)
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")