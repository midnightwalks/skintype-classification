#train_model_new.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

print("="*60)
print("TRAINING MOBILENETV2 - 5 KELAS")
print("="*60)

# ==================== KONFIGURASI ====================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5
CLASS_NAMES = ['dry', 'normal', 'oily', 'sensitive', 'combination']

# ==================== DATA GENERATOR ====================
print("\n[1] Loading datasets...")

train_datagen = ImageDataGenerator(
    rescale=1./255
)

val_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")

# ==================== BUILD MODEL ====================
print("\n[2] Building MobileNetV2 model...")

def create_model():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

model, base_model = create_model()
model.summary()

# ==================== COMPILE ====================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==================== CALLBACKS ====================
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/mobilenetv2_5classes.h5',  # Nama model baru
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7
    )
]

# ==================== TRAINING ====================
print("\n[3] Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

print("\n===== HASIL MODEL TERBAIK =====")

best_epoch = np.argmax(
    history.history['val_accuracy']
)

print(
    f"Epoch terbaik: {best_epoch+1}"
)

print(
    f"Training Accuracy : {history.history['accuracy'][best_epoch]:.4f}"
)

print(
    f"Validation Accuracy : {history.history['val_accuracy'][best_epoch]:.4f}"
)

print(
    f"Training Loss : {history.history['loss'][best_epoch]:.4f}"
)

print(
    f"Validation Loss : {history.history['val_loss'][best_epoch]:.4f}"
)

# ==================== PLOT RESULTS ====================
print("\n[4] Plotting training history...")
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy - 5 Classes')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss - 5 Classes')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history_5classes.png')
    plt.show()

plot_training_history(history)

# ==================== EVALUATE ====================
print("\n[5] Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f}")
print(f"✅ Test Loss: {test_loss:.4f}")

# ==================== SAVE MODEL ====================
os.makedirs('models', exist_ok=True)
model.save('models/mobilenetv2_5classes_final.h5')
print("\n[6] Model saved to models/mobilenetv2_5classes_final.h5")

print("\n" + "="*60)
print("✅ TRAINING COMPLETED!")
print("="*60)