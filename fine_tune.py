# fine_tune.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print("🔄 FINE-TUNING MODEL...")

# 1. Load model yang sudah ada
model = tf.keras.models.load_model('models/mobilenetv2_5classes_final.h5')

# 2. Unfreeze beberapa layer terakhir
base_model = model.get_layer('mobilenetv2_1.00_224')
base_model.trainable = True

# Unfreeze 50 layer terakhir
for layer in base_model.layers[:100]:
    layer.trainable = False

# 3. Compile dengan learning rate lebih kecil
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Data augmentation 
train_datagen = ImageDataGenerator(
    rescale=1./255
)

val_datagen = ImageDataGenerator(rescale=1./255)

# 5. Load data
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['dry', 'normal', 'oily', 'sensitive', 'combination'],
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=['dry', 'normal', 'oily', 'sensitive', 'combination'],
    shuffle=False
)

# 6. Fine-tuning
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
    ]
)

# 7. Save model baru
model.save('models/mobilenetv2_finetuned.h5')
print("✅ Fine-tuning selesai! Model disimpan sebagai 'mobilenetv2_finetuned.h5'")