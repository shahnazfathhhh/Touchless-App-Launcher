"""
train_model.py
Jalanin file ini SEKALI untuk melatih CNN model dari dataset.
Hasil model disimpan sebagai 'finger_count_model.h5'
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Path dataset ──────────────────────────────────────────
DATASET_TRAIN = r'dataset/train'   

# ── Load Dataset ──────────────────────────────────────────
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    DATASET_TRAIN,
    target_size   = (64, 64),
    color_mode    = 'grayscale',
    batch_size    = 32,
    class_mode    = 'categorical',
    subset        = 'training'
)

val_gen = train_datagen.flow_from_directory(
    DATASET_TRAIN,
    target_size   = (64, 64),
    color_mode    = 'grayscale',
    batch_size    = 32,
    class_mode    = 'categorical',
    subset        = 'validation'
)

print("Train samples :", train_gen.samples)
print("Val samples   :", val_gen.samples)
print("Classes       :", train_gen.class_indices)

# ── Buat Model CNN ────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')   # 6 kelas: 0-5 jari
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ── Training ──────────────────────────────────────────────
callbacks_list = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('finger_count_model.h5', save_best_only=True)
]

history = model.fit(
    train_gen,
    steps_per_epoch  = len(train_gen),
    epochs           = 40,
    validation_data  = val_gen,
    validation_steps = len(val_gen),
    callbacks        = callbacks_list
)

print("\nTraining selesai! Model disimpan sebagai finger_count_model.h5 ✅")

# ── Plot hasil training ───────────────────────────────────
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
nepochs = len(history.history['loss'])
plt.plot(range(nepochs), history.history['loss'],         'r-', label='train')
plt.plot(range(nepochs), history.history['val_loss'],     'b-', label='val')
plt.legend(prop={'size': 14})
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(range(nepochs), history.history['accuracy'],     'r-', label='train')
plt.plot(range(nepochs), history.history['val_accuracy'], 'b-', label='val')
plt.legend(prop={'size': 14})
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.tight_layout()
plt.savefig('training_result.png')
plt.show()
print("Plot disimpan sebagai training_result.png ✅")
