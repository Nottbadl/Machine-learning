import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_dir = "dataset_cabai_unified/train"
val_dir = "dataset_cabai_unified/val"
test_dir = "dataset_cabai_unified/test"

old_model_path = "model/model_cabai.h5"
unified_model_path = "model/model_cabai_unified.h5"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("Membuat tf.data pipeline untuk 10 kelas gabungan...")
raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

raw_test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = raw_train_ds.class_names
num_classes = len(class_names)
print(f"\nTotal Kelas: {num_classes}")
print("Daftar Kelas:", class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.08, 0.08),
])

train_ds = raw_train_ds.map(
    lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_ds = raw_val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

test_ds = raw_test_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Memuat base model MobileNetV2 yang sudah terlatih
print("\nMemuat pretrained MobileNetV2 base weights...")
old_model = load_model(old_model_path)
base_model = None
for layer in old_model.layers:
    if "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

# Arsitektur model baru untuk 10 kelas
base_model.trainable = True
# Unfreeze 30 layer terakhir
fine_tune_at = max(0, len(base_model.layers) - 30)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=unified_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

print("\nMemulai pelatihan model pada 4.600+ gambar gabungan (5 Epoch)...")
model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    callbacks=callbacks
)

# Evaluasi pada data testing
print("\nMengevaluasi model pada data test gabungan...")
if os.path.exists(unified_model_path):
    best_m = load_model(unified_model_path)
else:
    best_m = model

test_loss, test_acc = best_m.evaluate(test_ds, verbose=1)
print(f"\nHasil Evaluasi Testing Gabungan: Akurasi = {test_acc*100:.2f}%, Loss = {test_loss:.4f}")

# Simpan sebagai model utama
best_m.save("model/model_cabai.h5")
print("Model gabungan berhasil disimpan di model/model_cabai.h5 dan model/model_cabai_unified.h5")
