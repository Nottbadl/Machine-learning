
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

train_dir = "dataset_cabai_baru/train"
val_dir = "dataset_cabai_baru/val"
test_dir = "dataset_cabai_baru/test"

os.makedirs("model", exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==========================================
# 4. MEMERIKSA KELAS DATASET
# ==========================================

print("\nInformasi Dataset")
print("----------------------------------")

print(f"Jumlah data training   : {train_generator.samples}")
print(f"Jumlah data validation: {val_generator.samples}")
print(f"Jumlah data testing    : {test_generator.samples}")

jumlah_kelas = train_generator.num_classes

print(f"Jumlah kategori        : {jumlah_kelas}")
print(f"Urutan kelas           : {list(train_generator.class_indices.keys())}")

# Memastikan urutan kelas sama
if train_generator.class_indices != val_generator.class_indices:
    raise ValueError("Urutan kelas training dan validation berbeda!")

if train_generator.class_indices != test_generator.class_indices:
    raise ValueError("Urutan kelas training dan testing berbeda!")

# ==========================================
# 5. MEMBANGUN MODEL MOBILENETV2
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Membekukan layer dasar MobileNetV2
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(jumlah_kelas, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 6. PROSES PELATIHAN MODEL
# ==========================================

print("\nMemulai pelatihan model MobileNetV2...")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ==========================================
# 7. EVALUASI MENGGUNAKAN DATA TESTING
# ==========================================

print("\nMemulai evaluasi menggunakan data testing...")

test_loss, test_accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\nHASIL EVALUASI MODEL")
print("----------------------------------")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")

# ==========================================
# 8. CONFUSION MATRIX DAN METRIK EVALUASI
# ==========================================

print("\nMemulai evaluasi lengkap model...")

# Memastikan generator dimulai dari awal
test_generator.reset()

# Melakukan prediksi terhadap seluruh data testing
predictions = model.predict(
    test_generator,
    verbose=1
)

# Mengambil kelas hasil prediksi
y_pred = np.argmax(predictions, axis=1)

# Mengambil label kelas sebenarnya
y_true = test_generator.classes

# Mengambil nama kelas
class_names = list(test_generator.class_indices.keys())

# Membuat confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(jumlah_kelas))
)

print("\nCONFUSION MATRIX")
print("----------------------------------")
print(cm)

# Menghitung precision, recall, dan f1-score
print("\nHASIL PRECISION, RECALL, DAN F1-SCORE")
print("----------------------------------")

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(jumlah_kelas)),
    target_names=class_names,
    digits=4,
    zero_division=0
)

print(report)

# ==========================================
# 9. MENYIMPAN GAMBAR CONFUSION MATRIX
# ==========================================

fig, ax = plt.subplots(figsize=(12, 9))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    values_format="d",
    colorbar=False
)

plt.title("Confusion Matrix - MobileNetV2")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nConfusion Matrix berhasil disimpan.")
print("Lokasi: model/confusion_matrix.png")

# ==========================================
# 10. MENYIMPAN MODEL
# ==========================================

model.save("model/model_cabai.h5")

print("\nPelatihan dan evaluasi selesai!")
print("Model berhasil disimpan di model/model_cabai.h5")
