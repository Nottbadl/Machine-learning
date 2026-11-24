import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_dir = "dataset_cabai_baru/train"
val_dir = "dataset_cabai_baru/val"
test_dir = "dataset_cabai_baru/test"
model_path = "model/model_cabai.h5"
backup_path = "model/model_cabai_backup.h5"
best_model_path = "model/model_cabai_best.keras"

if not os.path.exists(backup_path) and os.path.exists(model_path):
    shutil.copy2(model_path, backup_path)
    print(f"Backup model lama dibuat di {backup_path}")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8

# Data pipeline menggunakan tf.data
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
print("Urutan kelas:", class_names)

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
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

# Muat model
print("Memuat model yang sudah ada...")
model = load_model(model_path)

# Unfreeze top layers dari MobileNetV2
base_model = None
for layer in model.layers:
    if "mobilenetv2" in layer.name.lower():
        base_model = layer
        break

if base_model is not None:
    base_model.trainable = True
    fine_tune_at = max(0, len(base_model.layers) - 30)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    print(f"Base model unfreeze dari layer {fine_tune_at} sampai {len(base_model.layers)}")
else:
    model.trainable = True

# Compile dengan learning rate 5e-5 untuk fine tuning yang stabil
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
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
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

print("\nMemulai proses Fine-Tuning...")
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=callbacks
)

# Evaluasi pada data testing
print("\nMengevaluasi model pada data test...")
if os.path.exists(best_model_path):
    eval_model = load_model(best_model_path)
else:
    eval_model = model

test_loss, test_acc = eval_model.evaluate(test_ds, verbose=1)
print(f"Hasil Evaluasi Testing: Akurasi = {test_acc*100:.2f}%, Loss = {test_loss:.4f}")

# Simpan ke model_cabai.h5 dan model_cabai.keras
eval_model.save(model_path)
eval_model.save("model/model_cabai.keras")
print(f"Model berhasil diperbarui dan disimpan di {model_path}")
