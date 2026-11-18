import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 1. KONFIGURASI PARAMETER
IMG_SIZE = 150  # Ukuran gambar diubah menjadi 150x150 piksel
BATCH_SIZE = 32
EPOCHS = 20     # Jumlah perulangan training
NUM_CLASSES = 2 # Sesuaikan dengan jumlah folder/orang di dataset kamu (budi, andi, unknown)

# 2. AUGMENTASI DATA (Penting untuk skripsi agar model tidak Overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalisasi nilai piksel ke 0-1
    rotation_range=20,         # Rotasi gambar acak up to 20 derajat
    width_shift_range=0.2,     # Pergeseran horizontal
    height_shift_range=0.2,    # Pergeseran vertikal
    shear_range=0.2,           # Peregangan gambar
    zoom_range=0.2,            # Zoom acak
    horizontal_flip=True,      # Membalik gambar secara horizontal
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255) # Validasi hanya butuh normalisasi

# 3. MEMUAT DATASET DARI FOLDER
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Cetak urutan indeks kelas untuk dokumentasi Bab 4 skripsi kamu
print("Indeks Kelas:", train_generator.class_indices)

# 4. MEMBANGUN ARSITEKTUR CUSTOM CNN (Jantung Skripsi Kamu)
model = Sequential([
    # Layer Konvolusi Ke-1 + Max Pooling
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    
    # Layer Konvolusi Ke-2 + Max Pooling
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Layer Konvolusi Ke-3 + Max Pooling
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Mengubah matriks 2D menjadi Vektor 1D
    Flatten(),
    
    # Fully Connected Layer (Dense Layer)
    Dense(512, activation='relu'),
    Dropout(0.5), # Mencegah overfitting
    
    # Output Layer (Menggunakan Softmax untuk klasifikasi multi-kelas)
    Dense(NUM_CLASSES, activation='softmax')
])

# 5. COMPILE MODEL
model.compile(
    optimizer='adam',
    loss='binary_crossentropy' if NUM_CLASSES == 2 else 'categorical_crossentropy',
    metrics=['accuracy']
)

# Menampilkan ringkasan arsitektur untuk disalin ke Bab 4 Skripsi
model.summary()

# 6. PROSES TRAINING
print("\n--- Memulai Proses Training CNN ---")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# 7. MENYIMPAN MODEL (.h5)
model.save('model_face_recognition_cnn.h5')
print("\n[INFO] Model CNN berhasil dilatih dan disimpan dengan nama 'model_face_recognition_cnn.h5'")

# 8. MEMBUAT GRAFIK AKURASI & LOSS (Wajib Ada di Bab 4 Skripsi!)
plt.figure(figsize=(12, 4))

# Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Grafik Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Grafik Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('grafik_evaluasi_skripsi.png')
plt.show()
