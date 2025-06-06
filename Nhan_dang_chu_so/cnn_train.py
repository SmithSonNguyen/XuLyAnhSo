import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ============================
# 1. Load dữ liệu và tiền xử lý
# ============================

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape và chuẩn hóa dữ liệu
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encoding cho labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ================================
# 2. Data Augmentation
# ================================
datagen = ImageDataGenerator(
    rotation_range=20,       # Xoay ảnh ngẫu nhiên
    width_shift_range=0.2,   # Dịch chuyển chiều ngang
    height_shift_range=0.2,  # Dịch chuyển chiều dọc
    zoom_range=0.2,          # Thu/phóng ảnh
    shear_range=0.2,         # Cắt ảnh
    fill_mode='nearest'      # Điền vào vị trí trống sau khi biến đổi
)

datagen.fit(X_train)

# ============================
# 3. Xây dựng mô hình CNN
# ============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),    # Regularization
    Dense(10, activation='softmax')   # Output layer cho 10 chữ số (0–9)
])

# Compile mô hình với Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ============================
# 4. Huấn luyện mô hình
# ============================
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          epochs=30,
          validation_data=(X_test, y_test))

# Lưu mô hình sau khi huấn luyện
model.save("mnist_cnn_augmented.h5")
print("✅ Mô hình đã được lưu vào mnist_cnn_augmented.h5")
