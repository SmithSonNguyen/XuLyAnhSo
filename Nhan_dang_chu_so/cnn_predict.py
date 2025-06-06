import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ============================
# 1. Load mô hình đã huấn luyện
# ============================
model = load_model("mnist_cnn_augmented.h5")

# ============================
# 2. Tiền xử lý ảnh đầu vào
# ============================
image_path = "digit2.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Không tìm thấy ảnh!")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

# ============================
# 3. Tìm contour của các chữ số
# ============================
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # Sắp xếp từ trái sang phải

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    roi = binary[y:y + h, x:x + w]

    # Padding và resize
    roi = np.pad(roi, ((20, 20), (20, 20)), mode='constant', constant_values=0)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 28, 28, 1)

    # ============================
    # 4. Dự đoán với mô hình CNN
    # ============================
    prediction = model.predict(roi)
    digit = np.argmax(prediction)

    # Hiển thị kết quả trên ảnh
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, str(digit), (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

# ============================
# 5. Hiển thị và lưu kết quả
# ============================
cv2.imshow("CNN Result", image)
cv2.imwrite("cnn_result.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
