import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, confusion_matrix
import joblib
import os
import numpy as np
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

# Đảm bảo thư mục checkpoints tồn tại
os.makedirs('/home/thuytrang/C_CODE/1_HOC-MAY/spam_email_ver-3/checkpoints', exist_ok=True)

# Bước 1: Đọc tệp dữ liệu
print("Đang đọc dữ liệu...")
data = pd.read_csv('/home/thuytrang/C_CODE/spam_email_ver-3/spam_ham_dataset.csv')
print(f"Số lượng mẫu: {len(data)}")
print(f"Số email spam: {len(data[data['label'] == 'spam'])}")
print(f"Số email ham: {len(data[data['label'] == 'ham'])}")

# Bước 2: Chuẩn bị dữ liệu và nhãn
print("\nChuẩn bị dữ liệu...")
X = data['text']
y = data['label'].apply(lambda x: 1 if x == 'spam' else 0)
print("Đã chuyển đổi nhãn thành dạng số (1: spam, 0: ham)")

# Bước 3: Chia tập huấn luyện và kiểm tra
print("\nChia tập dữ liệu...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Kích thước tập huấn luyện: {len(X_train)} mẫu")
print(f"Kích thước tập kiểm tra: {len(X_test)} mẫu")

# Bước 4: Biến đổi văn bản thành vector đặc trưng
print("\nChuyển đổi văn bản thành vector TF-IDF...")
start_time = time()
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Kích thước ma trận đặc trưng: {X_train_vec.shape}")
print(f"Thời gian chuyển đổi: {time() - start_time:.2f} giây")

# Bước 5: Huấn luyện mô hình SVM
print("\nBắt đầu huấn luyện mô hình LinearSVC...")
start_time = time()
model = LinearSVC(
    random_state=42,
    max_iter=1000,
    C=1.0,
    verbose=1
)

print("Thông số mô hình:")
print(f"- C (regularization parameter): {model.C}")
print(f"- Max iterations: {model.max_iter}")
print(f"- Random state: {model.random_state}")

# Huấn luyện và đo thời gian
print("\nTiến trình huấn luyện:")
model.fit(X_train_vec, y_train)
training_time = time() - start_time

# Hiển thị các thông số sau khi huấn luyện
print(f"\nThời gian huấn luyện: {training_time:.2f} giây")
print(f"Số lần lặp thực tế: {model.n_iter_}")
print(f"Đã hội tụ: {model.n_iter_ < model.max_iter}")

# Đánh giá mô hình
print("\nĐánh giá mô hình...")
y_pred = model.predict(X_test_vec)

# Tính các chỉ số đánh giá
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Hiển thị các chỉ số
print("\nCác chỉ số đánh giá hiệu suất mô hình:")
print(f"- Accuracy: {accuracy * 100:.2f}%")
print(f"- Precision: {precision:.4f}")
print(f"- Recall: {recall:.4f}")
print(f"- F1 Score: {f1:.4f}")
print(f"- MAE (Mean Absolute Error): {mae:.4f}")
print(f"- RMSE (Root Mean Squared Error): {rmse:.4f}")

# Tạo và hiển thị Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("[[True Negative  False Positive]")
print(" [False Negative True Positive]]")
print(cm)

# Hiển thị Confusion Matrix dưới dạng bảng đẹp hơn bằng Pandas
cm_df = pd.DataFrame(cm, index=['Ham (0)', 'Spam (1)'], columns=['Predicted Ham (0)', 'Predicted Spam (1)'])
print("\nConfusion Matrix (Bảng chi tiết):")
print(cm_df)

# Vẽ Confusion Matrix bằng heatmap và lưu thành file
print("\nĐang tạo và lưu Confusion Matrix dưới dạng heatmap...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham (0)', 'Spam (1)'], yticklabels=['Ham (0)', 'Spam (1)'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/home/thuytrang/C_CODE/1_HOC-MAY/spam_email_ver-3/checkpoints/confusion_matrix.png')
print("Heatmap đã được lưu tại: /home/thuytrang/C_CODE/1_HOC-MAY/spam_email_ver-3/checkpoints/confusion_matrix.png")

# Bước 6: Lưu mô hình và vectorizer
model_path = '/home/thuytrang/C_CODE/1_HOC-MAY/spam_email_ver-3/checkpoints/spam_detection_model.pkl'
vectorizer_path = '/home/thuytrang/C_CODE/1_HOC-MAY/spam_email_ver-3/checkpoints/tfidf_vectorizer.pkl'

print("\nLưu mô hình...")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")
print("Hoàn tất quá trình huấn luyện và lưu trữ!")