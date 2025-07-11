# 📧 Dự án Phân loại Email (Email Classification)

Dự án này giúp phân loại email thành **spam** hoặc **không spam** bằng thuật toán học máy. Toàn bộ dữ liệu và mã nguồn đã được chuẩn bị sẵn trong thư mục dự án. Bạn chỉ cần tạo máy ảo, train mô hình và chạy chương trình.

---

## 🧾 Yêu cầu hệ thống

- Python 3.8 trở lên
- pip
- Máy ảo Ubuntu (20.04 hoặc 22.04 khuyến nghị)

---

## ⚙️ Hướng dẫn chạy dự án

### 1. Tạo máy ảo Ubuntu

- Tạo máy ảo bằng VirtualBox, VMware hoặc WSL
- Cài đặt Ubuntu (khuyến nghị Ubuntu Server 20.04)
- Cấp ít nhất 2GB RAM, 20GB ổ cứng

### 2. Cài Python & pip

```bash
sudo apt update
sudo apt install python3 python3-pip -y
### 3. Clone dự án về máy
### 4. Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate
### 5. Cài các thư viện trong requirements.txt
### 6. Train mô hình
python train_model.py
### 7. Chạy file
python app-svm.py
