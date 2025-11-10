# 🧠 Digit Classification using PyTorch (MNIST Dataset)

### 📘 Course Project – Class 25CTT3B - Faculty of Information Technology - HCMUS  
**Team:** HCMUS-ConChoCaoBangBoPC  
**Team Members:** 6 students  
**Framework:** NumPy, Matplotlib, PyTorch  
**Dataset:** MNIST Handwritten Digits  
**Language:** Python  

---

## 📍 1. Giới thiệu dự án

Đây là dự án học tập nhằm tìm hiểu và xây dựng mô hình **nhận diện chữ số viết tay (Digit Classification)** sử dụng **PyTorch** và **bộ dữ liệu MNIST**.

Mục tiêu của dự án:
- Hiểu rõ **quy trình huấn luyện mô hình học máy (Machine Learning pipeline)**.  
- Làm quen với **xử lý dữ liệu ảnh, xây dựng mạng nơ-ron (Neural Network)**.  
- Ứng dụng các kiến thức nền tảng về **toán học, tối ưu và lập trình Python** vào thực tế.  

Kết quả mong muốn:
- Huấn luyện thành công mô hình có **độ chính xác ≥ 95%** trên tập kiểm thử MNIST.

---

## 🧩 2. Mô tả bài toán

**Bài toán:**  
Cho một ảnh viết tay kích thước **28×28 pixel**, dự đoán chữ số (0–9) mà ảnh biểu diễn.

**Đầu vào (Input):**  
Ảnh grayscale 28×28, mỗi pixel ∈ [0, 255].

**Đầu ra (Output):**  
Một vector xác suất gồm 10 phần tử tương ứng các lớp số (0–9).  
Lớp có xác suất cao nhất được chọn làm kết quả dự đoán.

---

## 🧠 3. Kiến thức và nền tảng sử dụng

### 🔹 Machine Learning / Deep Learning
- **Phân loại (Classification)** là một trong những bài toán cơ bản của học máy.  
- Sử dụng mô hình **Neural Network (NN)** và **Convolutional Neural Network (CNN)**.  
- Huấn luyện bằng thuật toán **Gradient Descent** và hàm mất mát **Cross-Entropy Loss**.

### 🔹 Toán học nền tảng
| Mảng | Ứng dụng trong dự án |
|------|----------------------|
| **Đại số tuyến tính** | Biểu diễn ảnh và phép nhân ma trận trong mạng nơ-ron |
| **Giải tích (Đạo hàm)** | Cập nhật trọng số mô hình thông qua Gradient Descent |
| **Xác suất – Thống kê** | Hiểu xác suất dự đoán (Softmax) và đánh giá mô hình |

### 🔹 Python và PyTorch
- **Python**: Sử dụng cơ bản về `list`, `dict`, `for`, `class`, hàm.  
- **PyTorch core:**
  - `torch.Tensor`, `torch.autograd`
  - `torch.nn.Module`, `torch.nn.Sequential`
  - `torch.optim` (SGD, Adam)
  - `torchvision.datasets.MNIST`, `DataLoader`, `transforms`
- **Thư viện bổ trợ:** `numpy`, `matplotlib`, `torchvision`

---

## ⚙️ 4. Kiến trúc mô hình

Hai mô hình được thử nghiệm trong dự án:

### **1️⃣ Fully Connected Neural Network (FCNN)**
- Lớp ẩn: 128 neurons, kích hoạt ReLU  
- Lớp đầu ra: 10 neurons, kích hoạt Softmax  
- Loss: CrossEntropyLoss  
- Optimizer: SGD / Adam  

### **2️⃣ Convolutional Neural Network (CNN)**
- `Conv2d(1, 32, 3)` → `ReLU` → `MaxPool2d(2)`  
- `Conv2d(32, 64, 3)` → `ReLU` → `MaxPool2d(2)`  
- `Linear(64*5*5, 128)` → `ReLU` → `Linear(128, 10)`  
- Cho độ chính xác cao hơn rõ rệt so với FCNN.

---

## 🔄 5. Quy trình thực hiện và phân công

### Giản đồ **Gantt**:  
![Quy trình thực hiện và phân công](https://raw.githubusercontent.com/xdnhatnguyen/Digits-Classification-Project/main/GanttChart.png)
---

## 📊 6. Kết quả dự kiến
| Mô hình | Độ chính xác huấn luyện | Độ chính xác kiểm thử |
|----------|--------------------------|------------------------|
| FCNN | ~92–94% | ~91–93% |
| CNN | ~98–99% | ~97–98% |

Visualization:
- Biểu đồ loss/accuracy theo epoch.  
- Một số ảnh test kèm dự đoán mô hình.

---

## 🧩 7. Cấu trúc thư mục dự án
```bash
digits_classification/
├── configs/
│   └── config.yaml
│
├── src/
│   ├── losses/
│   │   └── loss.py
│   │
│   ├── models/
│   │   └── model.py
│   │
│   └── data/
│       └── dataloader.py
│
├── trainer.py
│
├── requirements.txt
│
└── README.md
```

---

## 🧰 8. Cách chạy dự án

### Cài đặt môi trường:
```bash
pip install torch torchvision matplotlib numpy
```
Huấn luyện mô hình:
```bash
python src/train.py
```
Kiểm thử mô hình:
```bash
python src/test.py
```
💡 9. Kết luận & Hướng phát triển


/---/


👨‍💻 10. Thành viên nhóm 25CTT3
| STT | Họ và Tên | MSSV                           |
| --- | --------- | ------------------------------ |
| 1   | Nhật        | 25120xxx                     |
| 2   | Phong       | 25120xxx                     |
| 3   | Quang       | 25120xxx                     |
| 4   | Quang       | 25120xxx                     |
| 5   | Quang       | 25120xxx                     |
| 6   | Quang       | 25120xxx                     |



---


