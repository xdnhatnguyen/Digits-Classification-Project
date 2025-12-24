![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)

> Course Project **Introduction to Information Technology** > **Faculty of Information Technology - VNU-HCM University of Science**

This project implements and compares the performance of Deep Learning models (MLP & CNN) for handwritten digit recognition using the MNIST dataset. The project is built with a modular architecture, designed for extensibility and academic research purposes

---

## 📄 Documentation & Paper

The project is accompanied by a detailed scientific report, providing an in-depth analysis of the mathematical foundations and empirical evaluation.

👉 **[Read the full report (PDF)](./article_paper.pdf)**

---

## ✨ Key Features

* **Modular Architecture:** Clear separation between Data Loading, Model, Loss function, and Training loop.
* **Multi-Architecture Support:**
    * **MLP (Multi-layer Perceptron):** Fully Connected Neural Network, Basic Feed-Forward Network (Baseline).
    * **CNN (Convolutional Neural Network):** Convolutional network optimized for spatial feature extraction.
* **Reproducibility:** Hyperparameter management via `yaml` configuration files.
* **Logging & Visualization:** Real-time loss/accuracy tracking and prediction visualization.

## 📅 Project Timeline and Team Members (HCMUS-ConChoCaoBangBoPC)

Below is the implementation progress and task distribution of the team throughout the development process:

![Gantt Chart](./assets/Gantt.png)

---

## 📂 Project Structure

```bash
digits_classification/
├── configs/            # Configuration files
│   └── config.yaml     # Main config (Epochs, LR, Model type...)
├── assets/             # Image files
│   └── GanttChart.png
├── src/                # Source code
│   ├── data/           # Data processing module (DataLoader, Transforms)
│   ├── models/         # Model architecture definitions (CNN, MLP)
│   ├── losses/         # Loss functions
│   └── utils/          # Utilities (Visualization, Logger)
├── saved_models/       # Directory for saving trained model weights
├── article_paper.pdf   # Scientific report file
├── trainer.py          # Training script
├── test.py             # Testing/Evaluation script
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```  
---
## 🚀 Installation & Usage  
### 1. Environment Setup  
Requires Python 3.8+.
```bash
# Clone repository
git clone [https://github.com/username/digits-classification.git](https://github.com/username/digits-classification.git)
cd digits-classification

# Create virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```
### 2. Training  
You can modify parameters in ```configs/config.yaml``` before running.
```
python trainer.py --config configs/config.yaml
```
The model with the highest accuracy will be automatically saved to ```saved_models/```.  
### 3. Testing:  
Evaluate the model on the Test set:
```bash
python test.py --model_path saved_models/best_model.pth
```
---
## 👥 Researchers:

### HCMUS - CONCHOCAOBANGBOPC - 25CTT3

| Members | StudentID |
| :--- | :--- |
| Nguyễn Minh Nhật | 25120215 |
| Vũ Thanh Phong | 25120219 |
| Đỗ Lê Nhật Quang | 25120223 |
| Nguyễn Phú Quang | 25120224 |
| Nguyễn Vũ Nhật Quang | 25120225 |
| Phạm Đăng Quang | 25120226 |

Lab Instructor: Thầy Lê Đức Khoan.

---

## 📝 License
This project is distributed under the MIT license.

