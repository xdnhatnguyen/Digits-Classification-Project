import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2

# FIX 1: Import your actual model class
from src.model import CNN  

# LOAD MODEL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ensure this path matches exactly where main.py saved it
MODEL_PATH = 'model/mnist_CNN.pt' 
model = CNN().to(DEVICE)

def load_pytorch_model_from_file():
    try:
 
        # FIX 3: Uncomment this to actually load the trained weights!
        # If you leave this commented out, you are predicting with a random, empty model.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        
        model.eval()
        print(f"✅ Successfully loaded model from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return None

# --- LOGIC XỬ LÝ BIẾN MODEL ---
model = load_pytorch_model_from_file()

# ==========================================
# HÀM XỬ LÝ ẢNH & DỰ ĐOÁN
# ==========================================
def predict_digit(image):
    # Handle Gradio 4.x dictionary input
    if isinstance(image, dict):
        image = image['composite']

    if image is None:
        return None

    # 1. Preprocessing
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (28, 28))

    # Invert colors if needed (black background logic)
    if np.mean(image) > 127: 
        image = 255 - image

    image = image.astype('float32') / 255.0
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # 2. Inference
    if model:
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            prob_arr = probabilities.cpu().numpy()[0]

        return {str(i): float(prob_arr[i]) for i in range(10)}
    else:
        return {"Error": 0.0}

# ==========================================
# GIAO DIỆN GRADIO
# ==========================================
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="Draw a digit"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="HCMUS-ConChoCaoBangBoPC: AI Digit Recognizer",
    live=True
)

if __name__ == "__main__":
    demo.launch(share=True)
