import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import src.dataloader
import src.model
import src.trainer
import src.utils
import gradio as gr
import numpy as np
import cv2

def trainModel(model, model_name):
    """ SETUP """

    print("Setup...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CNN_model = model.to(device) # Định nghĩa model, chuyển vào cuda để tính toán
    # model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001) # Tối ưu hóa Adam
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Theo dõi Validation Loss (giảm là tốt)
        factor=0.5,         # Hệ số giảm LR (LR_new = LR_old * 0.5)
        patience=5,         # Số epoch chờ đợi trước khi giảm LR
        #verbose=True        # In thông báo khi LR bị giảm
    )
    loss_fn = nn.CrossEntropyLoss() # Hàm loss cho bài toán classification

    """ CÁC BIẾN THEO DÕI """

    print("initialize monitoring variables...")

    running_loss_list = []
    accuracy = []
    NUM_CLASSES = 10
    confusion_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES).to(device)

    """ LẤY DỮ LIỆU HUẤN LUYỆN """

    print("loading data...")

    loaders = src.dataloader.getLoaders()

    """ LẶP HUẤN LUYỆN MÔ HÌNH CNN """

    EPOCHS = 5
    for epoch in range(1, 6):
        print(f"Training {model_name} model: Epoch", epoch, "...")
        running_loss_list.extend(src.trainer.train(model=CNN_model,
                                loaders=loaders,
                                epoch=epoch,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device))
        accuracy.extend(src.trainer.valid(model=CNN_model,
                                        loaders=loaders,
                                        device=device,
                                        loss_fn=loss_fn,
                                        confusion_matrix=confusion_matrix))
        
    """ LƯU MÔ HÌNH """
    save_path = f"model/mnist_{model_name}.pt"
    torch.save(CNN_model.state_dict(), save_path)
    print(f"{model_name} model saved successfully to {save_path}")

    return (running_loss_list, accuracy, confusion_matrix)

    

if __name__ == "__main__":

    eval_data = []

    MLP_model=src.model.MLP()
    CNN_model=src.model.CNN()

    eval_data.append(trainModel(MLP_model, model_name="MLP"))
    eval_data.append(trainModel(CNN_model, model_name="CNN"))

    """ IN RA KẾT QUẢ HUẤN LUYỆN MÔ HÌNH CNN """ 
    src.utils.plot_graph(list_1=eval_data[0][0],
                         list_2=eval_data[1][0],
                         name_1="MLP Model",
                         name_2="CNN Model",
                         title="Loss over time")
    
    """ IN RA KẾT QUẢ HUẤN LUYỆN MÔ HÌNH CNN """ 
    src.utils.plot_graph(list_1=eval_data[0][1],
                         list_2=eval_data[1][1],
                         name_1="MLP Model",
                         name_2="CNN Model",
                         title="Accuracy over time")

    """ IN RA BẢNG CONFUSION CỦA MÔ HÌNH """
    src.utils.plot_confusion_matrix(cm_tensor=src.utils.eval_model_correctness(eval_data[0][2]),
                                    classesX=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "Macro"],
                                    classesY=["TP", "FP", "FN", "Precision", "Recall", "F1-Score"], fmt='.3f',
                                    normalize=False, title="Evaluation table for MLP model", cmap='Greys'
                                    )
    
    src.utils.plot_confusion_matrix(cm_tensor=src.utils.eval_model_correctness(eval_data[1][2]),
                                    classesX=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "Macro"],
                                    classesY=["TP", "FP", "FN", "Precision", "Recall", "F1-Score"], fmt='.3f',
                                    normalize=False, title="Evaluation table for CNN model", cmap='Greys'
                                    )
