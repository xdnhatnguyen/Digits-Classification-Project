import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_graph(list_1, list_2, name_1, name_2, title):
    # 1. Setup the main plot (Left Y-axis for Loss)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # X-axis for loss (steps)
    x_1 = list(range(len(list_1)))
    
    # Plot Loss (Red line)
    ax1.set_xlabel(name_1)
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(x_1, list_1, color='tab:red', label=name_1)
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # 2. Setup the secondary axis (Right Y-axis for Accuracy)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # X-axis for accuracy (Fixed alignment to end of epoch)
    steps_per_epoch = len(list_1) / len(list_2)
    x_2 = [i * steps_per_epoch for i in range(len(list_2))]
    
    # Plot Accuracy (Blue line)
    ax2.set_ylabel(name_2, color='tab:blue')  
    ax2.plot(x_2, list_2, color='tab:blue', marker='o', label=name_2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 3. Final touches
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def add_confusion_matrix(pred, target, epsilon = 1e7):
    if pred.ndim > 1:
        pred = torch.argmax(pred, dim=1)

    target = target.view(-1)
    pred = pred.view(-1)

    num_classes = 10

    indices = target * num_classes + pred

    confusion_matrix = torch.bincount(
        indices,
        minlength = num_classes ** 2
    ).reshape(num_classes, num_classes).float()

    return confusion_matrix

def plot_confusion_matrix(cm_tensor, classesX=None, classesY=None, normalize=False, fmt = '.0f', title='Confusion Matrix', cmap='Blues'):

    cm = cm_tensor.detach().cpu().numpy()

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
        print("Đang hiển thị chế độ: PHẦN TRĂM (%)")
        fmt = '.5f'
    else:
        print("Đang hiển thị chế độ: SỐ LƯỢNG (Counts)")

    plt.figure(figsize=(10, 8))

    if classesX is None:
        classesX = [str(i) for i in range(cm.shape[0])]

    if classesY is None:
        classesY = classesX

    sns.heatmap(cm,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                xticklabels=classesY,
                yticklabels=classesX
               )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Nhãn Thực Tế (True)', fontsize=12)
    plt.xlabel('Dự Đoán (Predicted)', fontsize=12)
    plt.show()

def eval_model_correctness(confusion_matrix):
    tp = []
    fp = []
    fn = []
    precision = []
    recall = []
    f1_score = []
    for i in range(10):
        tp.append(confusion_matrix[i][i])
        fp.append(confusion_matrix[:,i].sum().float() - confusion_matrix[i][i].float())
        fn.append(confusion_matrix[i,:].sum().float() - confusion_matrix[i][i].float())
        precision.append((tp[i])/(tp[i] + fp[i]))
        recall.append((tp[i])/(tp[i] + fn[i]))
        f1_score.append(2*precision[i]*recall[i] / (precision[i] + recall[i]))
    tp.append(sum(tp).float() / 10)
    fp.append(sum(fp).float() / 10)
    fn.append(sum(fn).float() / 10)
    precision.append(sum(precision).float() / 10)
    recall.append(sum(recall).float() / 10)
    f1_score.append(sum(f1_score).float() / 10)
    return torch.stack(
        (torch.Tensor(tp),
         torch.Tensor(fp),
         torch.Tensor(fn),
         torch.Tensor(precision),
         torch.Tensor(recall),
         torch.Tensor(f1_score)
         ),
        dim = 1
    )
