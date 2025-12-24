import torch
from src.utils import add_confusion_matrix

# Train Loop
def train(model, loaders, epoch, optimizer, loss_fn, device):
    running_loss = []
    total = len(loaders['train'].dataset)
    # Hàm len(Tensor) luôn trả tra kích thước của dim[0]
    model.train() # bật trạng thái train: để dropout hoặc batch normalization
    for batch_idx, (data, target) in enumerate (loaders['train']):
        # data là tensor 4 chiều: images[64, 1, 28, 28]
        # target là tensor 1 chiều labels[64]
        data, target  = data.to(device), target.to(device)
        num_data = batch_idx*len(data) # len(data) trả ra số batch_size = 64
        # tối ưu
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0: # cứ mỗi 50 batches thì in ra
            print(f'Train epoch: {epoch} [{num_data}/{total}] \t {loss.item():.6f}')
            running_loss.append(loss.item())
    return running_loss

def valid(model, loaders, device, loss_fn, confusion_matrix):
    model.eval() # tắt trạng thái train: mô hình hoàn chỉnh
    # bây giờ mình đếm coi correct, loss biến động ntp, rồi in ra.
    accuracy = []
    correct = 0
    total = len(loaders['valid'].dataset)  # chiều dài thằng valid data
    with torch.no_grad():
        for data, target in loaders['valid']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            confusion_matrix += add_confusion_matrix(output, target)
            correct += pred.eq(target.view_as(pred)).sum().item() #đánh dấu bỏ
    accuracy.append(float(correct)/float(total))
    return accuracy