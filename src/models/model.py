class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,padding_mode = "zeros")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,padding_mode = "zeros")
        self.conv2_drop = nn.Dropout2d() # Kỹ thuật Dropout
        self.fc1 = nn.Linear(64*7*7, 128) # Tính tay qua các bước
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):# x = 28x28
        # Convolution 1:
        x = self.conv1(x) # 28x28
        x = F.max_pool2d(x, kernel_size=2, stride=2) # 2 là 2x2 = kernel size,
        x = F.relu(x)
        # Convolution 2:
        x = self.conv2(x) # 11x11
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) #3x3
        x = F.relu(x)
        #Flatten and FC layers, dropout:
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5) #p = Probability dropout: tắt 50%
        x = self.fc2(x)
        return x #Dùng CrossEntropyLoss
