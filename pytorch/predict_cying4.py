import os
import numpy as np
import cv2
import torch
import torch.nn as nn

#%%
RESIZE_TO = 100
DROPOUT = 0.1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
torch.manual_seed(42)
np.random.seed(42)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (5, 5), padding=2) # output (n_examples, 16, 100, 100)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2)) # output (n_examples, 16, 50, 50)

        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1) # output (n_examples, 32, 50, 50)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2)) # output (n_examples, 32, 25, 25)

        self.linear1 = nn.Linear(32*25*25, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, 500)
        self.linear2_bn = nn.BatchNorm1d(500)
        self.linear3 = nn.Linear(500,400)
        self.linear3_bn = nn.BatchNorm1d(400)
        self.linear4 = nn.Linear(400,7)
        self.act = torch.relu
    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.drop(self.linear2_bn(self.act(self.linear2(x))))
        x = self.drop(self.linear3_bn(self.act(self.linear3(x))))
        x = torch.sigmoid(self.linear4(x))
        return x
def predict(x):
    images=[]
    for i in x:
        images.append(cv2.resize(cv2.imread(i), (RESIZE_TO, RESIZE_TO)))
    test=np.array(images)
    test=test/255
    test = torch.tensor(test).view(len(test), 3, 100, 100).float().to(device)
    model = CNN().to(device)
    model.load_state_dict(torch.load("model_cying4.pt"))
    y_pred = model(test).detach()
    y_pred = (y_pred.data.cpu()>=0.5).float()
    return y_pred
#%%
x_test = ["/home/ubuntu/finalexam/train/cells_1.png", "/home/ubuntu/finalexam/train/cells_2.png"]  # Dummy image path list placeholder
y_test_pred = predict(x_test)
#%%
'''
assert isinstance(y_test_pred, type(torch.Tensor([1])))  # Checks if your returned y_test_pred is a Torch Tensor
assert y_test_pred.dtype == torch.float  # Checks if your tensor is of type float
assert y_test_pred.device.type == "cpu"  # Checks if your tensor is on CPU
assert y_test_pred.requires_grad is False  # Checks if your tensor is detached from the graph
assert y_test_pred.shape == (len(x_test), 7)  # Checks if its shape is the right one
assert set(list(np.unique(y_test_pred))) in [{0}, {1}, {0, 1}]
print("All tests passed!")
'''





