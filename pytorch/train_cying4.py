import os
import cv2
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#%%
DATA_DIR = "/home/ubuntu/finalexam/train/"
RESIZE_TO = 100, 100
x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read().splitlines()
    y.append(label)
x, y = np.array(x), np.array(y)
##%%
MLB = MultiLabelBinarizer(classes=['red blood cell', 'difficult', 'gametocyte',
                                   'trophozoite','ring', 'schizont', 'leukocyte'])
y = MLB.fit_transform(y)
#%%
print(x.shape, y.shape)
#%%
x=x/255

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2,shuffle=False)
#%%
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms, models
import torch.utils.data as data_utils
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100 * accuracy_score(y.cpu().numpy(), pred_labels)
#%%
x_train,y_train = torch.tensor(x_train).view(len(x_train),3,100,100).float().to(device),torch.tensor(y_train).to(device)
x_test,y_test = torch.tensor(x_test).view(len(x_test),3,100,100).float().to(device),torch.tensor(y_test).to(device)
#%%
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
#%%
LR = 5e-3
N_EPOCHS = 100
BATCH_SIZE = 10
DROPOUT = 0.1
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCELoss()
#%%
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    lr = LR
    if epoch > 25:
        lr = LR * 0.3
    if epoch > 50:
        lr = LR * 0.2
    if epoch > 75:
        lr = LR * 0.1
    loss_train = 0
    model.train()
    for batch in range(len(x_train)//BATCH_SIZE + 1):
        inds = slice(batch*BATCH_SIZE, (batch+1)*BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds].float())
        print(loss)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    with torch.no_grad():
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test.float())
        loss_test = loss.item()
#%%
torch.save(model.state_dict(), "model_cying4.pt")
print('Load model')
model.load_state_dict(torch.load("model_cying4.pt"))
model.eval()
print(model)
#%%
pp=model(x_test)








