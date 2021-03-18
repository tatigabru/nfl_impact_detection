import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

import time
from torch.utils.data import Dataset


from torch.utils.data import DataLoader

from config import *
# augmentation, rescaling
# focal loss,  regularization, hyperparameter search,
SIZE = 15
CENTER = 7
H = 5000
NUM_CLASS = 2
BATCH_SIZE = 32
N_EPOCHS = 50
LR = 0.0005
SAMPLE = 10000000
WEIGHT = 2
model_fp = f'../../data/ckpts/1dcnn/cnn_best_track_0.2_wh2_win{SIZE}_center{CENTER}_wdecay-5_norm.ckpt'
data_fp = cache_dir + f'./xywh_2_win{SIZE}_track_0.2_center{CENTER}_norm.npz'
#data_fp = f'./xy_2_win{SIZE}_v2.npz'
print(SIZE, H, NUM_CLASS, BATCH_SIZE, N_EPOCHS, LR, SAMPLE)



"""
conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)


#drop_out = Dropout(0.2)(conv1)
conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

#drop_out = Dropout(0.2)(conv2)
conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

full = keras.layers.GlobalAveragePooling2D()(conv3)
out = keras.layers.Dense(nb_classes, activation='softmax')(full)
"""
class CNN(nn.Module):
    def __init__(self, c1=256, c2=64, c3=16, h1=SIZE - 6, p=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(8, c1, 3)
        self.conv2 = nn.Conv1d(c1, c2, 3)
        self.conv3 = nn.Conv1d(c2, c3, 3)
        self.conv4 = nn.Conv1d(c3, 1, 1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.fc = nn.Linear(h1, 2)
        self.dropout = nn.Dropout(p=p)
        self.h1 = h1
        #self.fc2 = nn.Linear(h1, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        #x = F.avg_pool1d(x, kernel_size=5)
        x = self.dropout(F.relu(self.conv4(x)))
        x = x.view(-1, self.h1)
        #x = self.fc2(F.relu(self.fc1(x)))
        x = self.fc(x)
        return x

class CNN_(nn.Module):
    def __init__(self, c1=256, c2=64, c3=16, h1=SIZE - 6, p=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(4, c1, 3)
        self.conv2 = nn.Conv1d(c1, c2, 3)
        self.conv3 = nn.Conv1d(c2, c3, 3)
        self.conv4 = nn.Conv1d(c3, 1, 1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.fc = nn.Linear(h1, 2)
        self.dropout = nn.Dropout(p=p)
        self.h1 = h1
        self.c3 = c3
        self.c2 = c2
        #self.fc2 = nn.Linear(27, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = F.max_pool1d(x, kernel_size=SIZE-6)
        x = x.view(-1, self.c3)
        #x = self.fc2(F.relu(self.fc1(x)))
        x = self.fc(x)
        return x


class CNN_(nn.Module):
    def __init__(self, c1=256, c2=32, c3=16, h1=SIZE - 6, p=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(8, c1, 3)
        self.conv2 = nn.Conv1d(c1, c2, 3)
        self.conv3 = nn.Conv1d(c2, c3, 3)
        self.conv4 = nn.Conv1d(c3, 1, 1)
        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)
        self.h1 = SIZE - 6 + c2
        self.fc = nn.Linear(self.h1, 2)
        self.dropout = nn.Dropout(p=p)
        #self.fc2 = nn.Linear(27, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = F.max_pool1d(x, kernel_size=SIZE-4)
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.dropout(F.relu(self.conv4(x)))
        x = torch.cat([x.view(-1, SIZE - 6), x1.view(-1, 32)], dim=1)
        #x = self.fc2(F.relu(self.fc1(x)))
        x = self.fc(x)
        return x


class CNN_(nn.Module):
    def __init__(self, c1=256, c2=32, c3=16, h1=SIZE - 6, p=0.0):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(8, c1, 3)
        self.conv2 = nn.Conv1d(c1, c2, 3)
        self.conv3 = nn.Conv1d(c2, c3, 3)

        self.bn1 = nn.BatchNorm1d(c1)
        self.bn2 = nn.BatchNorm1d(c2)
        self.bn3 = nn.BatchNorm1d(c3)

        self.fc = nn.Linear(c2 + c3, 2)
        #self.dropout = nn.Dropout(p=p)
        self.h1 = h1
        self.c3 = c3
        self.c2 = c2
        self.c1 = c1
        #self.fc2 = nn.Linear(27, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = F.max_pool1d(x, kernel_size=SIZE - 4)
        x = F.relu(self.bn3(self.conv3(x)))
        x2 = F.max_pool1d(x, kernel_size=SIZE -6)
        x = torch.cat([x1.view(-1, self.c2), x2.view(-1, self.c3)], dim=1)
        x = self.fc(x)
        return x


class MyDataset(Dataset):
    def __init__(self, X, y=None, shuffle=False):
        super().__init__()
        """
        X2, y2 = np.stack([-X[:, 0], X[:, 1], -X[:, 2], X[:, 3]], axis=1), y

        X_, y_ = np.concatenate([X, X2]), np.concatenate([y, y2])
        """
        if y is None:
            y = np.array([0] * len(X))
        if shuffle:
            idxs = list(range(min(SAMPLE, len(X))))
            np.random.shuffle(idxs)
            X_ = X[idxs]
            y_ = y[idxs]
        else:
            X_, y_ = X, y

        self.data = (np.transpose(X_, (0, 2, 1)), y_)


    def __getitem__(self, index: int):
        return torch.tensor(self.data[0][index]/10, dtype=torch.float),\
                                    torch.tensor(self.data[1][index], dtype=torch.int64)

    def __len__(self) -> int:
        return self.data[0].shape[0]


def predict(dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN()
    model.to(device)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    output = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            with torch.no_grad():
                cur_output = model(data)
                cur_output_ = F.softmax(cur_output)[:, 1].detach().cpu().numpy()
                output.append(cur_output_)
    return np.concatenate(output)


def test(data_, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0
    acc = 0
    tp, preds, targs = 0, 0, 0
    dataloader = DataLoader(data_, batch_size=2048)
    for data, impact in dataloader:
        data, impact = data.to(device), impact.to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, impact)
            test_loss += loss.item()
            tp += torch.eq(output.argmax(1) + impact, 2).sum().item()
            preds += output.argmax(1).sum().item()
            targs += impact.sum().item()
    print('Prec:', tp / (preds + 0.00001))
    print('Rec:', tp / targs)
    return test_loss / len(data_), 2 * tp / (preds + targs)


def train_epochs():
    def train_func(train_dataset):
        model.train()
        # Train the model
        train_loss = 0
        tp, preds, targs = 0, 0, 0
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        for i, (data, impact) in enumerate(dataloader):
            optimizer.zero_grad()
            data, impact = data.to(device), impact.to(device)
            output = model(data)
            loss = criterion(output, impact)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            tp += torch.eq(output.argmax(1) + impact, 2).sum().item()
            preds += output.argmax(1).sum().item()
            targs += impact.sum().item()
        #with torch.no_grad():

            #print(torch.max(output[:, 1]).item())
            # print(impact)
            # print(model.fc1.weight)
            # print(model.fc2.weight)
            # print(torch.max(model.fc1.weight.grad))
            # print(torch.max(model.fc2.weight.grad))
        # Adjust the learning rate

        scheduler.step(train_loss)
        # print('Prec:', tp/preds)
        # print('Rec:', tp/targs)
        return train_loss / len(train_dataset), 2 * tp / (preds + targs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    print(model)



    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, WEIGHT], dtype=torch.float)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=10e-5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.0001)

    saved = np.load(data_fp)
    X_train, y_train, X_valid, y_valid = saved['X_train'], saved['y_train'], saved['X_valid'], saved['y_valid']
    print('Train example', X_train[0])
    print('Valid example', X_valid[0])
    print('Training on number of datapoints', len(X_train))
    print('Percent of positives in train', sum(y_train)/len(y_train) *100, '%')
    print('Percent of positives in valid', sum(y_valid) / len(y_valid) * 100, '%')

    train_dataset = MyDataset(X_train, y_train.flatten(), shuffle=True)
    valid_dataset = MyDataset(X_valid, y_valid.flatten(), shuffle=False)

    best_valid_loss = float('inf')
    best_f1 = 0
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_f1 = train_func(train_dataset)
        valid_loss, valid_f1 = test(valid_dataset, model, criterion)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60
        if valid_f1 > best_f1:
            torch.save(model.state_dict(), model_fp)
            best_f1 = valid_f1
           # best_valid_loss = valid_loss
        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.7f}(train)\t|\tF1: {train_f1 * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.7f}(valid)\t|\tF1: {valid_f1 * 100:.1f}%(valid)')


    print('Checking the results of test dataset...')
    model = CNN()
    model.to(device)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    test_loss, test_f1 = test(valid_dataset, model, criterion)
    print(f'\tLoss: {test_loss:.7f}(test)\t|\tAcc: {test_f1 * 100:.1f}%(test)')


def test_best():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saved = np.load(data_fp)

    X_train, y_train, X_valid, y_valid = saved['X_train'], saved['y_train'], saved['X_valid'], saved['y_valid']

    #train_dataset = MyDataset(X_train, y_train.flatten(), shuffle=True)
    valid_dataset = MyDataset(X_valid, y_valid.flatten(), shuffle=False)

    model = CNN()
    model.to(device)
    model.load_state_dict(torch.load(model_fp))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, WEIGHT], dtype=torch.float)).to(device)
    test_loss, test_f1 = test(valid_dataset, model, criterion)
    print(f'\tLoss: {test_loss:.7f}(test)\t|\tAcc: {test_f1 * 100:.1f}%(test)')


if __name__ == '__main__':
    train_epochs()
    #test_best()
