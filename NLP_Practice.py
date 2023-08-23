import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score, recall_score, precision_score
import os
import re
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class IMDB(Dataset):
    def __init__(self, data, max_len =500):
        self.data = []
        reviews = data['review'].tolist()
        sentiments = data['sentiment'].tolist()
        reviews, max_len = self.get_token2num_maxlen(reviews)
        max_len = 500
        
        for review, sentiment in zip(reviews,sentiments):
            if max_len > len(review):
                padding_cnt = max_len - len(review)
                review += padding_cnt * [0]
            else:
                review = review[:max_len]

            if sentiment == 'positive':
                label = 1
            else:
                label = 0

            self.data.append([review,label])

    def __getitem__(self,index):
        datas = torch.tensor(self.data[index][0])
        labels = torch.tensor(self.data[index][1], dtype= torch.float32).unsqueeze(-1)
        
        return datas, labels
    
    def __len__(self):
    
        return len(self.data)
        
    def preprocess_text(self,sentence):
        #移除html tag
        sentence = re.sub(r'<[^>]+>',' ',sentence)
        #刪除標點符號與數字
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        #刪除單個英文單字
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        #刪除多個空格
        sentence = re.sub(r'\s+', ' ', sentence)
    
        return sentence.lower()
    
    
    def get_token2num_maxlen(self, reviews,enable=True):
        token = []
        for review in reviews:
            review = self.preprocess_text(review)
            token += review.split(' ')
        
        token_to_num = {data:cnt for cnt,data in enumerate(list(set(token)),1)}
         
        num = []
        max_len = 0 
        for review in reviews:
            review = self.preprocess_text(review)
            tmp = []
            for token in review.split(' '):
                tmp.append(token_to_num[token])
                
            if len(tmp) > max_len:
                max_len = len(tmp)
            num.append(tmp)
            
                
        return num, max_len
        
       
        
class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layer):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        
        self.embedding = nn.Embedding(99422,  self.embedding_dim)
        self.lstm =nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layer, bidirectional = True)
        self.fc = nn.Linear(hidden_size * 4, 20)
        self.fc1 = nn.Linear(20,1)
        
    def forward(self, x):
        x = self.embedding(x)
        states, hidden  = self.lstm(x.permute([1,0,2]), None)
        x = torch.cat((states[0], states[-1]), 1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(x)
        return x
        
   



def train(train_loader,test_loader,val_loader, model ,optimizer, criterion):
    epochs = 20
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train_f1 = 0
        train_recall = 0
        train_precision = 0
        train_samples = 0
        train = tqdm(train_loader)
        total = 0 
        correct = 0
        model.train()
        for cnt,(data,label) in enumerate(train, 1): #抓資料訓練
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()  # 梯度重设
            outputs = model(data)  # 向前传递
            loss = criterion(outputs, label)  # 计算LOSS
            _, predict_label = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (predict_label == label).sum().item()
            train_f1 += f1_score(label.cpu(), predict_label.cpu())
            train_recall += recall_score(label.cpu(), predict_label.cpu())
            train_precision += precision_score(label.cpu(), predict_label.cpu(), zero_division=1)
            train_samples += len(label)
            
            train.set_description(f'train Epoch {epoch}')
            train.set_postfix({
                'loss': float(train_loss) / cnt,
                'acc': float(train_acc) / train_samples,
                'f1': float(train_f1) / cnt,
                'recall': float(train_recall) / cnt,
                'precision': float(train_precision) / cnt
            })
           
        total_test = 0
        correct_test = 0  
        test_loss = 0
        test_acc = 0
        test_f1 = 0
        test_recall = 0
        test_precision = 0
        test_samples = 0
        model.eval()
        test = tqdm(val_loader)
        test_acc = 0
        for cnt,(data,label) in enumerate(test, 1):
            data,label = data.cuda() ,label.cuda()
            outputs = model(data)
            _,predict_label = torch.max(outputs, 1)

            test_loss += loss.item()
            test_acc += (predict_label == label).sum().item()
            test_f1 += f1_score(label.cpu(), predict_label.cpu())
            test_recall += recall_score(label.cpu(), predict_label.cpu())
            test_precision += precision_score(label.cpu(), predict_label.cpu())
            test_samples += len(label)
            
            test.set_description(f'valid Epoch {epoch}')
            test.set_postfix({
                'loss': float(test_loss) / cnt,
                'acc': float(test_acc) / test_samples,
                'f1': float(test_f1) / cnt,
                'recall': float(test_recall) / cnt,
                'precision': float(test_precision) / cnt
            })

    val_loss = 0
    val_acc = 0
    val_f1 = 0
    val_recall = 0
    val_precision = 0
    val_samples = 0
    model.eval()
    val = tqdm(test_loader)
    test_acc = 0
    for cnt,(data,label) in enumerate(val, 1):
        data,label = data.cuda() ,label.cuda()
        outputs = model(data)
        _,predict_label = torch.max(outputs, 1)

        val_loss += loss.item()
        val_acc += (predict_label == label).sum().item()
        val_f1 += f1_score(label.cpu(), predict_label.cpu())
        val_recall += recall_score(label.cpu(), predict_label.cpu())
        val_precision += precision_score(label.cpu(), predict_label.cpu())
        val_samples += len(label)
            
        val.set_description(f'test:')
        val.set_postfix({
            'loss': float(val_loss) / cnt,
            'acc': float(val_acc) / val_samples,
            'f1': float(val_f1) / cnt,
            'recall': float(val_recall) / cnt,
            'precision': float(val_precision) / cnt
         })        



           


df = pd.read_csv('/home/P78081057/NCKU_NLP_Practice/IMDB Dataset1.csv')

dataset = IMDB(df)
train_set_size = int(len(dataset)*0.8)
test_set_size = len(dataset) - train_set_size
train_set, test_set = data.random_split(dataset, [train_set_size, test_set_size])
train_set_size = int(len(train_set)*0.8)
val_set_size = len(train_set) - train_set_size
train_set, val_set = data.random_split(train_set, [train_set_size, val_set_size])
train_loader = DataLoader(train_set, batch_size = 128,shuffle = True, num_workers = 0)
test_loader = DataLoader(test_set, batch_size = 128, shuffle = True, num_workers = 0)
val_loader = DataLoader(val_set, batch_size = 128, shuffle = True, num_workers = 0)

model = RNN(embedding_dim = 256, hidden_size = 64, num_layer = 2).cuda()
optimizer = opt.Adam(model.parameters())
criterion = nn.BCELoss()
train(train_loader, test_loader,val_loader, model,optimizer,criterion)

