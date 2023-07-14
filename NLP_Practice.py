import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import re
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class IMDB(Dataset):
    def __init__(self, data):
        self.data = []
        reviews = data['review'].tolist()
        sentiments = data['sentiment'].tolist()
        reviews = self.get_token2num_maxlen(reviews)
        
        print(len(reviews))
        print(reviews[0])
        for review, sentiment in zip(reviews,sentiments): #將review跟label結合
            if sentiment == 'positive':
                label = 1
            else:
                label = 0

            self.data.append([review,label])

    def __getitem__(self,index):
        datas = torch.tensor(self.data[index][0])
        labels = torch.tensor(self.data[index][1], dtype= torch.int64)
        
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
        
        token_to_num = {data:cnt for cnt,data in enumerate(list(set(token)),1)} #建立dictionary
         
        num = []
        
        for review in reviews:
            review = self.preprocess_text(review)
            tmp = []
            for token in review.split(' '):
                tmp.append(token_to_num[token])  #將文字轉為編號
            num.append(tmp)
            
                
        return num
class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layer):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size 
        self.num_layer = num_layer  #兩層LSTM
        
        self.embedding = nn.Embedding(99427,  self.embedding_dim)   #embbeding_layer
        self.lstm =nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layer, bidirectional = True) #output_size=256(兩層LSTM)
        self.fc = nn.Linear(hidden_size *4 , 20)
        self.fc1 = nn.Linear(20, 2) #輸出維度2
        
    def forward(self, x):
        x = self.embedding(x)
        states, hidden  = self.lstm(x.permute([1,0,2]), None)
        x = torch.cat((states[0], states[-1]), 1)  #取第一個和最後一個做相加(其實不知道為什麼)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(x)
        return x
        
def collate_batch(batch):
    # 抽每一個 batch 的第 0 個(注意順序)
    text = [i[0] for i in batch]
    # 進行 padding
    text = pad_sequence(text, batch_first=True)
    
    # 抽每一個 batch 的第 1 個(注意順序)
    label = [i[1] for i in batch]
    # 把每一個 batch 的答案疊成一個 tensor
    label = torch.stack(label)
    
    return text, label   



def train(train_loader,test_loader, model ,optimizer, criterion):
    epochs = 10
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train = tqdm(train_loader)
        total = 0 
        correct = 0
        model.train()
        for cnt,(data,label) in enumerate(train, 1): #抓資料訓練
            data,label = data.cuda() ,label.cuda()
            optimizer.zero_grad()   #梯度重設
            outputs = model(data)  #向前傳遞
            loss = criterion(outputs.squeeze(-1), label)  #計算LOSS
            _,predict_label = torch.max(outputs, 1)  #獲得預測LABEL
            
            total += label.size(0)
            correct += (predict_label==label).sum().item()   
            train_loss += loss.item()      
            loss.backward()  #計算反向傳播梯度
            optimizer.step() #更新權重

            accu = correct/total  #計算準確率
            #train_acc += (predict_label==label).sum()
            train.set_description(f'train Epoch {epoch}')
            train.set_postfix({'loss':float(train_loss)/cnt,'acc': float(accu)})
           
        total_test = 0
        correct_test = 0    
        model.eval()
        test = tqdm(test_loader)
        test_acc = 0
        for cnt,(data,label) in enumerate(test, 1):
            data,label = data.cuda() ,label.cuda()
            outputs = model(data)
            _,predict_label = torch.max(outputs, 1)
            total_test += label.size(0)
            correct_test += (predict_label==label).sum().item()   
            train_loss += loss.item()
            test_acc = correct_test/total_test
            test.set_description(f'test Epoch {epoch}')
            test.set_postfix({'acc': float(test_acc)})

df = pd.read_csv('/home/P78081057/NCKU_NLP_Practice/IMDB Dataset.csv')

dataset = IMDB(df)
train_set_size = int(len(dataset)*0.8)
test_set_size = len(dataset) - train_set_size
train_set, test_set = data.random_split(dataset, [train_set_size, test_set_size])
train_loader = DataLoader(train_set, batch_size = 128,collate_fn=collate_batch, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_set, batch_size = 128, collate_fn=collate_batch, shuffle = True, num_workers = 0)

model = RNN(embedding_dim = 256, hidden_size = 64, num_layer = 2).cuda()
optimizer = opt.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
train(train_loader, test_loader, model,optimizer,criterion)

#output:
#train Epoch 0: 100%|██████████| 313/313 [00:41<00:00,  7.61it/s, loss=0.663, acc=0.609]
#test Epoch 0: 100%|██████████| 79/79 [00:02<00:00, 33.31it/s, acc=0.616]
#train Epoch 1: 100%|██████████| 313/313 [00:41<00:00,  7.57it/s, loss=0.61, acc=0.725] 
#test Epoch 1: 100%|██████████| 79/79 [00:02<00:00, 33.60it/s, acc=0.734]
#train Epoch 2: 100%|██████████| 313/313 [00:41<00:00,  7.54it/s, loss=0.6, acc=0.746]  
#test Epoch 2: 100%|██████████| 79/79 [00:02<00:00, 32.39it/s, acc=0.76] 
#train Epoch 3: 100%|██████████| 313/313 [00:41<00:00,  7.53it/s, loss=0.583, acc=0.78] 
#test Epoch 3: 100%|██████████| 79/79 [00:02<00:00, 32.99it/s, acc=0.778]
#train Epoch 4: 100%|██████████| 313/313 [00:41<00:00,  7.62it/s, loss=0.574, acc=0.798]
#test Epoch 4: 100%|██████████| 79/79 [00:02<00:00, 32.99it/s, acc=0.798]
#train Epoch 5: 100%|██████████| 313/313 [00:41<00:00,  7.55it/s, loss=0.553, acc=0.842]
#test Epoch 5: 100%|██████████| 79/79 [00:02<00:00, 33.22it/s, acc=0.829]
#train Epoch 6: 100%|██████████| 313/313 [00:41<00:00,  7.59it/s, loss=0.549, acc=0.85] 
#test Epoch 6: 100%|██████████| 79/79 [00:02<00:00, 33.51it/s, acc=0.817]
#train Epoch 7: 100%|██████████| 313/313 [00:41<00:00,  7.52it/s, loss=0.534, acc=0.88] 
#test Epoch 7: 100%|██████████| 79/79 [00:02<00:00, 33.30it/s, acc=0.835]
#train Epoch 8: 100%|██████████| 313/313 [00:41<00:00,  7.58it/s, loss=0.529, acc=0.89] 
#test Epoch 8: 100%|██████████| 79/79 [00:02<00:00, 33.46it/s, acc=0.83] 
#train Epoch 9: 100%|██████████| 313/313 [00:41<00:00,  7.58it/s, loss=0.526, acc=0.896]
#test Epoch 9: 100%|██████████| 79/79 [00:02<00:00, 33.51it/s, acc=0.837]
