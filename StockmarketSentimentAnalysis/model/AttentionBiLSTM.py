from load_data import train_dataloader, valid_dataloader, vocabulary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

seed = 1234

# Attention BiLSTM model
class BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_num, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_num, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim*2, num_classes) #今回は二値分類なので
        
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, hidden_dim*2, 1) #hidden(batch, hidden_num*2, num_layer=1)
        attention_weights = torch.bmm(lstm_output, hidden).squeeze(2) #num_layer=1を削除
        soft_attention_weights = F.softmax(attention_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attention_weights.unsqueeze(2)).squeeze(2)
        return context #, soft_attention_weights.data.numpy()
    
    def forward(self, x):
        embedded_input = self.embedding(x) #embedded_inputの次元確認。もしかしたらbatchが一番最初に来てないかもしれない。
        
        hidden_state = torch.zeros(1*2, len(x), hidden_dim) #(num_layers=1*num_directions=2, batch_size, hidden_dim)
        cell_state = torch.zeros(1*2, len(x), hidden_dim)
        
        lstm_output, (final_hidden_state, _) = self.bilstm(embedded_input, (hidden_state, cell_state))
        attention_output = self.attention_net(lstm_output, final_hidden_state)
        output = self.out(attention_output)
        return output
    
# train function
def train(net, loss_func, optimizer, n_epochs):
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []
        
        net.train()
        n_train = 0
        acc_train = 0
        
        for label, line, len_seq in train_dataloader:
            n_train += label.size()[0]
            net.zero_grad()
            
            label = label.to(device)
            line = line.to(device)
            len_seq = len_seq.to(device) #やっぱこれいらんかも今回は
            
            line = torch.LongTensor(line)
            label = torch.LongTensor(label)
            
            net_output = net.forward(line)
            loss = loss_func(net_output, label)
            
            loss.backward()
            
            optimizer.step()
            
            pred = net_output.argmax(dim=1)
            
            acc_train += (pred == label).float().sum().item()
            
            losses_train.append(loss.tolist())
            
        
        net.eval()
        n_val = 0
        acc_val = 0
        
        with torch.no_grad():
            for label, line, len_seq in valid_dataloader:
                n_val += label.size()[0]
                
                label = label.to(device)
                line = line.to(device)
                
                label = torch.LongTensor(label)
                line = torch.LongTensor(line)
                
                net_output = net.forward(line)
                
                loss = loss_func(net_output, label)
                
                pred = net_output.argmax(dim=1)
                
                acc_val += (pred == label).float().sum().item()
                
                losses_valid.append(loss.tolist())
                
        print("EPOCH: {}, Train [Loss: {:.3f}, Accuracy: {:.3f}], Validation [Loss: {:.3f}, Accuracy: {:.3f}]".format(
            epoch+1,
            np.mean(losses_train),
            acc_train/n_train,
            np.mean(losses_valid),
            acc_val/n_val))

# RUN
# Parameters for model
vocab_num = len(vocabulary) #これはlen(vocabulary)inputの系列長ではない。
embedding_dim = 100
hidden_dim = 50
num_classes = 2

# Parameters for Adam
lr = 0.0002
weight_decay = 4e-7

# Make instance of the model class
model_1 = BiLSTM_Attention(vocab_num, embedding_dim, hidden_dim, num_classes)

# Parameters for train function
optimizer = optim.Adam(model_1.parameters(), lr=lr, weight_decay=weight_decay)
loss_func = nn.CrossEntropyLoss()
n_epochs = 50
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train(model_1, loss_func, optimizer, n_epochs)