from load_data import train_dataloader, valid_dataloader, vocabulary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna


def get_optim(trial, model):
    weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)
    lr = trial.suggest_float('lr', 1e-8, 1e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


class BiLSTM_Attention_optuna(nn.Module):
    def __init__(self, trial, vocab_num, embedding_dim, hidden_dim, num_classes):
        super(BiLSTM_Attention_optuna, self).__init__()
        
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


def train_optuna(net, loss_func, optimizer):
        
    net.train()
        
    for label, line, len_seq in train_dataloader:
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


def eval_optuna(net, loss_func, optimizer):
    net.eval()
    acc_val = 0
        
    with torch.no_grad():
        for label, line, len_seq in valid_dataloader:
                
            label = label.to(device)
            line = line.to(device)
                
            label = torch.LongTensor(label)
            line = torch.LongTensor(line)
                
            net_output = net.forward(line)
                
            loss = loss_func(net_output, label)
                
            pred = net_output.argmax(dim=1)
                
            acc_val += (pred == label).float().sum().item()
           
            accuracy = acc_val / len(valid_dataloader.dataset)
            
    return 1 - accuracy


vocab_num = len(vocabulary) #これはlen(vocabulary)inputの系列長ではない。
embedding_dim = 100
hidden_dim = 50
num_classes = 2
loss_func = nn.CrossEntropyLoss()
n_epochs = 20
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def objective(trial):
    model_2 = BiLSTM_Attention_optuna(trial, vocab_num, embedding_dim, hidden_dim, num_classes)
    optimizer = get_optim(trial, model_2) 
    for epoch in range(n_epochs):  
        train_optuna(model_2, loss_func, optimizer)
        error_rate = eval_optuna(model_2, loss_func, optimizer)
        
    return error_rate


# Tuning
TRIAL_SIZE = 10
study = optuna.create_study()
study.optimize(objective, n_trials=TRIAL_SIZE)