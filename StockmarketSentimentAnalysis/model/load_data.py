import pandas as pd
import re
from collections import Counter
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

seed = 1234

# Load data
df = pd.read_csv("../data/stock_data.csv")

# Preprocess table data
df['Sentiment'] = df['Sentiment'].replace(-1,0)

processed_text_lst = []
for text in df['Text']:
    processed_text = re.sub(r'[0-9+]', str(0), text)
    processed_text_lst.append(processed_text)

df['Text'] = pd.DataFrame(processed_text_lst)

# Create dataset
class Mydatasets(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        self.transform = transform

        self.data = df['Text']
        self.label = df['Sentiment']

        self.datanum = len(df)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(out_data)

        return out_data, out_label
    
dataset = Mydatasets()

train_size = int(0.8*len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# Preprocess text data
tokenizer = get_tokenizer('basic_english')

counter = Counter()

for text, label in train_dataset:
    counter.update(tokenizer(text))

vocabulary = vocab(counter,
                   min_freq=12,
                  specials=('<unk>','<PAD>','<BOS>','<EOS>'))

vocabulary.set_default_index(vocabulary['<unk>'])

#print(max([len(text) for text in df['Text']])) # we need to pad sentences shorter than 154 words

def text_transform(text, max_length=154):
    text_ids = [vocabulary[token] for token in tokenizer(text)][:max_length]
    text_ids = [vocabulary['<BOS>']] + text_ids + [vocabulary['<EOS>']]
    return text_ids, len(text_ids) #この時点では長さはバラバラ


def collate_batch(batch):
    label_list, tensor_text_list, len_seq_list = [], [], []
    
    for text, label in batch: #ここごたごたしたけど普通にtextとlabelの順番間違ってただけだった、、、
        label_list.append(label)
        processed_text, len_seq = text_transform(text)
        tensor_text_list.append(torch.tensor(processed_text))
        len_seq_list.append(len_seq)
        
    return torch.tensor(label_list), pad_sequence(tensor_text_list, padding_value=1).T, torch.tensor(len_seq_list)

# Make dataloader
batch_size = 64

train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_batch)

valid_dataloader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=collate_batch)