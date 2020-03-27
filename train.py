from __future__ import absolute_import, division, print_function, unicode_literals
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler
from typing import Tuple, List

from model.generator import Generator
import argparse
from pathlib import Path
from data_utils.utils import Config, CheckpointManager, SummaryManager
import json

from data_utils.vocab_tokenizer import Vocabulary
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW, BertPreTrainedModel,get_linear_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import numpy as np

from model.discriminator import BertDiscriminator

class GeneratorDatasetReader(Dataset):
    def __init__(self, dataframe, device, tokenizer, generator):
        self.device = device
        self.X = []
        self.Y = []
        test_cnt = 0
        global fileindex
        fileindex += 1
        # with open('df'+str(fileindex)+'.cvs','w', encoding='utf8') as f:
        for i, (row) in tqdm(dataframe.iterrows()):
            # try:
            if isinstance(row['question'],str)==False: continue
            pos_sample = self.__truncate_token__([row['question']+'|'+row['answer']], 120, tokenizer)
            # pos_sample = self.__truncate_token__(row['question']+'|'+row['answer'], 120, tokenizer)
            pos_label = [1]
            text = torch.LongTensor(pos_sample)
            tags = torch.LongTensor(pos_label)
            self.X.append(text)
            self.Y.append(tags)
            # print(row['answer'])
            # print(output)

            neg_sample = self.__truncate_token__([row['question']+'|'+output], 120, tokenizer)
            neg_label = [0]
            text = torch.LongTensor(neg_sample)
            tags = torch.LongTensor(neg_label)
            self.X.append(text)
            self.Y.append(tags)
                # except:
                #     print(row['question'])
                # test_cnt +=1
                # if test_cnt>4: break
        print('generate success.')
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        return self.X[index], self.Y[index]
    
    def __truncate_token__(self, row, minsize=50, tokenizer=None):
        tokens = []
        for s in row:
            if isinstance(s,str)!=True:
                break
            # sentence = re.sub(r'[。，?]','',s)
            sentence = s
            if len(sentence)>minsize: sentence = sentence[0:minsize]
            # print(sentence)
            text = tokenizer.encode(sentence)
            tokens += text
        if len(tokens)>512:
            minsize-=10
            tokens = self.__truncate_token__(row, minsize, tokenizer)
        return tokens

def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) \
        -> Tuple[torch.LongTensor, torch.LongTensor]:
    x, y = list(zip(*batch))
    # print(x)
    # print(y)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x.to(device), y.to(device)

def load_generator(args):
    # 载入预训练的生成器
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    data_config = Config(json_path=data_dir / 'config.json')
    model_config = Config(json_path=model_dir / 'config.json')

    checkpoint_manager = CheckpointManager(model_dir) # experiments/base_model
    checkpoint = checkpoint_manager.load_checkpoint('best.tar')

    with open(data_config.token2idx_vocab, mode='rb') as io:
        token2idx_vocab = json.load(io)
        print("token2idx_vocab: ", token2idx_vocab)
    vocab = Vocabulary(token2idx = token2idx_vocab)
    model_config.vocab_size = len(vocab.token2idx)

    return Generator(model_config, vocab, checkpoint['model_state_dict'])

def train_generator(gen, data_itr):
    data_itr = data_itr.sample(frac=0.5)
    g_steps = 10
    genenrator.switch_mode('train')
    for step in range(g_steps):
        for item in data_itr:
            print(item)
            return
        # gen.gen_output()

def train_discriminator(discriminator, genenrator, real_data):
    genenrator.switch_mode('eval')
    data_itr = real_data.sample(frac=0.4)
    d_steps = 10
    for d_step in range(d_steps):
        # 根据真实数据生成负样本
        inputs = []
        outputs = []
        for data in data_itr:
            print(data)
            qustion = str(data['question'])
            inputs.append(qustion)
            output = genenrator.gen_output(qustion)
            outputs.append(output)
        # 开始训练D
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing config.json of model")

    args = parser.parse_args()

    generator = load_generator(args)
    generator.switch_mode()

    data_dir = Path(args.data_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 创建正负样本
    corpus = pd.read_csv(data_dir / 'Chatbot_data-master/new_corpus.csv',engine='python',encoding="utf8", sep='\t')
    train_df, val_df = train_test_split(corpus, test_size=0.05)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # train_dataset = DiscriminatorDatasetReader(train_df, device, tokenizer, generator)
    # dev_dataset = DiscriminatorDatasetReader(val_df, device, tokenizer, generator)

    # BATCH_SIZE = 4
    # train_sampler = RandomSampler(train_dataset)
    # dev_sampler = RandomSampler(dev_dataset)
    # train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
    # dev_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE, sampler=dev_sampler, collate_fn=collate_fn)

    EPOCH = 2
    for epoch in range(EPOCH):
        # g-step
        train_generator(generator, train_df)