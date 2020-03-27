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
from model.seqgan import DiscriminatorDatasetReader, collate_fn, load_generator, save_generator, train_generator, evaluate_generator
from sklearn.metrics import roc_auc_score,accuracy_score

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
    train_dataset = DiscriminatorDatasetReader(train_df, device, tokenizer, generator)
    dev_dataset = DiscriminatorDatasetReader(val_df, device, tokenizer, generator)

    BATCH_SIZE = 8
    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)
    dev_iterator = DataLoader(dev_dataset, batch_size=BATCH_SIZE, sampler=dev_sampler, collate_fn=collate_fn)
    # train_data = []
    # for x,y in train_iterator:
    #     train_data.append(x.)

    discriminator = BertDiscriminator.from_pretrained('bert-base-chinese').to(device)
    EPOCH_NUM = 5

    losses = []
    val_losses = []
    rocs = []
    start_epoch = -1

    checkpoint_name='discriminator.pkl'

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in discriminator.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in discriminator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    print(f"current epoch: {start_epoch}")
    print(len(losses))
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays 
    warmup_steps = int(0.5 * len(train_iterator))
    total_steps = len(train_iterator) * EPOCH_NUM - warmup_steps
    print(total_steps, warmup_steps)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,last_epoch=start_epoch)

    # print(generator.gen_output('你好'))


    for i in range(EPOCH_NUM-start_epoch-1):
        print('=' * 50, f"EPOCH {start_epoch+i+1}", '=' * 50)
        tl = train_generator(discriminator, train_iterator, optimizer, scheduler)
        losses.append(tl)
        el,r = evaluate_generator(discriminator, dev_iterator)
        print(f"Train loss {tl}")
        print(f"accuracy {r}")
        print(f"Evaluate loss {el}")
        # val_losses.append(l)
        # rocs.append(r)
        save_generator(discriminator,optimizer,i,checkpoint_name)