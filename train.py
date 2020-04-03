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
from model.seqgan import DiscriminatorDatasetReader, collate_fn, load_generator, save_generator, train_discriminator, evaluate_discriminator, prepaire_D_dataset, prepaire_D_optimizer, prepaire_D_scheduler
from model.seqgan import load_discriminator, prepaire_G_dataset, train_generator

from model.optim import GradualWarmupScheduler

def train_discriminator_with_gen(discriminator, genenrator, real_data):
    genenrator.switch_mode('eval')
    data_itr = real_data.sample(frac=0.4)
    d_steps = 2
    for d_step in range(d_steps):
        # 构建正负样本数据集
        inputs = []
        outputs = []
        for data in data_itr:
            print(data)
            qustion = str(data['question'])
            inputs.append(qustion)
            outputs.append(str(data['answer']))
        corpus = pd.DataFrame({'question':inputs,'answer': outputs}).sample(frac=1)
        BATCH_SIZE = 8
        train_iterator, dev_iterator = prepaire_D_dataset(corpus,genenrator, batch=BATCH_SIZE)
        # 开始训练D
        EPOCH_NUM = 2
        optimizer = prepaire_D_optimizer()
        scheduler = prepaire_D_scheduler(optimizer, EPOCH_NUM, len(train_iterator))
        train_discriminator(discriminator, train_iterator, optimizer, scheduler)
        evaluate_discriminator(discriminator, dev_iterator)

def train_generator_with_discr(generator, discriminator, data_iterator, ignore_padid, tokenizer=None, checkpoint_manager=None):
    g_steps = 2
    for step in range(g_steps):
        BATCH_SIZE = 8
        # optim
        opt = optim.Adam(params=generator.parameters(), lr=generator.learning_rate) # torch.optim.SGD(params=model.parameters(), lr=model_config.learning_rate)
        # scheduler = ReduceLROnPlateau(opt, patience=5)  # Check
        epoch_size = generator.config.epochs//2  # 降低一半
        scheduler = GradualWarmupScheduler(opt, multiplier=8, total_epoch=epoch_size)
        for epoch in range(epoch_size):
            scheduler.step(epoch)
            # 1. generator 生成样本pred[batch_size, seq_len, word_emb]
            dataset = generator.sample(data_iterator)
            print('sample finish')
            # 2. discriminator根据样本[batch_size, seq_len]生成奖励[batch_size,reward]
            D_iterater = prepaire_D_dataset(dataset,generator=None,shuffle=False, default_label=0)
            print('prepaire_D_dataset finish')
            # print(dataset['question'])
            rewards = discriminator.reward(D_iterater)
            print('reward finish')
            # 3. 利用pred[batch_size, seq_len, word_emb], pred[batch_idx][t]表示第batch_idx个句子在0:t-1条件下的log(P(y_t|Y_1:Y_{t-1}))
            #    然后利用公式计算loss
            #    for t in [0:seq_len]:
            #       loss = -pred[batch_idx][t]*Q[batch_idx]
            #    loss/batch_size
            G_iterator = prepaire_G_dataset(dataset,tokenizer,shuffle=False,rewards=rewards)
            print('len of train: ',len(G_iterator))
            loss = train_generator(generator, G_iterator, opt, discriminator, ignore_padid, tokenizer)
            state = {'epoch': epoch + 1,
                'model_state_dict': generator.get_state_dict(),
                'opt_state_dict': opt.state_dict()}
            checkpoint_manager.save_checkpoint(state,'seqGAN.tar')
            print('Loss: ',loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing config.json of generator model")
    parser.add_argument('--discriminator_dir', default='experiments/discriminator_model',
                        help="Directory containing config.json of discriminator model")

    args = parser.parse_args()

    generator, tokenizer, ignore_padid, checkpoint_manager = load_generator(args)

    data_dir = Path(args.data_dir)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    discriminator = load_discriminator(args)

    corpus = pd.read_csv(data_dir / 'Chatbot_data-master/new_corpus.csv',engine='python',encoding="utf8", sep='\t')
    BATCH_SIZE = 2

    EPOCH = 2
    train_iterator, _ = prepaire_G_dataset(corpus.sample(frac=0.25), tokenizer, BATCH_SIZE)
    for epoch in range(EPOCH):
        # g-step
        train_generator_with_discr(generator, discriminator, train_iterator, ignore_padid, tokenizer, checkpoint_manager)
        # d-step
        # train_discriminator(discriminator,)