from __future__ import absolute_import, division, print_function, unicode_literals
from torch.utils.data import Dataset, RandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, List

from model.generator import Generator
import argparse
from pathlib import Path
from data_utils.utils import Config, CheckpointManager, SummaryManager

from transformers import AdamW, get_linear_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup
import pandas as pd

from model.discriminator import BertDiscriminator
from model.seqgan import DiscriminatorDatasetReader, collate_fn, load_generator, save_generator, train_discriminator, evaluate_discriminator, save_discriminator, prepaire_D_dataset, prepaire_D_optimizer, prepaire_D_scheduler, load_discriminator
from sklearn.metrics import roc_auc_score,accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data_in', help="Directory containing config.json of data")
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing config.json of model")

    args = parser.parse_args()

    generator, _, _, _ = load_generator(args)
    generator.switch_mode()

    data_dir = Path(args.data_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 创建正负样本
    corpus = pd.read_csv(data_dir / 'Chatbot_data-master/new_corpus.csv',engine='python',encoding="utf8", sep='\t')
    BATCH_SIZE = 8
    train_iterator, dev_iterator = prepaire_D_dataset(corpus, generator, BATCH_SIZE)
    # train_data = []
    # for x,y in train_iterator:
    #     train_data.append(x.)

    EPOCH_NUM = 15

    losses = []
    val_losses = []
    rocs = []
    start_epoch = -1

    checkpoint_name='discriminator.pkl'

    optimizer = prepaire_D_optimizer()

    print(f"current epoch: {start_epoch}")
    print(len(losses))
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays 
    scheduler = prepaire_D_scheduler(optimizer, EPOCH_NUM, len(train_iterator))

    # print(generator.gen_output('你好'))
    discriminator = load_discriminator()
    #  loss
    # 3     -0.38953
    # 4     -0.389512736511
    # 5     -0.38947504
    #       -0.38959519
    #       -0.38939564
    #       -0.389323457
    #       -0.3896472
    #       -0.389464639
    #       -0.3895617
    #       -0.3896406
    #       -0.38955125
    #       -0.3892908
    history_acc = []
    last_acc = 0
    discriminator_dir = args.model_dir + '/../discriminator_model/'
    for i in range(EPOCH_NUM-start_epoch-1):
        print('=' * 50, f"EPOCH {start_epoch+i+1}", '=' * 50)
        tl = train_discriminator(discriminator, train_iterator, optimizer, scheduler)
        losses.append(tl)
        el,r = evaluate_discriminator(discriminator, dev_iterator)
        print(f"Train loss {tl}")
        print(f"accuracy {r}")
        print(f"Evaluate loss {el}")
        history_acc.append(r)
        # val_losses.append(l)
        # rocs.append(r)
        if last_acc<r:
            save_discriminator(discriminator,discriminator_dir)
    print('acc: ', history_acc)