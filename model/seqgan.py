from model.generator import Generator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from typing import Tuple, List
from model.discriminator import BertDiscriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from data_utils.utils import Config, CheckpointManager, SummaryManager
from data_utils.vocab_tokenizer import Vocabulary
from tqdm import tqdm
import json
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW, BertPreTrainedModel,get_linear_schedule_with_warmup,get_cosine_with_hard_restarts_schedule_with_warmup

from data_utils.chatbot_dataset import ChatbotDataset
from data_utils.vocab_tokenizer import Tokenizer, Vocabulary, keras_pad_fn, mecab_token_pos_flat_fn
from metric import acc
from evaluate import decoding_to_pair, decoding_to_str

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class DiscriminatorDatasetReader(Dataset):
    def __init__(self, dataframe, device, tokenizer, generator=None, default_label=1):
        self.device = device
        self.X = []
        self.Y = []
        test_cnt = 0
        # global fileindex
        # fileindex += 1
        # with open('df'+str(fileindex)+'.cvs','w', encoding='utf8') as f:
        for i, (row) in tqdm(dataframe.iterrows()):
            # try:
            if isinstance(row['question'],str)==False: continue
            # print('Q: ', row['question'])
            if generator is not None:
                output = generator.gen_output(row['question'])
                if output==row['answer']: continue

                neg_sample = self.__truncate_token__([row['question']+'|'+output], 120, tokenizer)
                neg_label = [0]
                text = torch.LongTensor(neg_sample)
                tags = torch.LongTensor(neg_label)
                self.X.append(text)
                self.Y.append(tags)
            # print('A: ', row['answer'])
            # f.write(row['question']+'|'+row['answer']+'\t1\n')
            # f.write(row['question']+'|'+output+'\t0\n')

            pos_sample = self.__truncate_token__([row['question']+'|'+row['answer']], 120, tokenizer)
            # pos_sample = self.__truncate_token__(row['question']+'|'+row['answer'], 120, tokenizer)
            pos_label = [default_label]
            text = torch.LongTensor(pos_sample)
            tags = torch.LongTensor(pos_label)
            self.X.append(text)
            self.Y.append(tags)
            # print(row['answer'])
            # print(output)

            
                # except:
                #     print(row['question'])
            # test_cnt +=1
            # if test_cnt>4: break
        # print('generate success.')
    
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

    tokenizer = Tokenizer(vocab=vocab, split_fn=mecab_token_pos_flat_fn, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)
    return Generator(model_config, vocab, checkpoint['model_state_dict']), tokenizer, vocab.PAD_ID, checkpoint_manager

def collate_fn(batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) \
        -> Tuple[torch.LongTensor, torch.LongTensor]:
    x, y = list(zip(*batch))
    # print(x)
    # print(y)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x.to(device), y.to(device)

def save_generator(model,optimizer,start_epoch,filename):
    all_state = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': start_epoch,'train_loss':losses,'val_loss':val_losses,'auc':rocs}
    torch.save(all_state, filename)

def save_discriminator(model, path):
    model.save_pretrained(path)

def train_discriminator(model, iterator, optimizer, scheduler, use_part=False):
    model.train()
    total_loss = 0
    try:
        with tqdm(iterator) as t:
            for x, y in t:
                optimizer.zero_grad()
                mask = (x != 0).float()
        #         print(x.shape,y.shape)
                loss, outputs = model(x, attention_mask=mask, labels=y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
    except KeyboardInterrupt:
        t.close()
    avg_loss = total_loss / len(iterator)
    return avg_loss

def evaluate_discriminator(model, iterator):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in tqdm(iterator):
            mask = (x != 0).float()
            loss, outputs = model(x, attention_mask=mask, labels=y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
#     print(true.shape,pred.shape)
#     for i, name in enumerate(similar_classes):
    
    pred_indexes = np.argmax(pred,axis=1)
    true_indexes = np.argmax(true,axis=1)
#     print(pred)
#     print(true)
    denominator = len(pred_indexes)
    numerator = np.sum(pred_indexes!=true_indexes)
    acc = 1 - numerator/denominator
    
    avg_loss = total_loss / len(iterator)
    return avg_loss,acc

def load_discriminator(args=None):
    if args is None: return BertDiscriminator.from_pretrained('bert-base-chinese').to(device)
    # 载入预训练的生成器
    model_path = args.discriminator_dir
    model = BertDiscriminator.from_pretrained('bert-base-chinese').to(device)
    return model.from_pretrained(model_path).to(device)

def prepaire_D_dataset(corpus, generator=None, batch = 8, shuffle=True, default_label=1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    if shuffle==True:
        train_df, val_df = train_test_split(corpus, test_size=0.05)
        train_dataset = DiscriminatorDatasetReader(train_df.sample(frac=0.4), device, tokenizer, generator)
        dev_dataset = DiscriminatorDatasetReader(val_df.sample(frac=0.5), device, tokenizer, generator)

        train_sampler = RandomSampler(train_dataset)
        dev_sampler = RandomSampler(dev_dataset)
        train_iterator = DataLoader(train_dataset, batch_size=batch, sampler=train_sampler, collate_fn=collate_fn)
        dev_iterator = DataLoader(dev_dataset, batch_size=batch, sampler=dev_sampler, collate_fn=collate_fn)
        return train_iterator, dev_iterator
    else:
        all_dataset = DiscriminatorDatasetReader(corpus, device, tokenizer, generator, default_label=default_label)
        all_sampler = SequentialSampler(all_dataset)
        all_iterator = DataLoader(all_dataset, batch_size=batch, sampler=all_sampler, collate_fn=collate_fn)
        return all_iterator


def prepaire_G_dataset(corpus, tokenizer, batch = 8, shuffle=True, rewards = None):
    if rewards is None:
        train_df, val_df = train_test_split(corpus, test_size=0.05)
        tr_ds = ChatbotDataset(tokenizer.list_of_string_to_arr_of_pad_token_ids, corpus=train_df)
        tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True, drop_last=False)

        val_ds = ChatbotDataset(tokenizer.list_of_string_to_arr_of_pad_token_ids, corpus=val_df)
        val_dl = DataLoader(val_ds, batch_size=batch, shuffle=True, drop_last=False)

        return tr_dl, val_dl
    else:
        corpus.insert(0,'reward',rewards)
        all_ds = ChatbotDataset(tokenizer.list_of_string_to_arr_of_pad_token_ids, corpus=corpus)
        all_dl = DataLoader(all_ds, batch_size=batch, shuffle=shuffle, drop_last=False)
        return all_dl

def prepaire_D_optimizer():
    discriminator = BertDiscriminator.from_pretrained('bert-base-chinese').to(device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in discriminator.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in discriminator.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

def prepaire_D_scheduler(optimizer, epoch_num, train_num):
    warmup_steps = int(0.5 * train_num)
    total_steps = train_num * epoch_num - warmup_steps
    # print(total_steps, warmup_steps)
    return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,last_epoch=-1)

def train_generator(generator, iterator, optimizer, discriminator, ignore_padid, tokenizer=None):
    model = generator.seq2seq
    model.train()
    tr_loss = 0
    # tr_acc = 0
    for step, mb in tqdm(enumerate(iterator), desc='steps', total=len(iterator)):
        optimizer.zero_grad()
        mb_loss = 0
        
        enc_input, _, dec_output, reward = map(lambda elm: elm.to(device), mb)
        # print('[reward]: ', reward.shape)
        dec_input = torch.full((enc_input.shape[0],1),generator.vocab.token2idx[generator.vocab.START_TOKEN]).long().to(device)
        skip_row = []
        for i in range(generator.config.maxlen):
            # if i == generator.config.maxlen - 1:
            #     break
            # print('decode input: ',dec_input.shape)
            # print(dec_input)
            y_pred = model(enc_input, dec_input)
            # y_pred 第i个预测字符 [batch_size, vocab_size]
            # print('y_pred:',y_pred.shape)
            y_pred_copy = y_pred.detach()
            y_pred_ids = y_pred_copy.max(dim=-1)[1]

            # print('VVVVVVVVVVVV: ', y_pred_ids[:,-1].view(-1,1))
            # print('2222222222: ', y_pred.shape)
            y_pred_ids = y_pred_ids[:,-1].view(-1,1)
            # pred_values.append(y_pred[y_pred_ids[:,-1].view(-1,1)])
            # decoding_from_result(enc_input, y_pred, tokenizer)
            dec_input = torch.cat([dec_input, y_pred_ids], dim=1)

            # 保存训练得到的负样本到数组中, 为训练Discriminator做准备
            if tokenizer is not None:
                str_input, str_pred = decoding_to_pair(enc_input, y_pred_copy, tokenizer)
                # print('input: ', str_input)
                # print('pred: ',str_pred)
                # print('decinput: ', decoding_to_str(dec_input, tokenizer))

            # y_pred = y_pred.reshape(-1, y_pred.size(-1))
            dec_output = dec_output.view(-1).long()

            
            # padding 제외한 value index 추출
            # real_value_index = [dec_output != 0]

            # print(real_value_index)
            # print('=================')
            # print(y_pred.shape, dec_output.shape)
            # 根据log(P(y_t|Y_1:Y_{t-1})) * Q来计算loss
            for idx in range(y_pred.shape[0]):
                if idx in skip_row: continue
                if generator.is_end_token(y_pred_ids[idx][0]): skip_row.append(idx)
                pred_value = y_pred[idx][i][y_pred_ids[idx][0]]
                # pred_values.append(pred_value)
                mb_loss = -pred_value*reward[idx] # Input: (N, C) Target: (N)

            # print('reward:',reward.shape)
            # print('loss:',mb_loss.shape)
        mb_loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #     mb_acc = acc(y_pred, dec_output)

        tr_loss += mb_loss.item()
        # tr_acc = mb_acc.item()
        tr_loss_avg =  tr_loss / (step + 1)
        tr_summary = {'loss': tr_loss_avg}
        # total_step = epoch * len(iterator) + step
        
    return tr_loss/len(iterator)


class SqeGAN():
    def __init__(self, opt):
        self.generator = Generator(opt.gen_config, opt.vocab)

    def train(self, epoch, corpus):
        # 1. Initialize Gθ, Dφ with random weights θ, φ.

        # 2. Pre-train Gθ using MLE on S
        # 3. Generate negative samples using Gθ for training Dφ
        # 4. Pre-train Dφ via minimizing the cross entropy
        for i in range(epoch):
            # 5. g steps.
            print('epoch {i}')
            # 6. d steps.
