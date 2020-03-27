from generator import Generator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler
from typing import Tuple, List

class DiscriminatorDatasetReader(Dataset):
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
            # print('Q: ', row['question'])
            output = generator.gen_output(row['question'])
            if output==row['answer']: continue
            # print('A: ', row['answer'])
            # f.write(row['question']+'|'+row['answer']+'\t1\n')
            # f.write(row['question']+'|'+output+'\t0\n')

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

def train_generator(model, iterator, optimizer, scheduler):
    model.train()
    total_loss = 0
    for x, y in tqdm(iterator):
        optimizer.zero_grad()
        mask = (x != 0).float()
#         print(x.shape,y.shape)
        loss, outputs = model(x, attention_mask=mask, labels=y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    avg_loss = total_loss / len(iterator)
    return avg_loss

def evaluate_generator(model, iterator):
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
