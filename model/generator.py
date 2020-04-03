from model.net import TransformerNet
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Vocabulary
from data_utils.vocab_tokenizer import Tokenizer, Vocabulary, keras_pad_fn, mecab_token_pos_flat_fn
from evaluate import evaluate, decoding_from_result, decoding_to_pair, decoding_to_str
import pandas as pd
import torch
from tqdm import tqdm

class Generator():
    def __init__(self, config: Config, vocab: Vocabulary, state_dict = None):
        self.seq2seq = TransformerNet(config=config, vocab=vocab)
        self.config = config
        self.vocab = vocab
        self.tokenizer = Tokenizer(vocab=vocab, split_fn=mecab_token_pos_flat_fn, pad_fn=keras_pad_fn, maxlen=config.maxlen)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if state_dict is not None:
            self.seq2seq.load_state_dict(state_dict)
        self.seq2seq.to(self.device)
        self.learning_rate = config.learning_rate

    def switch_mode(self, mode='eval'):
        if mode=='eval': self.seq2seq.eval()
        else: self.seq2seq.train()
        
    def parameters(self):
        return self.seq2seq.parameters()

    def get_state_dict(self):
        state_dict = self.seq2seq.to(torch.device('cpu')).state_dict()
        self.seq2seq.to(self.device)
        return state_dict

    def eval(self): self.seq2seq.eval()
    def train(self): self.seq2seq.train()

    def gen_output_with_ids(self, input_ids):
        # dec_input = torch.tensor([[self.vocab.token2idx[self.vocab.START_TOKEN]]])
        dec_input = torch.full((input_ids.shape[0],1),self.vocab.token2idx[self.vocab.START_TOKEN]).long()
        # print(dec_input)
        for i in range(self.config.maxlen):
            # print('input_ids', input_ids)
            # print('dec_input', dec_input)
            # print('device', self.device)
            # y_pred = self.seq2seq(input_ids, dec_input)
            y_pred = self.seq2seq(input_ids.to(self.device), dec_input.to(self.device))
            y_pred_ids = y_pred.max(dim=-1)[1]
            # if (y_pred_ids[0,-1] == self.vocab.token2idx[self.vocab.END_TOKEN]).to(torch.device('cpu')).numpy():
            #     # 填充PAD_ID
            #     fill_pad  = torch.full((input_ids.shape[0],self.config.maxlen-dec_input.shape[1]), self.vocab.PAD_ID).long().to(self.device)
            #     # print(fill_pad.shape,y_pred_ids.shape)
            #     y_pred_ids = torch.cat([y_pred_ids, fill_pad], dim=1)
            #     break
            # 对self.vocab.END_TOKEN之后的位置填充pad_id
            # end_indices = (y_pred_ids==self.vocab.token2idx[self.vocab.END_TOKEN]).nonzero()
            # for val in end_indices:
            #     fill_pad  = torch.full((1,self.config.maxlen-val[1]), self.vocab.PAD_ID).long().to(self.device)
            #     print(y_pred_ids[val[0]])
            #     y_pred_ids[val[0]] = torch.cat([y_pred_ids[val[0],-1], fill_pad], dim=1)
            # print(end_indices)

            # decoding_from_result(enc_input, y_pred, tokenizer)
            # print('y_pred_ids',y_pred_ids.shape)
            # print(dec_input.shape, y_pred_ids.shape,y_pred.shape)
            # dec_input = torch.cat((dec_input.to(torch.device('cpu')), y_pred_ids[:,-1].view(-1,1).to(torch.device('cpu'))), dim=1)
            dec_input = torch.cat((dec_input.to(self.device), y_pred_ids[:,-1].view(-1,1)), dim=1)
            # print(dec_input)

            if i == self.config.maxlen - 1:
                # output_str = decoding_from_result(enc_input=enc_input, y_pred=y_pred, tokenizer=self.tokenizer)
                break
        
        # 对self.vocab.END_TOKEN之后的位置填充pad_id
        # end_indices = (y_pred_ids==self.vocab.token2idx[self.vocab.END_TOKEN]).nonzero()
        # 1. 寻找每行最早的结束token
        # end_tokens = []
        # last_r = -1
        # last_col = -1
        # print(end_indices)
        # for val in end_indices:
        #     if val[0]>last_r:
        #         last_r = val[0]
        #         last_col = 500
        #     if val[1]<last_col:
        #         last_col = val[1]
        #         end_tokens.append([last_r.cpu().tolist(), last_col.cpu().tolist()])
        #         continue
        # for item in end_tokens:
        #     fill_pad  = torch.full((1, self.config.maxlen-item[1]), self.vocab.PAD_ID).long().to(self.device)
        #     print(y_pred_ids[item[0]][0:item[1]])
        #     y_pred_ids[item[0]] = torch.cat([y_pred_ids[item[0]][0:item[1]], fill_pad], dim=1)
        # print('end_tokens',end_tokens)
        return y_pred_ids, y_pred

    def is_end_token(self, token):
        if (self.vocab.token2idx[self.vocab.END_TOKEN]==token).cpu().numpy():
            return True
        return False

    def gen_output(self, input_text):
        enc_input = torch.tensor(self.tokenizer.list_of_string_to_arr_of_pad_token_ids([input_text]))
        dec_input = torch.tensor([[self.vocab.token2idx[self.vocab.START_TOKEN]]])
        output_str = ''
        for i in range(self.config.maxlen):
            y_pred = self.seq2seq(enc_input.to(self.device), dec_input.to(self.device))
            y_pred_ids = y_pred.max(dim=-1)[1]
            if (y_pred_ids[0,-1] == self.vocab.token2idx[self.vocab.END_TOKEN]).to(torch.device('cpu')).numpy():
                output_str = decoding_from_result(enc_input=enc_input, y_pred=y_pred, tokenizer=self.tokenizer)
                break

            # decoding_from_result(enc_input, y_pred, tokenizer)
            dec_input = torch.cat([dec_input.to(torch.device('cpu')), y_pred_ids[0,-1].unsqueeze(0).unsqueeze(0).to(torch.device('cpu'))], dim=-1)

            if i == self.config.maxlen - 1:
                output_str = decoding_from_result(enc_input=enc_input, y_pred=y_pred, tokenizer=self.tokenizer)
                break
        
        output_str = output_str.replace('\n', '').replace('\r','')
        return output_str

    def sample(self, dataset):
        # 根据输入数据集生成回复数据集
        data_enc_input = []
        data_dec_input = []
        data_dec_output = []
        question = []
        answer = []
        self.seq2seq.eval()
        # preds = []
        for item in tqdm(dataset,desc='sampling'):
            enc_input, dec_input, dec_output = map(lambda elm: elm, item)
            pred_ids, pred = self.gen_output_with_ids(enc_input)
            # print(pred.shape)
            output_str = decoding_to_str(pred_ids, self.tokenizer)
            input_str = decoding_to_str(enc_input, self.tokenizer)
            # print(input_str)
            # print('---------------')
            # print(output_str)
            # discriminator_inputs = []
            # for r in range(len(input_str)):
            #     question += input_str[r]
            #     answer += output_str[r]
            data_enc_input += enc_input
            data_dec_input += dec_input
            data_dec_output += dec_output
            question += input_str
            answer += output_str
            # preds += pred
            # data_D_set += discriminator_inputs
            # batch_data.append([enc_input, dec_input, dec_output, discriminator_inputs])
            break
        # print(batch_data)
        df = pd.DataFrame({'enc_input': data_enc_input, 'dec_input': data_dec_input, 'dec_output': data_dec_output, 'question': question, 'answer': answer})
        return df
            