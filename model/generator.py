from model.net import TransformerNet
from data_utils.utils import Config
from data_utils.vocab_tokenizer import Vocabulary
from data_utils.vocab_tokenizer import Tokenizer, Vocabulary, keras_pad_fn, mecab_token_pos_flat_fn
from evaluate import evaluate, decoding_from_result, decoding_to_pair
import torch

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

    def switch_mode(self, mode='eval'):
        if mode=='eval': self.seq2seq.eval()
        else: self.seq2seq.train()
        
    def parameters(self):
        return self.seq2seq.parameters()

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