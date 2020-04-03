from __future__ import absolute_import, division, print_function, unicode_literals
from torch import nn
from .positional_encoding import PositionalEmbedding
from .token_embedding import TokenEmbedding
import torch

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class Embeddings(nn.Module):
    def __init__(self, config, vocab):
        super(Embeddings, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size=config.vocab_size, embed_size=config.embed_size, pad_id=vocab.token2idx[vocab.PAD])#.to(device)
        self.pos_embedding = PositionalEmbedding(d_model = config.embed_size, max_len=config.maxlen)#.to(device)

    def forward(self, x):
        token_embed = self.token_embedding(x)
        pos_embed = self.pos_embedding(x)
        return token_embed + pos_embed

def main():
    print("Embeddings")

if __name__ == '__main__':
    main()