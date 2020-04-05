# PyTorch_Transformer_Chatbot

Simple Chinese Generative Chatbot Implementation based on new PyTorch Transformer API (PyTorch v1.x / Python 3.x)

![transformer_fig](./assets/transformer_fig.png)

### ToDo
- Dynamic Memory Networks
- Beam Search
- Search hyperparams
- Attention Visualization

```python
def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor) -> torch.Tensor:
    x_enc_embed = self.input_embedding(enc_input.long())
    x_dec_embed = self.input_embedding(dec_input.long())

    # Masking
    src_key_padding_mask = enc_input == self.vocab.PAD_ID # tensor([[False, False, False,  True,  ...,  True]])
    tgt_key_padding_mask = dec_input == self.vocab.PAD_ID
    memory_key_padding_mask = src_key_padding_mask
    tgt_mask = self.transfomrer.generate_square_subsequent_mask(dec_input.size(1))

    # einsum ref: https://pytorch.org/docs/stable/torch.html#torch.einsum
    # https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
    x_enc_embed = torch.einsum('ijk->jik', x_enc_embed)
    x_dec_embed = torch.einsum('ijk->jik', x_dec_embed)


    # transformer ref: https://pytorch.org/docs/stable/nn.html#torch.nn.Transformer
    feature = self.transfomrer(src = x_enc_embed,
                               tgt = x_dec_embed,
                               src_key_padding_mask = src_key_padding_mask,
                               tgt_key_padding_mask = tgt_key_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask,
                               tgt_mask = tgt_mask.to(device)) # src: (S,N,E) tgt: (T,N,E)

    logits = self.proj_vocab_layer(feature)
    logits = torch.einsum('ijk->jik', logits)

    return logits
```

### Experiments


### 실행순서

```bash
python build_vocab.py # 构建词典
python train.py # 训练seq2seq模型
python inference.py # 推理测试
```

### Requirements

```bash
pip install mxnet
pip install gluonnlp
pip install konlpy
pip install python-mecab-ko
pip install chatspace
pip install tb-nightly
pip install future
pip install pathlib
```


### Reference Repositories
- [Chatbot Dataset by songys](https://github.com/songys/Chatbot_data)
- [Warmup Scheduler by ildoonet](https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py)
- [NLP implementation by aisolab](https://github.com/aisolab/nlp_implementation)
- [Transformer implementation by changwookjun](https://github.com/changwookjun/Transformer)
- [PyTorch_BERT_Implementation by codertimo](https://github.com/codertimo/BERT-pytorch)
- [pingpong-ai/chatspace](https://github.com/pingpong-ai/chatspace/tree/master)
- [PyTorch Seq2Seq Tutorial](https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/seq2seq_translation_tutorial.ipynb#scrollTo=OXkt42mheogQ)
