from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
from sklearn.model_selection import train_test_split
# from tensorflow import keras
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pprint import pprint
from typing import Tuple, Callable, List # https://m.blog.naver.com/PostView.nhn?blogId=passion053&logNo=221070020739&proxyReferer=https%3A%2F%2Fwww.google.com%2F
from torch.autograd import Variable

from tqdm import tqdm


class ChatbotDataset(Dataset):
    def __init__(self, transform_fn: Callable[[str], List[int]], filepath=None, corpus=None) -> None:
        """

        :param filepath:
        :param transform_fn:
        """
        if corpus is None:
            question, answer = self.load_data_from_txt(filepath)
        else:
            question, answer, reward = self.load_data_from_dataframe(corpus)
        self._corpus = question
        self._label = answer
        self._reward = reward
        self._transform = transform_fn

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_input = torch.tensor(self._transform([self._corpus[idx].lower()]))
        dec_input, dec_output = torch.tensor(self._transform([self._label[idx].lower()], add_start_end_token=True))
        # 使用带梯度的张量
        enc_input = Variable(enc_input).type(torch.LongTensor)
        dec_input = Variable(dec_input).type(torch.LongTensor)
        dec_output = Variable(dec_output).type(torch.LongTensor)
        if self._reward is not None:
            reward = torch.tensor(self._reward[idx])
            reward = Variable(reward)
            return enc_input[0], dec_input[0], dec_output[0], reward

        return enc_input[0], dec_input[0], dec_output[0]

    def load_data_from_txt(self, data_path):
        with open(data_path, mode='r', encoding='utf-8') as io:
            lines = io.readlines()
            question = []
            answer = []
            for line in lines:
                if line == "":
                    continue
                try:
                    question_item, answer_item = line.split('\t')
                    question.append(question_item)
                    answer.append(answer_item)
                except:
                    print(line)
                # print('Q: ', question_item, 'A: ',answer_item)
        return question, answer

    def load_data_from_dataframe(self, dataframe, train_val_split=False):
        question, answer = list(dataframe['question']), list(dataframe['answer'])
        reward = None
        if list(dataframe).__contains__('reward'): reward = list(dataframe['reward'])
        if train_val_split:
            train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33,
                                                                            random_state=42)
            return train_input, train_label, eval_input, eval_label
        else:
            return question, answer, reward

    def load_data(self, data_path, train_val_split=True):
        """
        code from "https://github.com/changwookjun/Transformer/blob/25d9472155cb0788d11dbfe274526690915fe95e/data.py#L27"
        :param data_path:
        :return: train_input, train_label, eval_input, eval_label
        """
        data_df = pd.read_csv(data_path, header=0)
        return load_data_from_dataframe(data_df)