
# encoding:utf-8
import random

from paddlenlp.transformers import AutoModelForConditionalGeneration
from paddlenlp.data import DataCollatorForSeq2Seq
import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
# from visualdl import LogWriter
from datasets import load_dataset
from paddlenlp.transformers import AutoTokenizer
from functools import partial
import pandas as pd
from paddlenlp.transformers import AutoModelForConditionalGeneration
from paddlenlp.data import DataCollatorForSeq2Seq
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
# from visualdl import LogWriter
from rouge import Rouge
import paddle
from paddlenlp.transformers import GPTModel, GPTTokenizer

import time
# from paddlenlp.utils.log import logger
from paddlenlp.metrics import BLEU
from tqdm import tqdm
import numpy as np
import os
from paddlenlp.transformers import GPTTokenizer

from config import *


class DealData(object):

    def __init__(self,tokenizer,config):
        # ---1 数据划分----
        # self.split_dataset()

        #--- 2 通过load_dataset读取划分好的数据集-------
        train_dataset = load_dataset("csv", data_files='data/train.csv', split="train")
        dev_dataset = load_dataset("csv", data_files='data/dev.csv', split="train")
        test_dataset = load_dataset("csv", data_files='data/test.csv', split="train")

        # ----3 分词-------
        # 初始化分词器
        self.tokenizer = tokenizer
        # 对内容进行分词，并映射为id ;max_len是保留的分词个数，不pad,padding=True也不管用；truncation=True表示允许截断
        # {'input_ids': [113, 30502, 2363, 4882, 1], 'attention_mask': [1, 1, 1, 1, 1]}
        # attention_mask?

        # 原始字段需要移除
        remove_columns = ['content', 'title']
        # 定义转换器
        trans_func = partial(self.convert_example,
                             text_column='content',
                             summary_column='title',
                             tokenizer=self.tokenizer,
                             max_source_length=config['max_source_length'],
                             max_target_length=config['max_target_length'])

        # train_dataset和dev_dataset分别转换
        # map() 方法有一个重要的参数 batched，当设置为 True 时（默认为 False ），数据处理函数 trans_func() 的输入不再是单条数据，而是数据集的所有数据：
        # 没有pad
        self.train_dataset = train_dataset.map(trans_func,
                                          batched=True,
                                          load_from_cache_file=True,
                                          remove_columns=remove_columns)
        self.dev_dataset = dev_dataset.map(trans_func,
                                      batched=True,
                                      load_from_cache_file=True,
                                      remove_columns=remove_columns)

        self.test_dataset = test_dataset.map(trans_func,
                                        batched=True,
                                        load_from_cache_file=True,
                                        remove_columns=remove_columns)


    '''数据集清洗划分'''
    def split_dataset(self):
        # 划分数据集
        df_t = pd.read_csv('data/summary.txt', header=None)  # [2400591 rows x 1 columns]
        df_c = pd.read_csv('data/article.txt', header=None)
        df_t = df_t.dropna()
        df_c = df_c.dropna()

        print(len(df_c), len(df_t))

        df_c['title'] = df_t[0]

        df_c.columns = ['content', 'title']
        # df_c['content']=df_c[0]
        # del df_c[0]
        print(df_c['content'])
        print(df_c['title'])

        train_dataset = df_c[:100]

        dev_dataset = df_c[100:130]

        test_dataset = df_c[130:150]

        train_dataset.to_csv('data/train.csv', index=False)
        dev_dataset.to_csv('data/dev.csv', index=False)

        test_dataset.to_csv('data/test.csv', index=False)

    '''数据分词+word2id'''
    def convert_example(self,example, text_column, summary_column, tokenizer,
                        max_source_length, max_target_length):
        """
        构造模型的输入.
        """
        inputs = example[text_column]
        targets = example[summary_column]
        # 对content进行分词
        model_inputs = tokenizer(inputs,
                                 max_length=max_source_length,
                                 padding=False,
                                 truncation=True,
                                 return_attention_mask=True)
        # 对title进行分词
        summary_inputs = tokenizer(targets,
                                   max_length=max_target_length,
                                   padding=False,
                                   truncation=True)
        # decoder端的labels为摘要id，decoder端的输入为shifted摘要id
        model_inputs["labels"] = summary_inputs["input_ids"]
        return model_inputs
