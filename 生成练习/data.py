
# encoding:utf-8
import random
from torch.utils.data import TensorDataset, DataLoader

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
from gpt2_model import Gpt2Model
import pickle
from paddle.io import Dataset
import torch
from config import *

import pandas as pd
import json



import pandas as pd
import os
# import jieba
import numpy as np
import json
from collections import Counter
from config import *
# from jieba import posseg as psg

class TransformerData:
    '''
    管理从数据处理到模型能用的整个过程
    '''
    def __init__(self):

        self.cur_path=os.path.dirname(__file__)
        self.config=data_config
        self.stopwords=self.stopw()

    '''停用词典'''
    def stopw(self):
        stopword=[line.replace('\n','')for line in open(self.config['stopword_path'],'r+',encoding='utf-8').readlines()]
        return dict(zip(stopword,range(0,len(stopword))))

    '''分词 去停用词'''
    def cut_word(self,corpus):
        # str
        res=[list(i)for i in corpus]
        # word
        # res=[]
        # for i in corpus:
        #     # res.append([word for word,p in psg.cut(i) if word not in self.stopwords and len(word)>1 and p not in ['m']])
        #     res.append([word for word in jieba.cut(i) ])

        return res



    '''训练数据词典'''
    def Vocabulary(self, corpus):
        """
        corpus:数据的list[[linecut],[linecut]]
        """

        allWords = [word for words in corpus for word in words]

        # 去掉停用词
        # subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 3]


        # 给decoder加开头结尾标识
        words.insert(0, 'SOS')
        words.insert(0, 'SOE')
        # 添加 "pad"index=0 和 "UNK"index=1
        words.insert(0, 'UNK')
        words.insert(0, 'PAD')
        word2idx = dict(zip(words, list(range(len(words)))))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open(self.config['vocab_path'], "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        return word2idx

    '''统一句子长度'''

    # def pad_len(self, sentences, seq_len):
    #     res = []
    #     for sen in sentences:
    #         if len(sen) < seq_len:
    #             sen = sen + [0] * (seq_len - len(sen))
    #         else:
    #             sen = sen[:seq_len]
    #         res.append(sen)
    #     return res

    '''word2id'''
    def word_to_index(self, sentence, word2idx,seq_len,type='x'):
        """
        将词转换成索引
        """
        sentence_index = []
        if type=='y':
            for sent in sentence:
                # decoder 句子加上'S'和'E'
                idx=[word2idx['S']]
                for item in sent:
                    idx.append(word2idx.get(item, word2idx["UNK"]))

                if len(idx)<seq_len-1:
                    idx = idx + [0] * (seq_len-1 - len(idx))
                else:
                    idx = idx[:seq_len-1]

                idx.append(word2idx['E'])
                sentence_index.append(idx)

        else:
            for sent in sentence:
                idx = [word2idx.get( item, word2idx["UNK"]) for item in sent]
                if len(idx) < seq_len :
                    idx = idx + [0] * (seq_len - len(idx))
                else:
                    idx = idx[:seq_len]
                sentence_index.append(idx)

        # sentence_index = [[word2idx.get(item, word2idx["UNK"]) for item in sent] for sent in sentence]
        # sentence_index=self.pad_len(sentence_index,seq_len)
        return sentence_index

    '''idx2word'''
    def index_to_word(self,word2idx):
        # idx2word保存
        id2w = {}
        for k, v in word2idx.items():
            id2w[v] = k
        with open(data_config['idx2word_dict'], "w", encoding="utf-8") as f2:
            json.dump(id2w, f2)



    '''主函数'''
    def load_data(self):
        '''
        word2index,equal seqlen
        '''

        # 1 读数据
        filename=os.path.join(self.cur_path, 'data/train_y.jsonl')
        f = open(filename, 'r+', encoding='utf-8').readlines()
        df = pd.DataFrame(json.loads(line) for line in f)
        # df['summary']=df['summary'].map(lambda x:'S'+str(x)+'E')
        #['id', 'text', 'url', 'summary']
        # print(df)

        # 查看句子长度
        # df['text_l']=df['text'].map(lambda x:len(list(x)))
        # df['sum_l']=df['summary'].map(lambda x:len(list(x)))
        # print(df['text_l'].describe())
        # print(df['sum_l'].describe())

        # 2 分词(encoder与decoder用一套词典)
        sentences = self.cut_word(df['text'].to_list()+df['summary'].to_list())

        # 3 建立词典，只有训练数据需要
        if not os.path.exists(self.config['vocab_path']):
            word2idx=self.Vocabulary(sentences)
            self.index_to_word(word2idx)

        with open(self.config['vocab_path'], "r", encoding="utf-8") as f:
            vocab=json.load(f)
            # print(vocab['S'])
            print('词典大小：',len(vocab))

        # 4 划分数据集
        df = df.sample(frac=1)
        train_df = df[:int(len(df) * .7)]
        test_df = df[int(len(df) * .7):]

        # 5 将词转换成idx
        idx_train = self.word_to_index(self.cut_word(train_df['text'].to_list()), vocab,seq_len=self.config['enc_len'])
        idx_train_y = self.word_to_index(self.cut_word(train_df['summary'].to_list()), vocab,seq_len=self.config['dec_len'],type='y')
        idx_test = self.word_to_index(self.cut_word(test_df['text'].to_list()), vocab,seq_len=self.config['enc_len'])
        idx_test_y = self.word_to_index(self.cut_word(test_df['summary'].to_list()), vocab,seq_len=self.config['dec_len'],type='y')
        # print(idx_train[:3],len(idx_train[0]))
        # print(idx_train_y[:3],len(idx_train_y[0]))
        # print(idx_test[:3],len(idx_test[0]))
        # print(idx_test_y[:3],len(idx_test_y[0]))

        return train_df,test_df,idx_train,idx_train_y,idx_test,idx_test_y

def load_batch(x, y, batchSize,give_emb=True):
    '''
    give_emb==True:用户自定义emb,则X=torch.FloatTensor(x)
    give_emb==False:nn生成emb,则X=torch.LongTensor(x)
    '''
    X=torch.FloatTensor(x)
    if not give_emb:
        X = torch.LongTensor(x)
    data_set = TensorDataset(X,
                              torch.LongTensor(y))
    data_loader = DataLoader(dataset=data_set,
                              batch_size=batchSize,
                              shuffle=False)
    return data_loader

class PegasusData(object):

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


class Gpt2Data(object):
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        # train_set = [{'content': '会发你发附加赛佛开票', 'title': '和大'},
        #              {'content': '都还是金广发女法律方式看见你', 'title': '大家发疯'}]

        train_dataset = load_dataset("csv", data_files='data/train.csv', split="train")
        dev_dataset = load_dataset("csv", data_files='data/dev.csv', split="train")
        test_dataset = load_dataset("csv", data_files='data/test.csv', split="train")

        self.train_dataset=self.data(train_dataset,filaname='gpt_train.pkl')
        self.dev_dataset=self.data(dev_dataset,filaname='gpt_dev.pkl')
        self.test_dataset=self.data(test_dataset,filaname='gpt_test.pkl')


    '''1 数据处理并保存'''
    def data(self, train_set,filaname):

        all_data = []
        for data in (train_set):
            content = data["content"].lower()
            title = data["title"].lower()
            content_id = [self.tokenizer.sep_token_id]
            c = self.tokenizer(content, return_token_type_ids=False)["input_ids"]
            # t2.append(len(c))
            content_id.extend(c[:150])
            length = len(content_id)
            content_id.append(self.tokenizer.sep_token_id)
            t = self.tokenizer(title, return_token_type_ids=False)["input_ids"]
            content_id.extend(t[:19])

            label = t[:19]
            label.append(self.tokenizer.sep_token_id)

            all_data.append([content_id, label, length])

        # 保存为二进制文件
        with open("data/"+filaname, "wb") as f:
            pickle.dump(all_data, f)

        return all_data
        # print(all_data)

    # def load_batch(self,batchSize,data_path):
    #     '''
    #     data:[[[contentid],[labid],len],,,,]
    #     '''
    #     with open(data_path, "rb") as f:
    #         data_list = pickle.load(f)
    #     data = data_list
    #
    #     X = torch.tensor([i[0] for i in data])
    #     Y = torch.tensor([i[1] for i in data])
    #     l = torch.tensor([i[2] for i in data])
    #     # X = [i[0] for i in data]
    #     # Y = [i[1] for i in data]
    #     # l = [i[2] for i in data]
    #     data_set = TensorDataset(X,Y,l)
    #
    #     data_loader = DataLoader(dataset=data_set,
    #                               batch_size=batchSize,
    #                               shuffle=False)
    #     return data_loader



'''2 读取保存的数据'''
class MyDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            data_list = pickle.load(f)
        self.data = data_list

    def __getitem__(self, idx):
        return paddle.to_tensor([self.data[idx][0]]), paddle.to_tensor([self.data[idx][1]]), self.data[idx][2]

    def __len__(self):
        return len(self.data)
