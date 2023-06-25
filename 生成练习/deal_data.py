
import pandas as pd
import json



import pandas as pd
import os
import jieba
import numpy as np
import json
from collections import Counter
from config import *
class DealData:
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
        res=[]
        for i in corpus:
            res.append([word for word in jieba.cut(i) if word not in self.stopwords and len(word)>1])
        return res

    '''统一句子长度'''
    def pad_len(self,sentences,seq_len):
        res=[]
        for sen in sentences:
            if len(sen)<seq_len:
                sen=sen+[0]*(seq_len-len(sen))
            else:sen=sen[:seq_len]
            res.append(sen)
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
        words = [item[0] for item in sortWordCount if item[1] >= 2]

        # 添加 "pad"index=0 和 "UNK"index=1
        words.insert(0, 'UNK')
        words.insert(0, 'PAD')
        word2idx = dict(zip(words, list(range(len(words)))))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open(self.config['vocab_path'], "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        return word2idx

    '''word2id'''
    def word_to_index(self, sentence, word2idx,seq_len):
        """
        将词转换成索引
        """
        sentence_index = [[word2idx.get(item, word2idx["UNK"]) for item in sent] for sent in sentence]
        sentence_index=self.pad_len(sentence_index,seq_len)

        return sentence_index

    '''idx2word'''
    def index_to_word(self,sentences):
        # 待完成
        # sentences:[[1,09,32,41,]]
        # 在评估数据时使用
        with open(self.config['vocab_path'], "r", encoding="utf-8") as f:
            vocab = json.load(f)#{ '指挥员': 79169, '一字排开': 79170}
        count=0
        for k,v in vocab:
            print(k,v)
            count+=1
            if count==10:
                break
        pass



    '''主函数'''
    def load_data(self):
        '''
        word2index,equal seqlen
        '''

        # 1 读数据
        filename=os.path.join(self.cur_path,'data/train.jsonl')
        f = open(filename, 'r+', encoding='utf-8').readlines()
        df = pd.DataFrame(json.loads(line) for line in f)
        #['id', 'text', 'url', 'summary']

        # 2 分词(encoder与decoder用一套词典)
        sentences = self.cut_word(df['text'].to_list()+df['summary'].to_list())

        # 3 建立词典，只有训练数据需要
        if not os.path.exists(self.config['vocab_path']):
            self.Vocabulary(sentences)

        with open(self.config['vocab_path'], "r", encoding="utf-8") as f:
            vocab=json.load(f)

        # 4 划分数据集
        df = df.sample(frac=1)
        train_df = df[:int(len(df) * .7)]
        test_df = df[int(len(df) * .7):]

        # 5 将词转换成idx
        idx_train = self.word_to_index(self.cut_word(train_df['text'].to_list()), vocab,seq_len=self.config['enc_len'])
        idx_train_y = self.word_to_index(self.cut_word(train_df['summary'].to_list()), vocab,seq_len=self.config['dec_len'])
        idx_test = self.word_to_index(self.cut_word(test_df['text'].to_list()), vocab,seq_len=self.config['enc_len'])
        idx_test_y = self.word_to_index(self.cut_word(test_df['summary'].to_list()), vocab,seq_len=self.config['dec_len'])


        return train_df,test_df,idx_train,idx_train_y,idx_test,idx_test_y
# DealData().index_to_word()



