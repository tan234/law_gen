
import pandas as pd
import json



import pandas as pd
import os
import jieba
import numpy as np
import json
from collections import Counter
from config import *
from jieba import posseg as psg
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
            res.append([word for word,p in psg.cut(i) if word not in self.stopwords and len(word)>1 and p not in ['m']])

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
        # 给decoder加开头结尾标识
        words.insert(len(words), 'S')
        words.insert(len(words), 'E')
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
        filename=os.path.join(self.cur_path,'data/train.jsonl')
        f = open(filename, 'r+', encoding='utf-8').readlines()
        df = pd.DataFrame(json.loads(line) for line in f)
        # df['summary']=df['summary'].map(lambda x:'S'+str(x)+'E')
        #['id', 'text', 'url', 'summary']
        # print(df)

        # 2 分词(encoder与decoder用一套词典)
        sentences = self.cut_word(df['text'].to_list()+df['summary'].to_list())
        # df['t']=self.cut_word(df['text'].to_list())
        # df['s']=self.cut_word(df['summary'].to_list())
        # df['tl']=df['t'].map(lambda x:len(x))
        # df['sl']=df['s'].map(lambda x:len(x))
        # df.to_excel('cut_word.xlsx',index=False)

        # 3 建立词典，只有训练数据需要
        if not os.path.exists(self.config['vocab_path']):
            word2idx=self.Vocabulary(sentences)
            self.index_to_word(word2idx)

        with open(self.config['vocab_path'], "r", encoding="utf-8") as f:
            vocab=json.load(f)
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
DealData().load_data()



