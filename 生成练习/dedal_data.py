
import pandas as pd
import json

class DealData(object):
    '''
    返回训练数据
    '''
    def __init__(self):
        pass

    def read_data(self):

        # 读取数据
        f=open('data/train.jsonl', 'r+', encoding='utf-8').readlines()
        df=pd.DataFrame(json.loads(line) for line in f)

        # 划分数据集
        df=df.sample(frac=1)
        train_df=df[:int(len(df)*.7)]
        test_df=df[int(len(df)*.7):]

        return train_df,test_df

    def data_explore(self):
        '''
        Index(['id', 'text', 'url', 'summary'], dtype='object')

        :return:
        '''
        df=self.read_data()
        df['summary_len']=df['summary'].map(lambda x:len(str(x)))
        df['text_len']=df['text'].map(lambda x:len(str(x)))

        # 数据探索
        print(df.head())
        print(df.columns)
        print(df.info())
        print(df['summary_len'].describe())
        print(df['text_len'].describe())

    def start(self):
        '''

        :return:
        '''
        train_df,test_df=self.read_data()
        return train_df,test_df

