
import pandas as pd
import json

class DealData(object):
    def __init__(self):
        pass

    def read_data(self):
        f=open('data/train.jsonl', 'r+', encoding='utf-8').readlines()
        df=pd.DataFrame(json.loads(line) for line in f)
        return df

        # print(json_data)
    def data_explore(self):
        df=self.read_data()
        df['summary_len']=df['summary'].map(lambda x:len(str(x)))
        df['text_len']=df['text'].map(lambda x:len(str(x)))

        # 数据探索
        print(df.head())
        print(df.columns)
        print(df.info())
        print(df['summary_len'].describe())
        print(df['text_len'].describe())





# DealData().data_explore()