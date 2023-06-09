
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
        print(df)




DealData().data_explore()