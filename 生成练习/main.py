
from dedal_data import DealData
from textrank_model import TextRank
from evaluetion import EvaluationTestData
import numpy as np


if __name__=="__main__":
    # ---1 加载训练数据----
    train_df,test_df=DealData().start()
    # ---2 加载模型-------
    TextRank(train_df)
    print(train_df)
    # ---3 训练结果输出----

    train_df['score']=train_df.apply(lambda row:EvaluationTestData(row['pred_line'],row['text']).model_evaluation(),axis=1)

    print(np.mean(train_df['score']))