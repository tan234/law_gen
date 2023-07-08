
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

from data import DealData
from train_model import pegasus_train
from evaluating import pegasus_evaluate
from pred import PegasusPred
from gpt2_model import Gpt2Model
# 文本摘要任务多种模型实现

def choice_model():

    # =======Pegasus model start==============
    # 1 分词器
    model_name = 'Pegasus'

    pegasus_tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese')
    # 2 参数
    pegasus_config = Pegasus_config

    # 3 数据获取
    dataset = DealData(pegasus_tokenizer, pegasus_config)
    train_dataset = dataset.train_dataset
    dev_dataset = dataset.dev_dataset
    test_dataset = dataset.test_dataset
    tokenizer = dataset.tokenizer

    print('--------开始训练 %s --------'%(model_name))
    evaluate=pegasus_evaluate
    pegasus_train(tokenizer, model_name, pegasus_config, train_dataset, dev_dataset, test_dataset, evaluate)


    print('--------开始推断 %s --------'%(model_name))
    content= '在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！'
    summary=PegasusPred(content,model_name).infer()
    # =======Pegasus model end ==============





    # =======Gpt2 model start =====================================


    model_name = 'GPT2'
    gpt2_tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')# 使用的是gpt2-mediu-en分词器
    gpt2_tokenizer.add_special_tokens({"sep_token": "<sep>"})# 添加一个特殊字符来区分内容和摘要<>

    model=Gpt2Model(vocab_size=tokenizer.vocab_size+1)

    # 数据获取
    dataset = DealData(gpt2_tokenizer, GPT2_config)
    train_dataset = dataset.train_dataset
    dev_dataset = dataset.dev_dataset
    test_dataset = dataset.test_dataset
    tokenizer = dataset.tokenizer




    #=======Gpt2 model end ==============







# ===========下面是transformer模型的分界线==========
# from deal_data import DealData
# from textrank_model import TextRank
# from evaluetion import EvaluationTestData
# import numpy as np
# if __name__=="__main__":
#     # ---1 加载训练数据----
#     train_df,test_df=DealData().start()
#     # ---2 加载模型-------
#     TextRank(train_df)
#     print(train_df)
#     # ---3 训练结果输出----
#
#     train_df['score']=train_df.apply(lambda row:EvaluationTestData(row['pred_line'],row['text']).model_evaluation(),axis=1)
#
#     print(np.mean(train_df['score']))



