
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

from data import PegasusData,Gpt2Data,MyDataset,TransformerData
from train import pegasus_train,gpt2_train,transformer_train
from evaluating import pegasus_evaluate
from pred import PegasusPred,Gpt2Pred,TransformerPred
from deal_data import DealData
from textrank_model import TextRank
from evaluetion import EvaluationTestData,trans_eval
import numpy as np
from gpt2_model import Gpt2Model
# 文本摘要任务多种模型实现

def choice_model():
    content = '综述了人工神经网络FPGA实现的研究进展和关键技术，分析了如何利用FPGA的可重构技术来实现人工神经网络，探讨了实现过程中的一些问题，并介绍了作为神经网络FPGA实现的基础一可重构技术。指出测试平台设计、软件工具、FPGA友好学习算法及拓扑结构自适应等方面的研究，是今后研究的热点。'


    # =======transformer model start==============

    if __name__=="__main__":
        # ---1 加载训练数据----

        train_df,test_df,idx_train,idx_train_y,idx_test,idx_test_y=TransformerData().load_data()
        model_name='transformer'
        config=Transformer_config
        # ---2 加载模型-------
        TextRank(train_df)

        transformer_train(idx_train, idx_train_y, idx_test, idx_test_y,model_name,config)
        print(train_df)
        TransformerPred.start(content)
        # ---3 训练结果输出----

        # train_df['score']=train_df.apply(lambda row:trans_eval(row['pred_line'],row['text']).model_evaluation(),axis=1)

        print(np.mean(train_df['score']))

    # =======transformer model end ==============



    # =======Pegasus model start==============
    # 1 分词器
    model_name = 'Pegasus'

    # pegasus_tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese')
    # # 2 参数
    # pegasus_config = Pegasus_config
    #
    # # 3 数据获取
    # dataset = PegasusData(pegasus_tokenizer, pegasus_config)
    # train_dataset = dataset.train_dataset
    # dev_dataset = dataset.dev_dataset
    # test_dataset = dataset.test_dataset
    # tokenizer = dataset.tokenizer
    #
    # print('--------开始训练 %s --------'%(model_name))
    # # evaluate=pegasus_evaluate
    # pegasus_train(tokenizer, model_name, pegasus_config, train_dataset, dev_dataset, test_dataset)
    #

    print('--------开始推断 %s --------'%(model_name))
    content= '在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！'
    content='最高人民法院院长周强在21日召开的全国高级法院院长会议上明确提出，法官要着力提高庭审驾驭能力和水平，正确发挥在庭审程序运行中的指挥、控制职能，尊重和保障律师依法履职。,最高法：尊重和保障律师依法履职'
    content = '综述了人工神经网络FPGA实现的研究进展和关键技术，分析了如何利用FPGA的可重构技术来实现人工神经网络，探讨了实现过程中的一些问题，并介绍了作为神经网络FPGA实现的基础一可重构技术。指出测试平台设计、软件工具、FPGA友好学习算法及拓扑结构自适应等方面的研究，是今后研究的热点。'

    summary=PegasusPred(content,model_name).infer()
    # =======Pegasus model end ==============

    '''
    
    # =======Gpt2 model start 未完成 =====================================
    # 1 模型
    model_name = 'GPT2'
    gpt2_tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')# 使用的是gpt2-mediu-en分词器
    gpt2_tokenizer.add_special_tokens({"sep_token": "<sep>"})# 添加一个特殊字符来区分内容和摘要<>
    # print(gpt2_tokenizer.convert_ids_to_string([101, 7564, 6378]))
    # print(gpt2_tokenizer.decode([101, 7564, 6378]))
    # kk
    model=Gpt2Model(vocab_size=gpt2_tokenizer.vocab_size+1)
    config=GPT2_config
    # 2 数据
    dataset=Gpt2Data(gpt2_tokenizer)
    train_dataset = dataset.train_dataset
    dev_dataset = dataset.dev_dataset
    test_dataset = dataset.test_dataset

    # batch data
    # train_loder=dataset.load_batch(config['batch_size'],'data/gpt_train.pkl')
    # dev_loder=dataset.load_batch(config['batch_size'],'data/gpt_dev.pkl')
    # test_loder=dataset.load_batch(config['batch_size'],'data/gpt_test.pkl')

    train_loder=DataLoader(MyDataset('data/gpt_train.pkl'), batch_size=config['batch_size'], shuffle=False)
    dev_loder=DataLoader(MyDataset('data/gpt_dev.pkl'), batch_size=config['batch_size'], shuffle=False)
    test_loder=DataLoader(MyDataset('data/gpt_test.pkl'), batch_size=config['batch_size'], shuffle=False)

    # for data in dev_loder:
    #     content, label, lenght = data
    #     print(content)
    #     print(label)
    #     kk
    print('--------开始训练 %s --------'%(model_name))
    # gpt2_train(train_loder)
    gpt2_train(gpt2_tokenizer, config, model, train_loder,dev_loder,len(train_dataset),len(dev_dataset))

    kk
    # evaluate=pegasus_evaluate
    # pegasus_train(tokenizer, model_name, pegasus_config, train_dataset, dev_dataset, test_dataset, evaluate)

    print('--------开始推断 %s --------'%(model_name))
    content= '在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！'
    summary=Gpt2Pred().infer(content,gpt2_tokenizer)
    '''
    #=======Gpt2 model end ==============

choice_model()









