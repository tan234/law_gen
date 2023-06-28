import pandas as pd
import torch
import torch.nn as nn
import numpy as np
# criterion = nn.CrossEntropyLoss()  # 使用ignore_index参数，使得计算损失的时候不计算pad的损失

# LayerNorm

from nltk.translate.bleu_score import sentence_bleu
from rouge  import Rouge

import jieba
import json
from config import *

projected =torch.tensor([[[-0.1292,  1.3807, -0.0862, -4.0357, 15.7966, -1.2049],
         [14.7269,  1.7715,  1.0128, -5.0443,  0.0181,  0.4881],
         [14.9428,  1.3934,  0.9537,   -5.4548, -0.1483,  0.3296],
         [14.9024,  1.7100,  1.2857,  -4.2892, -0.1035, -0.0721],
         [14.6627,  1.4085,  1.4940,  -5.8918, -0.7217,  0.3800],
         [14.9593,  1.7859,  1.0038,   -4.7113, -0.4765,  0.1360]]])

with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
    vocab = json.load(f)
print(vocab['S'])
print(vocab['E'])

# print([i[:-2]+i[-1]  for i in projected.data])

# print(torch.add((projected[:,:-2],projected[:,-2]),1))
# print(projected.size())

# with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
#     vocab = json.load(f)
# print(len(vocab))
# print(vocab)
# # print(vocab['S'])
# print(vocab['E'])

# batch=2
# input_len=5
# encoder_input=torch.zeros(batch,input_len)
# tgt_l=6
# a=torch.zeros(1,tgt_l)
# print(a)
# print(encoder_input)
#
# print((encoder_input.data))
# y_hat=y_hat.view(-1,y_hat.size()[-1])#[batch_size*tgt_len, tgt_vocab_size]
# y=y.view(-1)#真实值平铺 batch_size*tgt_len

# batch=1
# tgt_len=2
# vocab=3
# y_hat=torch.tensor([[2,4,9],[10,0.5,7]])
# y=[0,2]
# # y_hat.argmax(dim=1) == y
# y=' '.join([str(i)for i in y])
# y_hat=y_hat.argmax(dim=1)
# y_hat= ' '.join([str(i)for i in y_hat.tolist()])

# def rouge_value(y_pred,y_true):
#     rouge = Rouge()
#
#     # y_pred=" ".join(jieba.cut(y_pred))
#     # y_true=" ".join(jieba.cut(y_true))
#
#     print(y_pred,y_true)
#     rouge_scores = rouge.get_scores(y_pred,y_true)  # "Installing collected packages", "Installing "
#     # print('rouge_scores:', rouge_scores)
#     rouge_f = [rouge_scores[0][k]['f'] for k in rouge_scores[0]]
#     score = 0.2 * rouge_f[0] + 0.3 * rouge_f[1] + 0.5 * rouge_f[2]
#     # rl_p = rouge_scores[0]['rouge-l']['p']
#     print("score", score)
#     return score
#
#
# text= "记者从元谋县人民法院获悉,10月30日下午,元谋县人民法院在第一审判法庭,依法对元谋县人民检察院提起公诉的被告人金某甲、金某乙等58名被告人犯组织、领导、参加黑社会性质组织罪等10项罪名案一审公开宣判。主犯金某甲犯组织、领导黑社会性质组织罪,抢劫罪、强迫交易罪、敲诈勒索罪、" \
#       "寻衅滋事罪、开设赌场罪、非法拘禁罪、窝藏罪,数罪并罚,判处有期徒刑二十五年,剥夺政治权利五年,并处没收个人全部财产;主犯金某乙犯组织、" \
#       "领导黑社会性质组织罪,故意伤害罪、强迫交易罪、寻衅滋事罪、开设赌场罪、非法侵入住宅罪,数罪并罚,判处有期徒刑二十年,剥夺政治权利三年" \
#       ",并处没收个人全部财产;其余被告人分别被判处有期徒刑十六年及以下不等的刑罚,并处没收个人全部财产或没收个人部分财产或处24万元至5000元不等的罚金" \
#       "。法院审理认为,以被告人金某甲、金某乙为组织者、领导者,张某明、陈某国、思某金、康某等被告人参加的黑社会性质犯罪组织,多年来在元谋县实施抢劫、" \
#       "故意伤害、强迫交易、敲诈勒索、寻衅滋事、开设赌场、非法拘禁、非法侵入住宅、窝藏包庇犯罪行为,该犯罪组织人数众多,组织结构严密稳定" \
#       ",藐视国家法律,欺压百姓,为非作恶,非法攫取巨额经济利益,严重扰乱了蔬菜水果交易市场和社会管理秩序,严重侵害了公民的人身权利和民主权利," \
#       "造成恶劣的社会影响和严重的社会危害后果。该案是元谋县开展扫黑除恶专项斗争以来社会影响最广,涉案人数和资产最多,案情最为复杂," \
#       "人民群众最为关心关注的一起黑社会性质组织犯罪案件。案件的成功审判,实现了政治效果、法律效果和社会效果的有机统一,为建设平安元谋、" \
#       "美丽元谋营造了良好的环境。县人大代表、政协委员和社会各界代表60余人参加了案件宣判。(闵以荣)"
# summary="元谋县人民法院在第一审判法庭,依法对元谋县人民检察院提起公诉的被告人金某甲、金某乙等58名被告人犯组织、" \
#         "领导、参加黑社会性质组织罪等10项罪名案一审公开宣判。该案是元谋县开展扫黑除恶专项斗争以来社会影响最广,涉案人数和资产最多," \
#         "案情最为复杂,人民群众最为关心关注的一起黑社会性质组织犯罪案件。案件的成功审判,实现了政治效果、法律效果和社会效果的有机统一," \
#         "为建设平安元谋、美丽元谋营造了良好的环境。"
#
# rouge_value('0 13 16 17 18 9','9 13 16 17')
# rouge_value('今天 我 是 一只 猫 吗','吗 我 是 一只')
# rouge_value(y_hat,y)

# a.size()=(3,2);参数2 表示维度，的到 3 个均值方差mean（2+4）/var(2,4)
# 参数为两个维度（3，2），得到一个方差

# https://blog.csdn.net/weixin_39228381/article/details/107939602
# import numpy as np
# attn_shape=[5,3,3]
# subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # [batch_size, tgt_len, tgt_len]
# print(subsequence_mask)
# subsequence_mask = torch.from_numpy(subsequence_mask).byte()  # 转化成byte类型的tensor
# print(subsequence_mask)

# print(a)
# print(b)
# print(a.size(),b.size())
# seq_len=7
# emb_size=5
# a=[[[1,2,3,4,],[6,7,8,9]],[[1,1,1,1,],[2,2,2,2]]]
# # pe = torch.ones(2,4,4)
# batch=3
# seq=4
# emb=5
# X=torch.tensor(a).float()
# print(X,X.size())#2,2,4
# print(nn.LayerNorm(4)(X))
# print(pe,pe.size())
#
# dropout=nn.Dropout(p=0.5)
# a=dropout(pe)
# print(a,a.size())
# kk
# batch_size=4
# len_q=2
# len_k=3
# seq_k=torch.rand(4,3)
# seq_k[1][1:]=0
# seq_k[3][2:]=0
#
# print(seq_k[1])
#
# pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
# # print(pad_attn_mask,pad_attn_mask.size())
# pad_attn_mask=pad_attn_mask.expand(batch_size, len_q, len_k)
# print(pad_attn_mask[1])

# return pad_attn_mask.expand(batch_size, len_q, len_k)
