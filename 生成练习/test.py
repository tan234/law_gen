# -*- coding:utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
# criterion = nn.CrossEntropyLoss()  # 使用ignore_index参数，使得计算损失的时候不计算pad的损失
import pickle
import paddle
from torch.utils.data import TensorDataset, DataLoader

# data_path='data/gpt_train.pkl'
# with open(data_path, "rb") as f:
#     data_list = pickle.load(f)
# # data: [[[contentid], [labid], len],, , , ]
#
#     data = data_list
# data=data[:5]
# a=torch.tensor([i[0] for i in data])
# b=torch.tensor([i[1] for i in data])
# # X, Y, l = torch.tensor([i[0] for i in data]), paddle.to_tensor([i[1] for i in data]), [i[2] for i in data]
# c=torch.tensor([i[2] for i in data])
#
# print(a.size())
# print(b.size())
# print(a[0])
# print(b[0])
# print(c[0])


# data_set = TensorDataset(a,b,c)
# print(data_set)

# def aa():
#     data_loader = DataLoader(dataset=data_set,
#                              batch_size=10,
#                              shuffle=False)
#     # print(i[0])
#     # print(len(i[0]),len(i[1]),i[2])
#
#     return data_loader
# for i in range(2):
#
#     print(aa())


# print(data)
# print(len(data), len(data[0]))
# print(data[0])
# print(data[:,0])
# print(torch.tensor([data]))
# paddle.to_tensor([data[:][0]])
# LayerNorm

from nltk.translate.bleu_score import sentence_bleu
from rouge  import Rouge

import jieba
import json
from config import *
import torch
import paddle
from paddle import fluid
from paddlenlp.transformers import GPTTokenizer
import pickle
from gpt2_model import Gpt2Model
from paddle.io import Dataset
import paddle
import pickle
#


a=30
b=29
if a%b!=0:
    c=a//b+1
else:c=a//b
print(c)
k
# [{'rouge-1': {'r': 0.5, 'p': 0.5, 'f': 0.4999999950000001}, 'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 'rouge-l': {'r': 0.5, 'p': 0.5, 'f': 0.4999999950000001}}]

score = Rouge().get_scores(' '.join(str(i)for i in pred), ' '.join(str(i) for i in target))

print(score)
kk
gpt2_tokenizer = GPTTokenizer.from_pretrained('gpt2-medium-en')# 使用的是gpt2-mediu-en分词器
gpt2_tokenizer.add_special_tokens({"sep_token": "<sep>"})# 添加一个特殊字符来区分内容和摘要<>
en=gpt2_tokenizer.encode('的骄傲了')
a=en['input_ids']
print(a,type(a))
print(gpt2_tokenizer.decode(a))
# b=list()
print(gpt2_tokenizer.decode([1009,79,101, 2, 6378,228]))


kk
print(gpt2_tokenizer.convert_ids_to_string)

print(gpt2_tokenizer.convert_ids_to_string([101, 7564, 6378]))
print(gpt2_tokenizer.decode([101, 7564, 6378]))
kk
loss = nn.CrossEntropyLoss(
                ignore_index=0)  # 使用ignore_index参数：可以忽略某一项y的损失，一般用于labelPAD的情况。，还有一个weihgt参数，可以给label加权重
# print(out_list,label)
seq_list=[17312, 222  , 165  , 45865, 37345]
s=[17312, 222  , 165  , 45865, 35 ]
print(' '.join(seq_list))
# [17312, 222  , 165  , 45865, 37345, 243  , 171  , 120  , 248  , 22887,
#         232  , 34932, 235  , 161  , 240  , 234  , 46479, 251  , 49694, 50257])
# <class 'list'> [20227, 44330, 36950, 16144, 28977, 45470, 14591, 46962, 22013, 41839, 17419, 34444, 20793, 26629, 29599, 2621, 44706, 39504, 32723, 36276]

rouge_score = Rouge().get_scores(''.join(seq_list), ''.join(s))
print(rouge_score)
# print()
# print(loss(torch.tensor(out_list), torch.tensor(label)))

# # print('-----------')
# # print(a[0],a[0].size())
# # print('-----------')
# # print(a[:,:,0])
# for i in range(3):
#     for j in range(2):
#         a[i][j][:]=i+j
#         print(a[i][j][:])
#         print('---------------------')
# print(a)
# a[:,:,:]
# GptData()
# filtered_logits=torch.tensor([0.9000, 0.9800, 0.6500,   -float('Inf'),  -float('Inf'), 0.4800,   -float('Inf')])
# cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # softmax操作后，累计计算概率分布
# next_token = torch.multinomial(filtered_logits.softmax(dim=-1), num_samples=1)

# next_token = paddle.multinomial(fluid.layers.softmax(filtered_logits, axis=-1), num_samples=1)

# print(filtered_logits)
# print(next_token)
# tran_data = MyDataset('data/gpt_train.pkl')


# for i, data in enumerate(tran_data):
#     print(i)
#     print(data)
#     kk




# data=pd.DataFrame([{'content':'会发你发附加赛佛开票','title':'和大'},{'content':'都还是金广发女法律方式看见你','title':'大家发疯'}])
# print(data)
# content_id=[gpt2_tokenizer.sep_token_id]
# print()
# c = gpt2_tokenizer(data['content'].to_list(),return_token_type_ids=False)["input_ids"]
# content_id.extend(c)
# print(content_id)
#
# length = len(content_id)
# content_id.append(gpt2_tokenizer.sep_token_id)
# t = gpt2_tokenizer(data['title'].to_list(),return_token_type_ids=False)["input_ids"]
# content_id.extend(t[:100])
# print(content_id)
# label = t[:100]
# label.append(gpt2_tokenizer.sep_token_id)
# print(label)
# all_data.append([content_id, label, length])

from paddle.io import Dataset
import paddle
import pickle



# input_id = [gpt2_tokenizer.sep_token_id]#'sep'对应的编码input_id=[50257]
# input_id.extend(gpt2_tokenizer(inputs)["input_ids"][:400])
# print(input_id)
# input_id.append(gpt2_tokenizer.sep_token_id)
# print(input_id)
#
# input_id = paddle.to_tensor([input_id])
# print(input_id)

# print(gpt2_tokenizer)
# model_inputs = gpt2_tokenizer(inputs,
#                                  max_length=100,
#                                  padding=False,
#                                  truncation=True,
#                                  return_attention_mask=True)
# print(model_inputs)
# print(len(model_inputs['input_ids']))
# print(gpt2_tokenizer['sep'])

# print(gpt2_tokenizer.sep_token_id)
# top_k=5
# logits=torch.tensor([0.9,0.98,0.65,0.01,0.32,0.48,0.12])#
# next_token_logits = logits[0, -1, :]
#
# print('logits:7',logits)
# print(next_token_logits)
#
# indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
# # print(indices_to_remove)
#
# logits[indices_to_remove] = -float('Inf')  # 对于topk之外的其他元素的logits值设为负无穷,tensor([0.9000, 0.9800,   -inf,   -inf, 0.4800,   -inf])
# print('top_k=5:',logits)
#
# print('----')
#
# sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
# print(sorted_logits)
# print(sorted_indices)
# print('---------------')
#
# cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # softmax操作后，累计计算概率分布
# # sorted_logits.softmax(dim=-1):tensor([0.2656, 0.2452, 0.1909, 0.1611, 0.1373, 0.0000, 0.0000])
# # cumulative_probs:tensor([0.2656, 0.5107, 0.7017, 0.8627, 1.0000, 1.0000, 1.0000])
#
# print('cum:',cumulative_probs)
# print('---------------')
# # Remove tokens with cumulative probability above the threshold
# sorted_indices_to_remove = cumulative_probs > 0.8#过滤
# # tensor([False, False, False,  True,  True,  True,  True])
#
# # print(sorted_indices_to_remove)
#
# # Shift the indices to the right to keep also the first token above the threshold
# sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
# sorted_indices_to_remove[..., 0] = 0
#
# # print(sorted_indices_to_remove)
# indices_to_remove = sorted_indices[sorted_indices_to_remove]
# # print(indices_to_remove)
# logits[indices_to_remove] = -float('Inf')
# print(logits)

# logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
# print(logits)

# enc_input=torch.zeros(1,5)
# enc_outputs = torch.zeros(1,5,3)
# dec_input = torch.zeros(1, 1).type_as(enc_input.data)
# print(dec_input.size(),dec_input)
# print(dec_input.detach())#不计算梯度,next_symbol是1,1
# next_symbol = 3
# a=torch.tensor([[next_symbol]])
# print(a,a.size())
#
# dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype)],-1)#类似append
# print(dec_input)
    #     dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
    #     projected = model.projection(dec_outputs)
    #     prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
    #     next_word = prob.data[-1]
    #     next_symbol = next_word
    #     if next_symbol == tgt_vocab["."]:
    #         terminal = True
    #     print(next_word)
    # return dec_input

# projected =torch.tensor([[[-0.1292,  1.3807, -0.0862, -4.0357, 15.7966, -1.2049],
#          [14.7269,  1.7715,  1.0128, -5.0443,  0.0181,  0.4881],
#          [14.9428,  1.3934,  0.9537,   -5.4548, -0.1483,  0.3296],
#          [14.9024,  1.7100,  1.2857,  -4.2892, -0.1035, -0.0721],
#          [14.6627,  1.4085,  1.4940,  -5.8918, -0.7217,  0.3800],
#          [14.9593,  1.7859,  1.0038,   -4.7113, -0.4765,  0.1360]]])
#
# with open(data_config['vocab_path'], "r", encoding="utf-8") as f:
#     vocab = json.load(f)
# print(vocab['S'])
# print(vocab['E'])

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
