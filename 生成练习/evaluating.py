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

from paddle import fluid

from paddle import fluid
from paddlenlp.transformers import GPTModel, GPTTokenizer
from rouge import Rouge
from paddlenlp.datasets import load_dataset
from tqdm import tqdm
import torch

from rouge  import Rouge
import jieba

# ---------gpt2_evaluate-----------

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    '''
    功能：对预测的结果进行过滤；模型预测通过top-k采样，每一步生成一个词不再是概率最大的一个（贪心搜索），从概率最大的k个中随机采样一个，这样子生成的效果不会太死板。
    top-k采样，模型会从概率前 k 大的单词中抽样选取下一个单词
    Top-p采样，设定概率阈值，取满足阈值条件的样本进行采样
    目的：先选择概率top-k个,然后对于这top-k选择累计贡献达到的top-p的前几个；其他位置的值为-inf
    输入:logits: tensor([0.9000, 0.9800, 0.6500, 0.0100, 0.3200, 0.4800, 0.1200]),topk=5,top_p=0.8
    输出：tensor([0.9000, 0.9800, 0.6500,   -inf,   -inf, 0.4800,   -inf])

    '''

    assert logits.dim() == 1  # []一维tensor
    top_k = min(top_k, logits.shape[-1])  # Safety check,logits长度，是否小于topk
    # logits: tensor([0.9000, 0.9800, 0.6500, 0.0100, 0.3200, 0.4800, 0.1200]),topk=5
    # print(type(logits),logits)
    if top_k > 0:
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices) # ...表示其他维度由计算机自行推断
        # print(paddle.topk(logits, top_k))
        indices_to_remove = logits < paddle.topk(logits, top_k)[0][..., -1, None]


        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷
        # tensor([0.9000, 0.9800, 0.6500, -inf, 0.3200, 0.4800, -inf])

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        # sorted_logits:tensor([0.9800, 0.9000, 0.6500, 0.4800, 0.3200, -inf, -inf])
        # sorted_indices：tensor([1, 0, 2, 5, 4, 3, 6])

        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # softmax操作后，累计计算概率分布
        # sorted_logits.softmax(dim=-1):tensor([0.2656, 0.2452, 0.1909, 0.1611, 0.1373, 0.0000, 0.0000])
        # cumulative_probs:tensor([0.2656, 0.5107, 0.7017, 0.8627, 1.0000, 1.0000, 1.0000])# 累计计算概率分布

        sorted_indices_to_remove = cumulative_probs > top_p
        # sorted_indices_to_remove:[F,F,,,]

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

# def gpt2_per(tokenizer,config,model,news,target):
#
#     # input_id = [tokenizer.sep_token_id]  # 'sep'对应的编码input_id=[50257]
#     # input_id.extend(tokenizer(news)["input_ids"][:400])
#     # input_id.append(tokenizer.sep_token_id)  # [sep ,line ,sep]
#     # input_id = paddle.to_tensor([input_id])  # tensor:1*[seq(line)+2]
#     #
#     # response = []
#
#     for _ in range(config['max_len']):
#         logits = model(input_id)  # 预测结果shape=[1, 29, 50258]
#
#         next_token_logits = logits[0, -1, :]  # shape=[ 50258]，取最后一个step的预测结果
#
#         # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
#         # print(next_token_logits.shape)
#
#         for id in set(response):
#             next_token_logits[id] /= config['repetition_penalty']
#         next_token_logits = next_token_logits / config['temperature']
#         # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
#         next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
#
#         filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=config['top_k'], top_p=0)
#
#         # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
#         next_token = torch.multinomial(filtered_logits.softmax(dim=-1), num_samples=1)
#
#         if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
#             break
#         response.append(next_token.item())
#
#         input_id = paddle.cat((input_id, next_token.unsqueeze(0)), axis=1)  # 将预测结果拼接到输入序列
#         # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
#         # print("his_text:{}".format(his_text))
#     # history.append(response)
#     text = tokenizer.convert_ids_to_string(response)
#
#     rouge_score = Rouge().get_scores(text, target)
#     score1 = rouge_score[0]["rouge-1"]["p"]
#     score2 = rouge_score[0]["rouge-2"]["p"]
#     score3 = rouge_score[0]["rouge-l"]["p"]
#     return score1, score2, score3

# def gpt2_evaluate(tokenizer,dev_data_loader,news,target,config,model):
#     '''
#
#     '''
#
#     # for i, data in enumerate(dev_data_loader):
#     #     content, label, lenght = data
#     #     out = model(content)# shape=[batch, 29, 50258]
#     #     out = out[:, -1:, :]# shape=[ 50258]，取最后一个step的预测结果
#
#
#         # loss = paddle.nn.functional.cross_entropy(out, label)
#         # print(f"epoch:{epoch} step:{i} loss:{loss.item()}")
#     #new:?
#     input_id = [tokenizer.sep_token_id]#'sep'对应的编码input_id=[50257]
#     input_id.extend(tokenizer(news)["input_ids"][:400])
#     input_id.append(tokenizer.sep_token_id)# [sep ,line ,sep]
#     input_id = paddle.to_tensor([input_id])#tensor:1*[seq(line)+2]
#
#
#     response = []
#
#     for _ in range(config['max_len']):
#         logits = model(input_id)#预测结果shape=[1, 29, 50258]
#
#         next_token_logits = logits[0, -1, :]#shape=[ 50258]，取最后一个step的预测结果
#
#         # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
#         #print(next_token_logits.shape)
#
#         for id in set(response):
#             next_token_logits[id] /= config['repetition_penalty']
#         next_token_logits = next_token_logits / config['temperature']
#         # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
#         next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
#         filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=config['top_k'], top_p=0)
#
#         # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
#         next_token = torch.multinomial(filtered_logits.softmax(dim=-1), num_samples=1)
#
#         if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
#             break
#         response.append(next_token.item())
#
#         input_id = paddle.cat((input_id, next_token.unsqueeze(0)), axis=1)# 将预测结果拼接到输入序列
#         # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
#         # print("his_text:{}".format(his_text))
#     #history.append(response)
#     text = tokenizer.convert_ids_to_string(response)
#
#     rouge_score = Rouge().get_scores(text,target)
#     score1 = rouge_score[0]["rouge-1"]["p"]
#     score2 = rouge_score[0]["rouge-2"]["p"]
#     score3 = rouge_score[0]["rouge-l"]["p"]
#     return score1,score2,score3

def gpt2_out_choice(out,tokenizer):
    '''
    out:shape = [seq, vocab]
    res:shape = [seq]
    目的：给一条样本的模型结果返回一个预测值
    '''
    # print(out)
    # print(out.shape)
    res=[]
    # print(out.shape)

    for seq in range(out.shape[0]):  # seq
        # print(out[seq][:])
        s=top_k_top_p_filtering(out[seq][:], top_k=5, top_p=0.0, filter_value=-float('Inf'))
        # print(s)
        next_token = paddle.multinomial(paddle.nn.functional.softmax(s, axis=-1), num_samples=1)
        # print(next_token)
        # next_token = torch.multinomial(s.softmax(dim=-1), num_samples=1)#选择一个id作为预测结果
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        res.append(next_token.item())
    # print(res)

    return res
def gpt2_evaluate(model,tokenizer,dev_data_loader,config,dev_len):

    r1, r2, rl,count = 0, 0, 0,0
    for i,data in enumerate(dev_data_loader):#batch
        content, label, lenght = data
        # print(content)
        out = model(content.squeeze(axis=1))#[10, 171, 50258]
        # print(out)
        # out = out[:, lenght:, :]# 预测结果vocab
        # out.shape()

        # out_list = []
        # for i in range(len(lenght)):
        #     out_per = out[i, lenght[i]:, :]
        #     # out_per=gpt2_out_choice(out_per,tokenizer)
        #     # print('out_per:',out_per)
        #     out_list.append(out_per)
        #     lenght[i] = lenght[i][:len(out_per)]

        # response=[]
        # 选top5
        '''rL值是对一条进行计算，这里是先得到每条样本的预测值（所有sep），然后去计算'''
        # print(out.shape)
        # print(lenght)
        for batch in range(out.shape[0]):#一个样本0-10
            seq_list=[]#存储每个sep预测值
            # for sep in range(len(lenght)):
            for seq in range(out.shape[1]-lenght[batch]):
                # print(seq+lenght[seq])

                per=out[batch][seq+lenght[batch]][:]#[50258]
                per= top_k_top_p_filtering(per, top_k=5, top_p=0.0, filter_value=-float('Inf'))
                # printper)

                next_token = paddle.multinomial(paddle.nn.functional.softmax(per, axis=-1), num_samples=1)

                # next_token = torch.multinomial(per.softmax(dim=-1), num_samples=1)
                if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
                    break
                seq_list.append(next_token.item())

            # seq_list:[31779, 30059, 17742, 43850, 18479, 27823, 11209, 42608, 7963, 20863, 46090, 25769, 907, 43549, 28351, 23061, 20470, 14265, 42053, 14383]
            # text = tokenizer.convert_ids_to_string(seq_list)
            label=label.squeeze(1)
            # print(type(label),label[batch])
            # print(type(seq_list),seq_list)
            # print(tokenizer.convert_ids_to_string(label))
            # print(tokenizer.convert_ids_to_string(seq_list))
            target=label[batch].tolist()

            rouge_score = Rouge().get_scores(' '.join(str(i) for i in seq_list), ' '.join(str(i) for i in target))

            # rouge_score = Rouge().get_scores(seq_list, )
            score1 = rouge_score[0]["rouge-1"]["p"]
            score2 = rouge_score[0]["rouge-2"]["p"]
            scorel = rouge_score[0]["rouge-l"]["p"]
            # print(score1,score2,scorel)
            # kk
            # s1, s2, s3 = gpt2_per(content, label)
            r1 += score1
            r2 += score2
            rl += scorel
            count += 1

    # print('he')


        c=dev_len//config['batch_size']+1 if dev_len%config['batch_size']!=0 else dev_len//config['batch_size']

        if i==c-1:
            break
    print(f"r1:{r1 / count},r2:{r2 / count},r3:{rl / count}")
    return r1,r2,rl


# ---------pegasus_evaluate-----------
@paddle.no_grad()
def pegasus_evaluate(model, data_loader, count,config,tokenizer):
    # print('--h1')

    model.eval()
    all_preds = []
    all_labels = []
    model = model._layers if isinstance(model, paddle.DataParallel) else model

    # for data in data_loader:

    for step, data in enumerate(data_loader):

        # for data in tqdm(data_loader, total=len(data_loader), desc="Eval step"):

        labels = data.pop('labels').numpy()
        # 模型生成
        preds = model.generate(input_ids=data['input_ids'],
                               attention_mask=data['attention_mask'],
                               min_length=config['min_target_length'],
                               max_length=config['max_target_length'],
                               use_cache=True)[0]
        # tokenizer将id转为string
        all_preds.extend(
            tokenizer.batch_decode(preds.numpy(),
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False))
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # print('--he2')
        # print('labels',labels)
        all_labels.extend(
            tokenizer.batch_decode(labels,
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False))

        if step == count // config['batch_size'] - 1:
            break

    rouge1, rouge2, rougel, bleu4 = compute_metrics(all_preds, all_labels)
    model.train()
    return rouge1, rouge2, rougel, bleu4

def compute_metrics(preds, targets):
    # 计算训练评估参数Rouge-1，Rouge-2，Rouge-L，BLEU-4
    assert len(preds) == len(targets), (
        'The length of pred_responses should be equal to the length of '
        'target_responses. But received {} and {}.'.format(
            len(preds), len(targets)))
    rouge = Rouge()

    bleu4 = BLEU(n_size=4)
    scores = []
    for pred, target in zip(preds, targets):
        try:
            score = rouge.get_scores(' '.join(pred), ' '.join(target))
            scores.append([
                score[0]['rouge-1']['f'], score[0]['rouge-2']['f'],
                score[0]['rouge-l']['f']
            ])
        except ValueError:
            scores.append([0, 0, 0])
        bleu4.add_inst(pred, [target])
    # print(scores)
    rouge1 = np.mean([i[0] for i in scores])
    rouge2 = np.mean([i[1] for i in scores])
    rougel = np.mean([i[2] for i in scores])
    bleu4 = bleu4.score()
    print('\n' + '*' * 15)
    print('The auto evaluation result is:')
    print('rouge-1:', round(rouge1 * 100, 2))
    print('rouge-2:', round(rouge2 * 100, 2))
    print('rouge-L:', round(rougel * 100, 2))
    print('BLEU-4:', round(bleu4 * 100, 2))
    return rouge1, rouge2, rougel, bleu4


# ---------transformer_evaluate-----------
def rouge_value(y_true,y_pred):
    rouge = Rouge()
    rouge_scores = rouge.get_scores(y_pred,y_true)

    # print('rouge_scores:', rouge_scores)
    rouge_f=[rouge_scores[0][k]['f'] for k in rouge_scores[0]]
    score=0.2*rouge_f[0]+0.3*rouge_f[1]+0.5*rouge_f[2]
    # rl_p = rouge_scores[0]['rouge-l']['p']
    # print("score", score)
    return score
def dev_evaluate( data_iter,net):
    train_rouge, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:

            # if isinstance(net, torch.nn.Module):
            net.eval()  # 评估模式, 会关闭dropout

            n += y.shape[0]
            y_hat,_,_,_, = net.forward(X, y)  # [batch_size, tgt_len, tgt_vocab_size]

            # 模型评估指标计算
            y = ' '.join([str(i) for i in y.view(-1).tolist()])# 真实值平铺 batch_size*tgt_len
            y_hat = y_hat.view(-1, y_hat.size()[-1])  # [batch_size*tgt_len, tgt_vocab_size]
            y_hat = ' '.join([str(i) for i in y_hat.argmax(dim=1).tolist()])
            # print(y_hat)
            train_rouge += rouge_value(y, y_hat)

            net.train()  # 改回训练模式

    return train_rouge / n

def trans_eval(y_pred,y_true):
    rouge = Rouge()

    rouge_scores = rouge.get_scores(" ".join(jieba.cut(y_pred)),
                                    " ".join(jieba.cut(y_true)))  # "Installing collected packages", "Installing "
    # print('rouge_scores:', rouge_scores)
    rouge_f = [rouge_scores[0][k]['f'] for k in rouge_scores[0]]
    score = 0.2 * rouge_f[0] + 0.3 * rouge_f[1] + 0.5 * rouge_f[2]
    # rl_p = rouge_scores[0]['rouge-l']['p']
    # print("score", score)
    return score
