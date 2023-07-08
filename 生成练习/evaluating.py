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


# ---------gpt2_evaluate-----------

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.shape[-1])  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < paddle.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = paddle.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = paddle.cumsum(fluid.layers.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def gpt2_evaluate(tokenizer,news,target,config,model):

    input_id = [tokenizer.sep_token_id]
    input_id.extend(tokenizer(news)["input_ids"][:400])
    input_id.append(tokenizer.sep_token_id)
    input_id = paddle.to_tensor([input_id])
    #logits = net(paddle.to_tensor([input_id]))
    response = []
    for _ in range(config['max_len']):
        logits = model(input_id)


        next_token_logits = logits[0, -1, :]
        # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
        #print(next_token_logits.shape)

        for id in set(response):
            next_token_logits[id] /= config['repetition_penalty']
        next_token_logits = next_token_logits / config['temperature']
        # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
        #next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=5, top_p=0)
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = paddle.multinomial(fluid.layers.softmax(filtered_logits, axis=-1), num_samples=1)
        if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明response生成结束
            break
        response.append(next_token.item())
        input_id = paddle.concat((input_id, next_token.unsqueeze(0)), axis=1)
        # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
        # print("his_text:{}".format(his_text))
    #history.append(response)
    text = tokenizer.convert_ids_to_string(response)

    rouge_score = Rouge().get_scores(text,target)
    score1 = rouge_score[0]["rouge-1"]["p"]
    score2 = rouge_score[0]["rouge-2"]["p"]
    score3 = rouge_score[0]["rouge-l"]["p"]
    return score1,score2,score3

def gpt2():
    tokenizer = GPTTokenizer.from_pretrained("gpt2-medium-en")
    tokenizer.add_special_tokens({"sep_token": "<sep>"})
    rouge = Rouge()
    max_len = 50
    repetition_penalty = 1.0
    temperature = 1

    net = MyModel(tokenizer.vocab_size + 1)
    net_dic = paddle.load("./model/model_2_262500.pkl")
    net.set_state_dict(net_dic)

    train_set, dev_set, test_set = load_dataset("cnn_dailymail", splits=["train", "dev", "test"])

    s1_all, s2_all, s3_all = 0, 0, 0
    for data in tqdm(test_set):
        content = data["article"].lower()
        title = data["highlights"].lower()
        s1, s2, s3 = eval(content, title)
        s1_all += s1
        s2_all += s2
        s3_all += s3
    print(f"r1:{s1_all / len(test_set)},r2:{s2_all / len(test_set)},r3:{s3_all / len(test_set)}")


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