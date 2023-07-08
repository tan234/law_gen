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
from gpt2_model import GptModel
import paddle
from paddlenlp.ops.optimizer import AdamWDL
import time
# from paddlenlp.utils.log import logger
from paddlenlp.metrics import BLEU
from tqdm import tqdm
import numpy as np
import os
from paddlenlp.transformers import GPTTokenizer

from config import *

from data import DealData

from evaluating import pegasus_evaluate
# ------模型训练------

def pegasus_train(tokenizer,model_name,config,train_dataset,dev_dataset,test_dataset,evaluate):

    # ----2 初始化模型---------------
    model = AutoModelForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese')

    # 初始化模型。AutoModelForConditionalGeneration封装实现了Pegasus用于生成任务的模型结构，可参考：https://github.com/PaddlePaddle/PaddleNLP/blob/v2.4.2/paddlenlp/transformers/pegasus/modeling.py#L533
    # 模型的输入包括：
    # input_ids: content分词后的id，模型的Decoder的输入
    # attention_mask: 对content分词后的id进行padding，在模型中去除padding带来的影响
    # labels: title分词后的id，模型的label
    # decoder_input_ids：title分词后并进行移位的id，模型的Decoder的输入
    # ----3 组装 Batch ----------------
    # 组装 Batch 数据 & Padding。DataCollatorForSeq2Seq用于对数据进行动态Paddding，可参考：https://github.com/PaddlePaddle/PaddleNLP/blob/v2.4.2/paddlenlp/data/data_collator.py#L319
    batchify_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    # 可视化
    # log_writer = LogWriter('visualdl_log_dir')

    # -------构造Dataloader----------
    # 分布式批采样器，用于多卡分布式训练
    train_batch_sampler = DistributedBatchSampler(
        train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 构造训练训练集Dataloader
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_sampler=train_batch_sampler,
                                   num_workers=0,
                                   collate_fn=batchify_fn,
                                   return_list=True)

    dev_batch_sampler = BatchSampler(dev_dataset,
                                     batch_size=config['batch_size'],
                                     shuffle=False)
    # 构造验证验证集Dataloader
    dev_data_loader = DataLoader(dataset=dev_dataset,
                                 batch_sampler=dev_batch_sampler,
                                 num_workers=0,
                                 collate_fn=batchify_fn,
                                 return_list=True)
    test_batch_sampler = BatchSampler(test_dataset,
                                      batch_size=1,
                                      shuffle=False)
    # 构造验证测试集Dataloader
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_sampler=test_batch_sampler,
                                  num_workers=0,
                                  collate_fn=batchify_fn,
                                  return_list=True)
    # test_data_loader:
    # 每条数据{
    # 'input_ids':Tensor[1, 69]
    # 'attention_mask':Tensor[1, 69],
    # 'labels':Tensor[1, 16],
    # 'decoder_input_ids':Tensor[1, 16]}



    # # ----------模型参数--------------：
    num_epochs=config['num_epochs']
    # min_target_length=config['min_target_length']
    # max_target_length=config['max_target_length']
    # # len(dev_dataset)

    num_training_steps = len(train_data_loader) * num_epochs  # 训练总步数
    # # LayerNorm参数不参与weight_decay
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    lr_scheduler = LinearDecayWithWarmup(config['learning_rate'], num_training_steps, config['warmup_proportion'])
    #
    # # 优化器AdamW
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=config['adam_epsilon'],
        parameters=model.parameters(),
        weight_decay=config['weight_decay'],
        apply_decay_param_fun=lambda x: x in decay_params)

    # 模型训练
    train(tokenizer,model, train_data_loader,optimizer,lr_scheduler,num_training_steps,num_epochs,dev_data_loader,len(dev_dataset),len(train_dataset),model_name,config,evaluate)

    # 模型评估
    evaluate(model, test_data_loader,len(test_dataset))


def gpt2_train(tokenizer,config,model,train_dataset,model_name):

    name_dict = dict()

    adam = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=config['learning_rate'])


    for epoch in range(config['num_epochs']):
        for i, data in enumerate(train_dataset):
            content, label, lenght = data
            out = model(content)
            out = out[:, lenght:, :]
            loss = paddle.nn.functional.cross_entropy(out, label)
            print(f"epoch:{epoch} step:{i} loss:{loss.item()}")
            loss.backward()
            # 梯度累计更新
            if i % config['batch_size'] == 0:
                adam.step()
                adam.clear_grad()
            if i % 500 == 0:
                # paddle.save(model.state_dict(), f'model/model_{epoch}_{i}.pkl')
                paddle.save(model.state_dict(), config['model_save_path'])


def train(tokenizer,model, train_data_loader,optimizer,lr_scheduler,num_training_steps,num_epochs,dev_data_loader,dev_len,train_len,model_name,config,evaluate):
    global_step = 0
    best_rougel = 0
    tic_train = time.time()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_data_loader):

            global_step += 1
            # 模型前向训练，计算loss
            lm_logits, _, loss = model(**batch)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            print('--------epoch%d,step%d,loss:%f------' % (epoch, step, loss))
            # print('----')
            # if global_step % log_steps == 0:
            # logger.info(
            #     "global step %d/%d, epoch: %d, batch: %d, rank_id: %s, loss: %f, lr: %.10f, speed: %.4f step/s"
            #     % (global_step, num_training_steps, epoch, step,
            #        paddle.distributed.get_rank(), loss, optimizer.get_lr(),
            #        log_steps / (time.time() - tic_train)))
            # log_writer.add_scalar("train_loss", loss.numpy(), global_step)
            # tic_train = time.time()
            if global_step % config['eval_steps'] == 0 or global_step == num_training_steps:
                tic_eval = time.time()
                rouge1, rouge2, rougel, bleu4 = evaluate(model, dev_data_loader,dev_len)


                # print(rouge1, rouge2, rougel, bleu4)
                # logger.info("eval done total : %s s" % (time.time() - tic_eval))
                # log_writer.add_scalar("eval/ROUGE-1", round(rouge1 * 100, 2), global_step)
                # log_writer.add_scalar("eval/ROUGE-2", round(rouge2 * 100, 2), global_step)
                # log_writer.add_scalar("eval/ROUGE-L", round(rougel * 100, 2), global_step)
                # log_writer.add_scalar("eval/BLEU-4", round(bleu4 * 100, 2), global_step)
                if best_rougel < rougel:
                    best_rougel = rougel
                    if paddle.distributed.get_rank() == 0:
                        # Need better way to get inner model of DataParallel
                        model_path =model_name+'_'+config['model_save_dir']
                        # print(model_path)
                        model_to_save = model._layers if isinstance(
                            model, paddle.DataParallel) else model
                        # print('aa')

                        model_to_save.save_pretrained(model_path)
                        # print('bb')
                        tokenizer.save_pretrained(model_path)

            if step == train_len // config['batch_size'] - 1:
                break








