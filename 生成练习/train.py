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
import paddle
from paddlenlp.ops.optimizer import AdamWDL
import time
# from paddlenlp.utils.log import logger
from paddlenlp.metrics import BLEU
from tqdm import tqdm
import numpy as np
import os
from paddlenlp.transformers import GPTTokenizer
from evaluating import gpt2_evaluate,gpt2_out_choice
from config import *
import torch.nn as nn
import torch
from data import DealData,load_batch

from evaluating import pegasus_evaluate,dev_evaluate,rouge_value
# ------模型训练------
import time
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from config import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# from  bilstm import *
from log import *
# from bilstm_attention import *
# from word_emb import *
# from encoder import *
from transformer_model import Transformer
from rouge  import Rouge
from nltk.translate.bleu_score import sentence_bleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from data import TransformerData
import numpy as np
# from textcnn import Model as textcnnMmodel

#===================Transformer=============




def  transformer_train( idx_train, idx_train_y, idx_test, idx_test_y,model_name,config):
    # train_df, test_df, idx_train, idx_train_y, idx_test, idx_test_y=deal_d.load_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = train_config['batch_size']
    give_emb = False
    '''batch 数据'''
    train_loader = load_batch(idx_train, idx_train_y, batch_size, give_emb=give_emb)
    test_loader = load_batch(idx_test, idx_test_y, batch_size, give_emb=give_emb)
    # dev_loader=self.load_batch(dev_vector,dev_lable,batch_size,give_emb=give_emb)

    '''加载模型'''
    # ---------3 transformer------------
    # y=Transformer().forward(enc_input, dec_input)

    # model_path = config['transformer_model_path']
    model = Transformer()
    # enc_inputs: [batch_size, seq_len]

    '''模型训练 验证集调参'''
    optimizer = torch.optim.AdamW(model.parameters(), train_config['lr'])
    loss = nn.CrossEntropyLoss(ignore_index=0)  # 使用ignore_index参数：可以忽略某一项y的损失，一般用于labelPAD的情况。，还有一个weihgt参数，可以给label加权重

    trans_tra(train_config['epochs'], model, train_loader, device, optimizer, test_loader, loss,model_name,config)


def trans_tra(epochs,model,train_loader,device,optimizer,dev_loader,loss,model_name,config):

        model = model.to(device)
        print("training on  ", device)
        # self.loger.logger.info("training start  ")

        best_acc = 0.0
        train_loss=[]
        train_acc=[]
        train_rouge_list=[]
        devacc_list=[]
        for epoch in range(epochs):
            train_loss_sum, train_rouge, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

            model.train()
            for X, Y in train_loader:


                X = X.to(device)
                y = Y.to(device)
                # t1=time.time()
                n += y.shape[0]
                y_hat,_,_,_, = model.forward(X, y) # [batch_size, tgt_len, tgt_vocab_size]

                y_hat=y_hat.view(-1,y_hat.size()[-1])#[batch_size*tgt_len, tgt_vocab_size]
                y=y.view(-1)#真实值平铺 batch_size*tgt_len

                # print('模型运行时间:',time.time()-t1)
                l = loss(y_hat, y)  # 交叉熵输入要求yhat：batch*numclass; y:batch*1

                optimizer.zero_grad()  # 梯度清空
                l.backward()  # 损失求导
                optimizer.step()  # 更新参数

                # 模型评估指标计算
                y = ' '.join([str(i) for i in y.tolist()])
                y_hat = y_hat.argmax(dim=1)
                y_hat = ' '.join([str(i) for i in y_hat.tolist()])

                train_rouge+=rouge_value(y,y_hat)

                train_loss_sum += l.cpu().item()
                batch_count += 1
                # print(y)
                # print(y_hat)
                # print(train_rouge)

            dev_acc = dev_evaluate(dev_loader, model)
            # print(dev_acc)
            # kk
            print('epoch %d, loss %.4f, train_rouge %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_loss_sum / batch_count, train_rouge / n, dev_acc, time.time() - start))
            # self.loger.logger.info('epoch %d, loss %.4f, train_rouge %.3f, test acc %.3f, time %.1f sec'
            #       % (epoch + 1, train_loss_sum / batch_count, train_rouge / n, dev_acc, time.time() - start))

            train_loss.append(train_loss_sum / batch_count)
            train_rouge_list.append(train_rouge / n)
            devacc_list.append(dev_acc)
            model_path = model_name + '_' + config['model_save_dir']

            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), model_path)  # 保存最好的模型参数
                # torch.save(model, self.model_path)  # 保存最好的模型
            # self.loger.logger.info("current testacc is {:.4f},best acc is {:.4f}".format(dev_acc, best_acc))
            print("current testacc is {:.4f},best acc is {:.4f}".format(dev_acc, best_acc))


        # self.train_plt(np.arange(1,epochs+1,1),train_loss,train_acc,devacc_list)




#===================pegasus=============
def pegasus_train(tokenizer,model_name,config,train_dataset,dev_dataset,test_dataset):

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
    train(tokenizer,model, train_data_loader,optimizer,lr_scheduler,num_training_steps,num_epochs,dev_data_loader,len(dev_dataset),len(train_dataset),model_name,config)

    # 模型评估
    pegasus_evaluate(model, test_data_loader,len(test_dataset),config, tokenizer)

def train(tokenizer,model, train_data_loader,optimizer,lr_scheduler,num_training_steps,num_epochs,dev_data_loader,dev_len,train_len,model_name,config):
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
                rouge1, rouge2, rougel, bleu4 = pegasus_evaluate(model, dev_data_loader,dev_len,config, tokenizer)


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
#===================gp2=============
def gpt2_train(tokenizer,config,model,train_loder,dev_loader,train_len,dev_len):

    # name_dict = dict()
    adam = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=config['learning_rate'])
    for epoch in range(config['num_epochs']):
        best_rougel=0

        for i, data in enumerate(train_loder):

            content, label, lenght = data

            content=content.squeeze(axis=1)#shape=[10, 171]
            # print(content)
            out = model(content)#[10, 171, 50258]
            # print(out.shape)
            # kk
            # print(lenght)
            # out = out[:, lenght:, :]
            # loss_all=0
            out_list=[]
            for i in range(len(lenght)):
                out_per=out[i, lenght[i]:, :]
                # out_per=gpt2_out_choice(out_per,tokenizer)
                # print('out_per:',out_per)
                out_list.append(out_per)
                lenght[i]=lenght[i][:len(out_per)]
                # print(, label[i])# out[i, lenght[i]:, :]:shape=[20, 50258],label:shape=[1, 20],
                # loss_all += paddle.nn.functional.cross_entropy(out[i, lenght[i]:, :], label[i])
            # loss = nn.CrossEntropyLoss(
            #     ignore_index=0)  # 使用ignore_index参数：可以忽略某一项y的损失，一般用于labelPAD的情况。，还有一个weihgt参数，可以给label加权重
            # print(out_list,label)

            out_list=paddle.to_tensor(out_list).squeeze(1)
            label=label.squeeze(1)
            # print(type(out_list),out_list.shape)
            # print('------')
            # print(type(label),label.shape)

            # print(out_list)
            # print(label)

            # kk
            # label=torch.tensor(label).squeeze()
            # print(loss(out_list,label ))

            loss = paddle.nn.functional.cross_entropy(out_list, label)
            print(f"epoch:{epoch} step:{i} loss:{loss.item()}")
            # kk
            loss.backward()
            adam.step()
            adam.clear_grad()

            # 评估
            rouge1, rouge2, rougel = gpt2_evaluate(model,tokenizer, dev_loader,config,dev_len)

                # print(rouge1, rouge2, rougel, bleu4)
                # logger.info("eval done total : %s s" % (time.time() - tic_eval))
                # log_writer.add_scalar("eval/ROUGE-1", round(rouge1 * 100, 2), global_step)
                # log_writer.add_scalar("eval/ROUGE-2", round(rouge2 * 100, 2), global_step)
                # log_writer.add_scalar("eval/ROUGE-L", round(rougel * 100, 2), global_step)
                # log_writer.add_scalar("eval/BLEU-4", round(bleu4 * 100, 2), global_step)
            if best_rougel < rougel:
                best_rougel = rougel
                paddle.save(model.state_dict(), config['model_save_path'])

                # if paddle.distributed.get_rank() == 0:
                    # Need better way to get inner model of DataParallel
                    # model_path = model_name + '_' + config['model_save_dir']
                    # print(model_path)
                    # model_to_save = model._layers if isinstance(
                    #     model, paddle.DataParallel) else model
                    # print('aa')

                    # model_to_save.save_pretrained(model_path)
                    # print('bb')
                    # tokenizer.save_pretrained(model_path)


            # if i % 500 == 0:
                # paddle.save(model.state_dict(), f'model/model_{epoch}_{i}.pkl')












