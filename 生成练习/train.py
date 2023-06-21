import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from deal_data import *
import numpy as np
# from textcnn import Model as textcnnMmodel
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


class train_model:

    def __init__(self):
        cur_dir = os.path.dirname(__file__)
        self.loger = Logger(level='info')

        #---1 加载数据------
        deal_d=DealData()

        #----2 数据处理成embedding之前-----
        train_df, test_df, idx_train, idx_train_y, idx_test, idx_test_y=deal_d.load_data()

        print('训练集数据量：',len(train_df))
        print(len(idx_train),len(idx_train[0]))
        print(len(idx_train_y),len(idx_train_y[0]))


        give_emb=False

        '''参数'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # embed=train_vector.shape[2]
        # num_classes=len(set(train_lable))
        batch_size = train_config['batch_size']


        dev_pred_path=os.path.join(cur_dir,'model/dev_pred.xlsx')
        text_pred_path=os.path.join(cur_dir,'model/text_pred.xlsx')


        '''batch 数据'''
        train_loader=self.load_batch(idx_train,idx_train_y,batch_size,give_emb=give_emb)
        test_loader=self.load_batch(idx_test,idx_test_y,batch_size,give_emb=give_emb)
        # dev_loader=self.load_batch(dev_vector,dev_lable,batch_size,give_emb=give_emb)


        '''加载模型'''
        #---------3 transformer------------
        # y=Transformer().forward(enc_input, dec_input)

        self.model_path = os.path.join(cur_dir, transformer_config['transformer_model_path'])
        model =Transformer()
        # enc_inputs: [batch_size, seq_len]




        '''模型训练 验证集调参'''
        optimizer = torch.optim.AdamW(model.parameters(), train_config['lr'])
        loss = nn.CrossEntropyLoss()# 使用ignore_index参数：可以忽略某一项y的损失，一般用于labelPAD的情况。，还有一个weihgt参数，可以给label加权重



        self.train_data(train_config['epochs'],model, train_loader, device, optimizer, dev_loader,loss)

        '''用训练好的模型预测'''
        self.loger.logger.info('-------------预测验证集----------')
        self.pred_save(model, dev_loader, device, dev_data, dev_pred_path)
        self.loger.logger.info('-------------预测测试集----------')
        self.pred_save(model, test_loader, device, test_data, text_pred_path)


    '''保存模型在验证集上的结果'''
    def pred_save(self,model,data_loder,device,data_source,save_path):
        data_pred=self.model_pred( model, data_loder, device)
        data_df=pd.DataFrame(data_source).T

        data_df[1]=data_df[1].map(lambda x:label_id[x] )
        data_df['pred']=[label_id[i] for i in data_pred ]

        self.loger.logger.info(classification_report(data_df[1].to_list(), data_df['pred'].to_list()))
        print(classification_report(data_df[1].to_list(), data_df['pred'].to_list()))
        data_df.to_excel(save_path,index=False)


    '''训练数据 保存最优模型'''
    def train_data(self,epochs,model,train_loader,device,optimizer,dev_loader,loss):

        model = model.to(device)
        print("training on  ", device)
        self.loger.logger.info("training start  ")

        best_acc = 0.0
        train_loss=[]
        train_acc=[]
        testacc_list=[]
        for epoch in range(epochs):
            train_loss_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

            model.train()
            for X, Y in train_loader:


                X = X.to(device)
                y = Y.to(device)
                # t1=time.time()
                # y_hat = model(X,y)  #batch*numclass
                y_hat = model.forward(X, y) # [batch_size, tgt_len, tgt_vocab_size]

                y_hat=y_hat.view(-1,y_hat.size()[-1])#[batch_size*tgt_len, tgt_vocab_size]
                y=y.view(-1)#真实值平铺 batch_size*tgt_len

                # print('模型运行时间:',time.time()-t1)
                l = loss(y_hat, y)  # 交叉熵输入要求yhat：batch*numclass; y:batch*1

                optimizer.zero_grad()  # 梯度清空
                l.backward()  # 损失求导
                optimizer.step()  # 更新参数

                train_loss_sum += l.cpu().item()
                # 模型评估指标计算
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1



            dev_acc = self.dev_evaluate(dev_loader, model,device)
            print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, dev_acc, time.time() - start))
            self.loger.logger.info('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  % (epoch + 1, train_loss_sum / batch_count, train_acc_sum / n, dev_acc, time.time() - start))

            train_loss.append(train_loss_sum / batch_count)
            train_acc.append(train_acc_sum / n)
            devacc_list.append(dev_acc)

            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(model.state_dict(), self.model_path)  # 保存最好的模型参数
                # torch.save(model, self.model_path)  # 保存最好的模型
            self.loger.logger.info("current testacc is {:.4f},best acc is {:.4f}".format(dev_acc, best_acc))
            print("current testacc is {:.4f},best acc is {:.4f}".format(dev_acc, best_acc))


        self.train_plt(np.arange(1,epochs+1,1),train_loss,train_acc,devacc_list)

    def bleu_value(self):
        reference = [
            'this is a dog'.split(),
            'it is dog'.split(),
            'dog it is'.split(),
            'a dog, it is'.split()
        ]
        candidate = 'it is dog'.split()
        print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))

        candidate = 'it is a dog'.split()
        print('BLEU score -> {}'.format(sentence_bleu(reference, candidate)))
    def rouge_value(self):
        rouge = Rouge()

        rouge_scores = rouge.get_scores(" ".join(jieba.cut(self.y_pred))," ".join(jieba.cut(self.y_true)))#"Installing collected packages", "Installing "
        # print('rouge_scores:', rouge_scores)
        rouge_f=[rouge_scores[0][k]['f'] for k in rouge_scores[0]]
        score=0.2*rouge_f[0]+0.3*rouge_f[1]+0.5*rouge_f[2]
        # rl_p = rouge_scores[0]['rouge-l']['p']
        # print("score", score)
        return score

    '''可视化训练结果'''
    def train_plt(self,epoch,train_loss,train_acc,dev_acc):
        plt.figure()
        plt.plot(epoch,train_loss, 'r-', label='train_loss')
        plt.plot(epoch,train_acc, 'b-.', label='train_acc')
        plt.plot(epoch,dev_acc, 'g-.', label='dev_acc')

        plt.legend()
        plt.show()

    '''模型预测数据'''
    def model_pred(self, model, data_loader,device):

        # 加载模型
        m_state_dict = torch.load(self.model_path)
        model = model.to(device)
        model.load_state_dict(m_state_dict)

        # 预测数据
        pred_list = []
        # target_list = []
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            # target = target.long().to(device)
            output = model(data)  # torch.Tensor([[1.1537,  0.2496]])
            try:
                output = torch.argmax(output, 1).numpy().tolist()  # 选最大值对应的索引 0，转为list
            except TypeError:
                output = torch.argmax(output, 1).cpu().numpy().tolist()
            pred_list.extend(output)
        return pred_list


    '''验证集评估'''
    def dev_evaluate(self, data_iter, net, device):
        acc_sum, n = 0.0, 0
        with torch.no_grad():
            for X, Y in data_iter:
                # if isinstance(net, torch.nn.Module):
                net.eval()  # 评估模式, 会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == Y.to(device)).float().sum().cpu().item()
                net.train()  # 改回训练模式
                # else:
                    # acc_sum += (net(X).argmax(dim=1) == Y).float().sum().item()
                n += Y.shape[0]
        return acc_sum / n

    '''batch数据集'''
    def load_batch(self,x, y, batchSize,give_emb=True):
        '''
        give_emb==True:用户自定义emb,则X=torch.FloatTensor(x)
        give_emb==False:nn生成emb,则X=torch.LongTensor(x)
        '''
        X=torch.FloatTensor(x)
        if not give_emb:
            X = torch.LongTensor(x)
        data_set = TensorDataset(X,
                                  torch.LongTensor(y))
        data_loader = DataLoader(dataset=data_set,
                                  batch_size=batchSize,
                                  shuffle=False)
        return data_loader



train_model()