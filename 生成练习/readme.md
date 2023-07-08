项目：法研杯-摘要生成任务

算法1： textrank:0.3937907069882525



# 算法2:transformer

模型评估：

(1) 采用ROUGE(Recall-Oriented Understudy for Gisting Evaluation)评价。
总分计算方式为：0.2f-score(R1)+0.3f-score(R2)+0.5*f-score(RL)。
目前GigaWord数据集最好的ROUGE-1、ROUGE-2、ROUGE-L分别是39.51、20.42、36.69，
来自论文 ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training 提出的ProphetNet，
使用了完整的GigaWord数据。更多的方法可见https://paperswithcode.com/sota/text-summarization-on-gigaword ，可以做更多的了解研究。

(2) BLEU 是一种文本生成评测指标，考虑生成摘要与参考摘要n元组共同出现的程度。一般计算所有元祖的bleu得分之和求取平均数


## 1 transformer 模型 结构
总输入：enc_input,dec_input(dec 的输入需要拼接上'S'和'E',因为预测的时候，dec_inp第一步的输入是S,然后如果后面碰到了E就停止预测)

一 encoder block
1 输入：enc_input[batch,enc_seq]
2 word_emb:得到[batch,enc_seq,enc_emb],现在是随机生成；后续可以用大模型训练好的词向量代替来做初始化
3 position_emb:sin,cos的方式，输入为[seq，emb],因为都是每个batch对应的位置值固定，所以不需要用batch;后续可以改为相对位置编码
4 enc_outputs=word_emb+pos_emb[batch,enc_seq,enc_emb]
5 dropout(enc_outputs)
6 enc_pad_mask:输入[enc_input,enc_input],PAD位置为True
7 self_attention:输入[enc_outputs(上面的5),enc_pad_mask]，矩阵计算：softmax((q,k)/dk+M)*V;可以考虑数学方法加快矩阵计算
8 add_norm
9 FFN:f(XW+b)W+b;激活函数改成了GELU
10 add_norm
11 输出：[batch_size, enc_seq, emb_size]

二 decoder block
1 输入：decoder_input[batch_size, dec_len]
2 word_emb:变成[batch_size, dec_len,dec_emb_size]
3 pos_emb:同encoder
4 decoder=dropout(word_emb+pos_emb)
5 decoder attention mask:
decoder_attention_mask:pad mask + dec_seq_mask:(mask上三角，看不到未来的decoder)
6 enc_dec_att_mask:
decoder-encoder-mask:pad mask，mask的是encoder的pad
7 decoder attention层：输入 (decoder[batch_size, dec_len,dec_emb_size]，decoder_mask[dec_input,dec_inp])
8 addnorm
9 decoder encoder attention层：输入(decoder-ouput,encoder-output,enc_dec_mask)
10 addnorm
11 FFN+addnorm
12 输出：[batch_size, dec_len, dec_emb_size]

三 全连接
decoder 输出的结果，经过全连接层得到：[batch_size, tgt_len, tgt_vocab_size]
也就是每一条数据，decoder每一步的预测值

四 共同部分

1 pad_mask：
- encoder PAD在encoder的self-attention 层用
- decoder PAD 在decoder层用;decoder 中当前位置之后的位置，设为 1（mask）,
- en-de mask 在encoder-decoderattention层使用;已经对decoder 自己进行了mask，这里面还差一个encoder decoder mask

2 pos_emb:encoder和decoder都是用的cos

3 attention:attention 三个矩阵的计算+add_norm

4 FFN+addnorm


## 2 模型训练

(1) 输入:(其中seq_x，和seq_y不一样，两者的vocab可以一样，也可以不一样)
    X_input():[batch,seq_x]
    Y_input():[batch,seq_y]

(2) 输出：[batch_size, seq_y, tgt_vocab_size] softmax后取最大的一个概率得到 [batch,seq_y]

(3) 损失：交叉熵loss( y_input)

(4) 评估：rouge

## 3 模型预测
(1) 预测时，输出seq的长度=max_len，是给定的，需要小于训练时的最大长度

(2) 训练时候输入是enc_input,dec_input
预测时先输入enc_input ,到encoder，再输入到decoder,dec_input一步步预测下一步。每次预测结果要拼接到下一步的dec_input上进行预测


## 4 问题及改进：
  (1) 文本生成的常见问题：重复单词，重复句子，OOV，不成句；decoder方法会导致可读性，重复性等问题，文本长的时候出现问题，训练时候输入的dec_input是真实的，
      但是预测的时候不是，这存在这训练与预测不一致问题

  (2) 相对位置编码去替换一下sin位置编码

  (3) 现在是从头训练，尝试用预训练好的模型参数和vocab去初始化参数，然后看预训练模型自己训练可以调整的部分包括什么

  (4) dropout层

  (5) 正常的调参

  (6) 早停，观察损失和epoch，曲线

  (7) 观察attention 矩阵能看出来什么？

  (8) 模型评估方法的选择及原因（相似度可以考虑？）,rouge多少算好,blue值多少算好?

  (9) 整个过程的参数量?,因为训练出来的模型文件很大，计算一下这个的关系

  (10) 预测的结果不成句，是不是可以在预测的最后一层加上一个语言模型？2-gram这样

  (11) 预测序列时可以考虑增加一个k的随机进行，有个beam search，是在输出做了一个路径最优处理(https://zhuanlan.zhihu.com/p/114669778)

  (12) 可以尝试如何利用深度无监督模型去做生成式摘要任务。 例如：以自编码器为主体架构，对其进行不同程度的改造，从压缩或者生成两个角度去无监督生成摘要文本，
       同时为了提升效果，也会利用GPT,XLNET等预训练语言模型做finetune。

  (13) 词向量可以选择tfidf,w2v,Glove等来生成，或者说来代表一个句子

  (14) 梯度归一化：其实就是计算出来梯度之后，要除以minibatch的数量

  (15) 调度学习率：如果在验证集上性能不再增加就让学习率除以2或者5，然后继续，学习率会一直变得很小，到最后就可以停止训练了。




## 5 参考：
Paddle : https://aistudio.baidu.com/aistudio/projectdetail/4876808?channelType=0&channel=0

https://www.bilibili.com/video/BV1mk4y1q7eK/?p=3

https://wmathor.com/index.php/archives/1455/

https://work.datafountain.cn/forum?id=130&type=2&source=1

