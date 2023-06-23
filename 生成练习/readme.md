法研杯-摘要生成任务

1 textrank:
0.3937907069882525


模型评估：
采用ROUGE(Recall-Oriented Understudy for Gisting Evaluation)评价。
ROUGE指标将模型生成的摘要与参考摘要进行比较,
其中ROUGE-1衡量unigram匹配情况，
ROUGE-2衡量bigram匹配，
ROUGE-L记录最长的公共子序列。
三者都只采用f-score，
总分计算方式为：0.2f-score(R1)+0.3f-score(R2)+0.5*f-score(RL)。

2 transformer
xemb+posemb->encoder{attention[k,q,v计算，k,v,q来自x*W]-addnorm-ffn[f(xw)w]-addnorm}->decoder{attention->addnorm->ffn->addnorm}
encoder层:
- X:batch,seq
  (1) word emb :随机生成
  (2) position emb :cos,sin生成,
- (3) word emb+pos emb +dropout:这里为什么还加个dropout


mask
(1) pad mask:
   1 encoder 自己的PAD mask
   2 decoder 自己的PAD mask
   3 在decoder 中的，encoder decoder attention 层，已经对decoder 自己进行了mask，这里面还差一个encoder decoder mask
(2) decoder mask:decoder 中当前位置之后的位置，设为 1（mask）,

- decoder层
- X:batch,seq
  (1) word emb :随机生成


encoder-decoder:

- 训练
输入:(其中seq_x，和seq_y不一样，但是vocab)
X_input():[batch,seq_x]
Y_input():[batch,seq_y]

输出：[batch_size, seq_y, tgt_vocab_size] softmax后取最大的一个概率得到 [batch,seq_y]

损失：交叉熵loss( y_input)

计算正确率：rouge

- 预测
预测时，输出seq的长度如何定？
https://work.datafountain.cn/forum?id=130&type=2&source=1
  
训练时候的输入时input output，预测时没有output怎么办，这里面的
output用的是上一步y的预测结果，所以需要调整模型？


问题，训练时，用了y的上一步结果了吗？
- 问题：
文本生成的问题，比如说重复单词，重复句子，OOV
- 需要sos，与eos吗？
torch	1.10.1

评估：BLEU bleu是一种文本生成评测指标，考虑生成摘要与参考摘要n元组共同出现的程度。一般计算所有元祖的bleu得分之和求取平均数，下面为计算公式：

改进：
1 相对位置编码
https://work.datafountain.cn/forum?id=124&type=2&source=1
2 gelu激活
3 dropout层
4 初始化的参数用的bert?
5 早停，观察损失和epoch，曲线
6 观察attention 矩阵能看出来什么？
模型评估用的rouge是因为，我们观察了训练书记的最长公共子序列，比较符合

残差连接是为了，梯度小时和模型退化？

问题：
decoder训练和测试，输入的mask问题？