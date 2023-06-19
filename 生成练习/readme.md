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
xemb+posemb->encoder(attention[k,q,v计算，k,v,q来自x*W]-addnorm-ffn-addnorm)->decoder
encoder层:
- X:batch,seq
  (1) word emb :随机生成
  (2) position emb :cos,sin生成,
- (3) word emb+pos emb +dropout:这里为什么还加个dropout


mask
(1) pad mask:输入seq_k,seq_q;输出(batch,seq_k,seq_q)，因为这个要和 (k*q^T/dk)相加,所以形状一样，只对k进行了pad mask


torch	1.10.1	