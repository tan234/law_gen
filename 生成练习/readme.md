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
- 
torch	1.10.1

问题：
decoder训练和测试，输入的mask问题？