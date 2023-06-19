

import torch
import torch.nn as nn
# LayerNorm
a = torch.tensor([[2,4],[3,9],[2,4]]).float()
b=nn.LayerNorm([2])(a)
# a.size()=(3,2);参数2 表示维度，的到 3 个均值方差mean（2+4）/var(2,4)
# 参数为两个维度（3，2），得到一个方差

# https://blog.csdn.net/weixin_39228381/article/details/107939602

print(a)
print(b)
print(a.size(),b.size())
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
