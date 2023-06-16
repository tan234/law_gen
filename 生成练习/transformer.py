
import math
import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# from config import *
#超参

heads = 6
max_decoding_len = 21
max_pos_len = 5000
learning_rate=1e-3
epochs = 10
fc_dim = 512
dropout_rate=0.1
attention_dropout_rate=0.1
encoder_layers = 1
decoder_layers = 1

config={
    'vocab_size':64,
    'emb_size':128,
    'seq_len':16,
    'n_layers':6,
    'n_heads':16,
    'd_k':64,
    'd_v':64,
    'd_ff':128,
    'tgt_vocab_size':'10',
    'tgt_len':64,
    'tgt_emb':128,
    

}
class Encoder(nn.Module):
    '''
    输入:X,embedding,position-encoder
    multi-head-self-attention,add-norm,feed-forward,add-norm
    '''
    def __init__(self):
        super(Encoder, self).__init__()

        self.word_emb = nn.Embedding(config['vocab_size'], config['emb_size'])

        self.pos_emb = PositionalEncoding(config['seq_len'],config['emb_size'])
        # 计算位置向量
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(config['n_layers'])])
        # 使用 nn.ModuleList() 里面的参数是列表，列表里面存了 n_layers 个 Encoder Layer
        # 由于我们控制好了 Encoder Layer 的输入和输出维度相同，所以可以直接用个 for 循环以嵌套的方式，
        # 将上一次 Encoder Layer 的输出作为下一次 Encoder Layer 的输入
        # 将6个EncoderLayer组成一个module


    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, seq_len]
        '''

        # embedding
        enc_outputs = self.word_emb(enc_inputs)#enc_outputs [batch_size, seq_len, emb]
        # 添加位置编码
        enc_outputs = self.pos_emb.forward(enc_outputs.transpose(0, 1)).transpose(0, 1)#  enc_outputs [batch_size, seq_len, emb]

        # 计算得到encoder-attention的pad martix,因为是maskpad 所以对没有embedding的数据进行操作？
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)# enc_self_attn: [batch_size, seq_len, seq_len]

        enc_self_attns = []# 创建一个列表，保存接下来要返回的字-字attention的值，不参与任何计算，供可视化用

        # 循环每个encoder
        for layer in self.layers:

            # enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # 再传进来就不用positional decoding
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)# enc_outputs: [batch_size, seq_len, emb]
            # 记录下每一次的attention
            enc_self_attns.append(enc_self_attn)

        # return enc_outputs, enc_self_attns
        # 只用encoder分类时，接入一个linear变成batch,num
        return nn.Linear(config['emb_size']*config['seq_len'],config['num_classes'])(enc_outputs.view(config['batch_size'], -1))

class PositionalEncoding(nn.Module):
    '''
    transformer的位置向量，是sin cos变换；形状=X
    output=position+X
    self.pe 已加入buffer里，
    pe形状为:(max_len,1,emb_size)
    '''
    def __init__(self, seq_len,emb_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, emb_size)
        # pe [seq_len, emb_size]

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # position  [max_len，1]

        _2i = torch.arange(0, emb_size, step=2).float()
        # _2i [emb_size / 2]

        # 两个相乘的维度为[max_len,emb_size/2]
        pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / emb_size)))
        pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / emb_size)))


        pe = pe.unsqueeze(1)
        # 维度变成(max_len,1,emb_size)，
        self.register_buffer('pe', pe)
        # 放入buffer中，参数不会训练
        # 因为无论是encoder还是decoder，他每一个字的维度都是d_model
        # 同时他们的位置编码原理是一样的
        # 所以一个sequence中所需要加上的positional encoding是一样的。
        # 所以只需要存一个pe就可以了
        # 同时pe是固定的参数，不需要训练
        # 后续代码中，如果要使用位置编码，只需要self.pe即可，因为pe已经注册在buffer里面了

    def forward(self, x):
        '''
        x: [seq_len, batch_size, emb_size]
        '''
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)




def get_attn_pad_mask(seq_q, seq_k):

    '''
    seq_q=seq_k：[batch_size, seq_len]
    对seq_len中的pad进行mask,PAD对应的vocab id为0
    # 对于每一个batch_size对应的一行，都扩充为len_q行
    # [batch_size, len_q, len_k]
    '''

    # 由于在 Encoder 和 Decoder 中都需要进行 mask 操作，
    # 因此就无法确定这个函数的参数中 seq_len 的值，
    # 如果是在 Encoder 中调用的，seq_len 就等于 src_len
    # 如果是在 Decoder 中调用的，seq_len 就有可能等于 src_len，
    # 也有可能等于 tgt_len（因为 Decoder 有两个attention模块，两次 mask）
    # src_len 是在encoder-decoder中的mask
    # tgt_len是decoder中的mask
    # 对于seq_q中的每一个元素，它都会和seq_k中的每一个元素有着一个相关联系数，这个系数组成一个矩阵：
    # 但是因为pad的存在，pad的这些地方是不参与我们attention的计算的
    # 那么就是我们这里要返回的东西就是辅助得到哪些位是需要pad的


    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # pad的位置标记上True
    # seq_q:[[1,2,3,4,0],[1,2,4,5,0]] ->pad_attn_mask [[F,F,F,F,T],[F,F,F,F,T]]-> [batch_size, 1, len_k]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # 多头注意力机制
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    # 提取特征

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, seq_len, emb_size]
        enc_self_attn_mask: [batch_size, seq_len, seq_len]
        '''

        # -----attention+addnorm层------------
        # encoder的输入是三个input一样的；decoder的attention输入的是decoder input ,encoder output, encoder output
        # attn: [batch_size, n_heads, src_len, src_len] 每一个头一个注意力矩阵
        # enc_inputs to same Q,K,V
        # enc_inputs乘以WQ，WK，WV生成QKV矩阵
        enc_outputs, attn = self.enc_self_attn.forward(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_outputs: [batch_size, seq_len, emb_size],


        # ---------FFN+addnorm 层--------------
        # 输入和输出的维度是一样的
        # 将上述组件拼起来，就是一个完整的 Encoder Layer
        enc_outputs = self.pos_ffn.forward(enc_outputs)# enc_outputs: [batch_size, seq_len, seq_len]

        return enc_outputs, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.n_heads=config['n_heads']
        self.emb_size=config['emb_size']
        self.d_k=config['d_k']
        self.d_v=config['d_v']


        self.W_Q = nn.Linear(self.emb_size, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.emb_size, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.emb_size, self.d_v * self.n_heads, bias=False)
        # 三个矩阵，分别对输入进行三次线性变化

        self.fc = nn.Linear(self.n_heads * self.d_v, self.emb_size, bias=False)
        # 变换维度

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        因为要QKT,所以要求 d_k=d_q
        input_Q: [batch_size, len_q, emb_size]
        input_K: [batch_size, len_k, emb_size]
        input_V: [batch_size, len_v(=len_k), emb_size]
        attn_mask: [batch_size, seq_len, seq_len]
        '''

        residual, batch_size = input_Q, input_Q.size(0)
        # 生成Q，K，V矩阵:[batch_size,n_heads, seq_len_q,d_k ]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)


        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)#[batch_size, n_heads, seq_len, seq_len]

        context, attn = ScaledDotProductAttention().forward(Q, K, V, attn_mask)#计算softmax(qk)V
        # context: [batch_size, n_heads, len_q, d_v],
        # attn: [batch_size, n_heads, len_q, len_k]

        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # context: [batch_size, len_q, n_heads * d_v]

        output = self.fc(context)
        # [batch_size, len_q, emb_size]

        # Add & Norm
        return nn.LayerNorm(self.emb_size)(output + residual), attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        '''
        这里要做的是，通过 Q 和 K 计算出 scores，然后将 scores 和 V 相乘，得到每个单词的 context vector
        第一步是将 Q 和 K 的转置相乘，相乘之后得到的 scores 还不能立刻进行 softmax，
        需要和 attn_mask 相加，把一些需要屏蔽的信息屏蔽掉，
        attn_mask 是一个仅由 True 和 False 组成的 tensor，并且一定会保证 attn_mask 和 scores 的维度四个值相同（不然无法做对应位置相加）
        mask 完了之后，就可以对 scores 进行 softmax 了。然后再与 V 相乘，得到 context

        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(config['d_k']) # scores : [batch_size, n_heads, len_q, len_k]

        scores.masked_fill_(attn_mask, -1e9)
        # attn_mask所有为True的部分（即有pad的部分），scores填充为负无穷，也就是这个位置的值对于softmax没有影响
        # masked_fill_(mask, value)用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。

        attn = nn.Softmax(dim=-1)(scores)

        context = torch.matmul(attn, V)
        # [batch_size, n_heads, len_q, d_v]

        return context, attn



class PoswiseFeedForwardNet(nn.Module):

    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(config['emb_size'], config['d_ff'], bias=False),
            nn.ReLU(),
            nn.Linear(config['d_ff'], config['emb_size'], bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, emb_size]
        这段代码非常简单，就是做两次线性变换，残差连接后再跟一个 Layer Norm
        '''

        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(config['emb_size'])(output + residual) # [batch_size, seq_len, emb_size]




class Decoder(nn.Module):
    '''
    mask-multi-head-self-attention,add-norm,multi-head-attention,add-norm,feed-forward,add-norm
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(config['tgt_vocab_size'], config['tgt_emb'])
        self.pos_emb = PositionalEncoding(config['tgt_len'],config['tgt_emb'])
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(config['n_layers'])])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, seq_len]
        enc_outputs: [batsh_size, seq_len, emb] 经过六次encoder之后得到的东西
        '''

        dec_outputs = self.tgt_emb(dec_inputs)# [batch_size, tgt_len, tgt_emb]

        dec_outputs = self.pos_emb.forward(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()# [batch_size, tgt_len, tgt_emb]


        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()# [batch_size, tgt_len, tgt_len]

        # 当前时刻我是看不到未来时刻的东西的，要把之后的部门mask掉
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()    # [batch_size, tgt_len, tgt_len]

        # 结果是所有需要被mask掉位置为True，不需要被mask掉的为False
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda()# [batch_size, tgt_len, tgt_len]

        # 在decoder的第二个attention里面使用
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)# [batc_size, tgt_len, seq_len]



        dec_self_attns, dec_enc_attns = [], []
        # decoder的两个attention模块

        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]

            dec_outputs, dec_self_attn, dec_enc_attn = \
                layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)

            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


def get_attn_subsequence_mask(seq):
    '''
    作用：屏蔽未来时刻单词的信息。
    seq: [batch_size, tgt_len]

    s=torch.Tensor([[1,1,1],[3,5,1]])
    get_attn_subsequence_mask(s)
    tensor([[[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]],
            [[0, 1, 1],
             [0, 0, 1],
             [0, 0, 0]]], dtype=torch.uint8)
    '''

    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]    # [batch_size, tgt_len, tgt_len]

    # np.triu(a, k)是取矩阵a的上三角数据，但这个三角的斜线位置由k的值确定。
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # [batch_size, tgt_len, tgt_len]

    subsequence_mask = torch.from_numpy(subsequence_mask).byte()# 转化成byte类型的tensor

    return subsequence_mask



class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, seq_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, seq_len]
        # 在 Decoder Layer 中会调用两次 MultiHeadAttention，第一次是计算 Decoder Input 的 self-attention，得到输出 dec_outputs。
        # 然后将 dec_outputs 作为生成 Q 的元素，enc_outputs 作为生成 K 和 V 的元素，再调用一次
        '''

        # 先是decoder的self-attention
        dec_outputs, dec_self_attn = self.dec_self_attn.forward(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]

        # 再是encoder-decoder attention部分
        dec_outputs, dec_enc_attn = self.dec_enc_attn.forward(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]

        # 最后 FFN
        dec_outputs = self.pos_ffn.forward(dec_outputs)  # [batch_size, tgt_len, d_model]


        return dec_outputs, dec_self_attn, dec_enc_attn




class Transformer(nn.Module):
    '''
    transformer:encoder+decoder+linear
    dec_logits view了之后的维度是 [batch_size * tgt_len, tgt_vocab_size]，可以理解为，
    一个长句子，这个句子有 batch_size*tgt_len 个单词.
    每个单词用 tgt_vocab_size 维表示，表示这个单词为目标语言各个单词的概率，取概率最大者为这个单词的翻译
    Transformer 主要就是调用 Encoder 和 Decoder。最后返回
    '''

    def __init__(self):
        super(Transformer, self).__init__()

        self.encoder = Encoder().cuda()
        self.decoder = Decoder().cuda()
        self.projection = nn.Linear(config['emb_size'], config['tgt_vocab_size'], bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs维度：[batch_size, seq_len]
        dec_inputs: [batch_size, tgt_len]
        '''

        enc_outputs, enc_self_attns = self.encoder(enc_inputs) # enc_outputs: [batch_size, src_len, emb_size]
        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 注意力矩阵，对encoder和decoder，每一层，每一句话，每一个头，每两个字之间都有一个权重系数，
        # 这些权重系数组成了注意力矩阵
        # 之后的dec_self_attns同理，当然decoder还有一个decoder-encoder的注意力矩阵

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # dec_outpus: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]

        dec_logits = self.projection(dec_outputs)# dec_logits: [batch_size, tgt_len, tgt_vocab_size]

        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

# X=torch.tensor([random.randint(0,1000) for i in range(32*8)], dtype=torch.long).reshape(32,8)
# print(X)
# Y=Encoder().forward(X)
# print(Y)
# print(Y.size())
