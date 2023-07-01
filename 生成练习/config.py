import os

cur_path = os.path.dirname(__file__)

data_config={
    'vocab_path':os.path.join(cur_path, "model/vocab.txt"),
    'idx2word_dict':os.path.join(cur_path, "model/idx2word.txt"),
    'enc_vocab_size': 5152,
    'dec_vocab_size': 5152,
    'enc_len':1200,#500
    'stopword_path':os.path.join(cur_path,"model/stop_word.txt"),
    'dec_len':200#48
}

train_config={
    'lr':2e-3,
    'batch_size':8,
     'epochs':2,
}

transformer_config={
    'emb_size':256,#positional encoding 维度,这两个要维度相加，应该是一样的维度
    'd_ff' : 2048 ,# FeedForward dimension
    'd_k':64,# 变成K，Q，V矩阵的维度,K和Q一定是一样的，因为要K乘Q的转置,V不一定,这里我们认为是一样的
    'd_v': 64,  # dimension of K(=Q), V
    'n_heads':  4,# multi-head attention有几个头
    'transformer_model_path':'model/transformer_model.pth',
    'enc_layers':1,
    'dec_layers':1,
    'tgt_emb':256,
    # 'tgt_vocab_size':56850

}

