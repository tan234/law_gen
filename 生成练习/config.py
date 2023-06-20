import os

cur_path = os.path.dirname(__file__)

# label_id={
# 0:'finance',
# 1:'realty',
# 2:'stocks',
# 3:'education',
# 4:'science',
# 5:'society',
# 6:'politics',
# 7:'sports',
# 8:'game',
# 9:'entertainment'
# }

# word2vec_config={
#     'embed_size':128,
#     'w2v_path' :'rel_file/baike_26g_news_13g_novel_229g.bin'
#
# }

# textcnn_config={
#     'model_path':'result/textcnn_model.pth'
# }

data_config={
    'num_classes':10,
    'vocab_path':os.path.join(cur_path,"model/vocab.txt"),
    'vocab_size': 56850,
    'seq_len':16,
    'stopword_path':os.path.join(cur_path,"model/stop_word.txt")
}

train_config={
    'lr':2e-3,
    'batch_size':32,
     'epochs':2,
    # 'seq_len':8,
}

# bilstm_config={
#     'hidden_dim':128,
#     'num_layers':2,
#     'dropout':.1,
#     'model_path':'result/bilstm_model.pth'
#
# }
#
#
# bilstm_attention_config={
#
#     'model_path': 'result/bilstm_attention_model.pth',
#     'head_num':8,
#     'atttion_wdim':64
# }


transformer_config={
        # encoder
        'emb_size':512,#positional encoding 维度,这两个要维度相加，应该是一样的维度
        'd_ff' : 2048 ,# FeedForward dimension
        'd_k':64,# 变成K，Q，V矩阵的维度,K和Q一定是一样的，因为要K乘Q的转置,V不一定,这里我们认为是一样的
        'd_v': 64,  # dimension of K(=Q), V
    'n_layers': 2,# encoder和decoder各有多少层
    'n_heads':  8,# multi-head attention有几个头
    'transformer_model_path':'result/transformer_model.pth',
# decoder
'tgt_emb':512,
    'tgt_len':8,
    'tgt_vocab_size':56850

}

# bert_config={
#     'bert_dir': 'chinese_L-12_H-768_A-12',
#
# }


















