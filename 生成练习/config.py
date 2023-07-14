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

Transformer_config={
    'emb_size':256,#positional encoding 维度,这两个要维度相加，应该是一样的维度
    'd_ff' : 2048 ,# FeedForward dimension
    'd_k':64,# 变成K，Q，V矩阵的维度,K和Q一定是一样的，因为要K乘Q的转置,V不一定,这里我们认为是一样的
    'd_v': 64,  # dimension of K(=Q), V
    'n_heads':  4,# multi-head attention有几个头
    'model_save_dir':'model/transformer_model.pth',
    'enc_layers':1,
    'dec_layers':1,
    'tgt_emb':256,
    # 'tgt_vocab_size':56850

}

Pegasus_config={
    'max_source_length':128,# 文本的最大长度
    'max_target_length':64,# 摘要的最大长度
    'min_target_length':0,# 摘要的最小长度
    'num_epochs':1,
    'model_save_dir' :'checkpoints/',  # 训练模型保存路径
    'batch_size':  10,
    'warmup_proportion': 0.02,  # 学习率预热比例
    'learning_rate': 5e-5,
    'adam_epsilon': 1e-6 , # AdamW优化器参数epsilon
    'weight_decay':  0.01,  # AdamW优化器参数weight_decay
    'log_steps': 10,  # 训练中，每个log_steps打印一次日志
    'eval_steps': 100,  # 训练中，每隔eval_steps进行一次模型评估

}


GPT2_config={
    'max_source_length':128,# 文本的最大长度
    'max_target_length':64,# 摘要的最大长度
    'min_target_length':0,# 摘要的最小长度
    'num_epochs':1,
    'model_save_path' :'/model/gpt.pkl',  # 训练模型保存路径
    'batch_size':  10,
    'warmup_proportion': 0.02,  # 学习率预热比例
    'learning_rate': 0.001,
    'adam_epsilon': 1e-6 , # AdamW优化器参数epsilon
    'weight_decay':  0.01,  # AdamW优化器参数weight_decay
    'log_steps': 10,  # 训练中，每个log_steps打印一次日志
    'eval_steps': 100,  # 训练中，每隔eval_steps进行一次模型评估
    'vocab_size':1000,
    # 推断阶段用
    'max_len':  50,
    'repetition_penalty':  1.0,    # 惩罚比例 ，如果输出重复的单词太多，适当调高这个比例
    'temperature':  1,
    'top_k':5

}

















