
import tensorflow.compat.v1 as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
tf.disable_v2_behavior()
tf.disable_eager_execution()


#占位符

tf_texts = tf.placeholder(tf.int32, [None,None])
tf_summaries = tf.placeholder(tf.int32, [None,None])
tf_text_lens = tf.placeholder(tf.int32,[None])
tf_summary_lens = tf.placeholder(tf.int32,[None])
tf_teacher_forcing = tf.placeholder(tf.bool)
tf_train = tf.placeholder(tf.bool)
tf_no_eval = tf.placeholder(tf.bool)
#----------------------------------------------

# ----激活函数-----
def gelu(x):
    return 0.5 * x * (1 + tf.nn.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

#---dropout---
def dropout(x,rate,training):
    return tf.cond(training,
                  lambda: tf.nn.dropout(x,rate=rate),
                  lambda:x)


#---层归一化---
def layerNorm(inputs, dim, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE, dtype=tf.float32):
        scale = tf.get_variable("scale", shape=[1, 1, dim],
                                dtype=tf.float32,
                                initializer=tf.ones_initializer())

        shift = tf.get_variable("shift", shape=[1, 1, dim],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

    mean, var = tf.nn.moments(inputs, [-1], keep_dims=True)

    epsilon = 1e-9

    LN = tf.multiply((scale / tf.sqrt(var + epsilon)), (inputs - mean)) + shift

    return LN

#---位置编码---

def spatial_encoding(D):
    global max_pos_len

    S = max_pos_len

    pe = np.zeros((2 * S + 1, D,), np.float32)

    for pos in range(-S, S + 1):
        for i in range(0, D):
            if i % 2 == 0:
                pe[pos + S, i] = math.sin(pos / (10000 ** (i / D)))
            else:
                pe[pos + S, i] = math.cos(pos / (10000 ** ((i - 1) / D)))

    return tf.constant(pe.reshape((2 * S + 1, D)), tf.float32)


PE = spatial_encoding(word_vec_dim)


#---mask---
def create_mask(Q, V, Q_mask, V_mask, neg_inf=-2.0 ** 32):
    global heads

    N = tf.shape(Q)[0]
    qS = tf.shape(Q)[1]
    vS = tf.shape(V)[1]

    y = tf.zeros([N, qS, vS], tf.float32)
    x = tf.cast(tf.fill([N, qS, vS], neg_inf), tf.float32)

    binary_mask = tf.reshape(V_mask, [N, 1, vS])
    binary_mask = tf.tile(binary_mask, [1, qS, 1])
    binary_mask = binary_mask * Q_mask

    mask = tf.where(tf.equal(binary_mask, tf.constant(0, tf.float32)),
                    x=x,
                    y=y)

    mask = tf.reshape(mask, [1, N, qS, vS])
    mask = tf.tile(mask, [heads, 1, 1, 1])
    mask = tf.reshape(mask, [heads * N, qS, vS])

    return mask

# 生成相对嵌入
def generate_relative_embd(qS, vS, embeddings):
    global max_pos_len

    S = tf.maximum(qS, vS)

    range_vec = tf.reshape(tf.range(S), [1, S])
    range_mat = tf.tile(range_vec, [S, 1])

    relative_pos_mat = range_mat - tf.transpose(range_mat)
    relative_pos_mat = relative_pos_mat[0:qS, 0:vS]

    relative_pos_mat_shifted = relative_pos_mat + max_pos_len

    RE = tf.nn.embedding_lookup(embeddings, relative_pos_mat_shifted)

    return RE

#构造模型
logits, predictions = encoder_decoder(tf_texts,tf_summaries,
                                      tf_text_lens,tf_summary_lens,
                                      tf_train,
                                      tf_no_eval)

#
trainables = tf.trainable_variables()
beta=1e-7

regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var in trainables])

pad_mask = tf.sequence_mask(tf_summary_lens, maxlen=tf.shape(tf_summaries)[1], dtype=tf.float32)

cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf_summaries)
cost = tf.multiply(cost,pad_mask)
cost = tf.reduce_mean(cost) + beta*regularization

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.98,epsilon=1e-9).minimize(cost)
