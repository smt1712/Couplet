import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# tf.reset_default_graph()

class Config(object):
    """RNN配置参数"""
    file_name = 'lstm_c'  #保存模型文件

    embedding_dim = 100      # 词向量维度
    seq_length = 30        # 序列长度
    # num_classes = 2        # 类别数
    vocab_size = 5000       # 词汇表达小


    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    # rnn = 'gru'             # lstm 或 gru
    share_emb_and_softmax = False  # 是否共享词向量层和sorfmax层的参数。（共享能减少参数且能提高模型效果）

    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 32  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config):
        self.config = config

        # 待输入的数据
        # 输入的序列[,30]
        self.en_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='encode_input')
        self.en_length = tf.placeholder(tf.int32, [None], name='ec_length')

        #placeholder 插入一个张量的占位符,这个张量将一直被提供.其值必须使用 feed_dict 可选参数来进行 session . run()
        self.zh_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='decode_input')
        self.zh_length = tf.placeholder(tf.int32, [None], name='zh_length')
        self.zh_seqs_label = tf.placeholder(tf.int32, [None, self.config.seq_length], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(3, dtype=tf.float32, trainable=False, name="global_loss")


        # seq2seq模型
        self.seq2seq()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.compat.v1.global_variables_initializer())
#    tf.compat.v1.global_variables_initializer  tf.global_variables_initializer
    def seq2seq(self):
        """seq2seq模型"""

        # 词嵌入层
        # 词向量维度100 vocab_size5000 创建[5000,100]的变量"xx_emb"
        en_embedding = tf.get_variable('en_emb', [self.config.vocab_size, self.config.embedding_dim])
        zh_embedding = tf.get_variable('zh_emb', [self.config.vocab_size, self.config.embedding_dim])
        # 创建常量 [1,100]
        embedding_zero = tf.constant(0, dtype=tf.float32, shape=[1, self.config.embedding_dim])
        self.en_embedding = tf.concat([en_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
        self.zh_embedding = tf.concat([zh_embedding, embedding_zero], axis=0)  # 增加一行0向量，代表padding向量值
        # 在 embedding 张量列表(en_embedding[5000,100])中查找 ids 索引（en_seqs[,30]）.
        # 根据输入在tensor列表里选择几行 组成新的tensor 作为encoder的输入
        embed_en_seqs = tf.nn.embedding_lookup(self.en_embedding, self.en_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
        embed_zh_seqs = tf.nn.embedding_lookup(self.zh_embedding, self.zh_seqs)

        # 在词嵌入上进行dropout 防止过度拟合的正则化
        embed_en_seqs = tf.nn.dropout(embed_en_seqs, keep_prob=self.keep_prob)
        embed_zh_seqs = tf.nn.dropout(embed_zh_seqs, keep_prob=self.keep_prob)

        # 定义rnn网络
        def get_en_cell(hidden_dim):
            # 创建单个lstm lstm(忘记控制 ，输出控制 ，输入控制)
            #参数：神经元数量，忘记门的参数
            enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
            return enc_base_cell

        # enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim)  # 注意将上面函数替换为这行，下面会报错

        with tf.variable_scope("encoder"):
            #定义多层RNN 2层隐藏层 每层128个神经元
            self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
                [get_en_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            # 通过dynamic_rnn对cell展开时间维度
            # 创建cell指定的神经网络，按inputs完全动态展开
            # time_major是false inputs就是一个[batch_size, max_time, ...]的Tensor....(不详细)
            # sequence_length 大小为batch_size的tensor ,为了正确性
            #enc_output encoder部分输出的语义向量 state是一个tensor
            enc_output, self.enc_state = tf.nn.dynamic_rnn(cell=self.enc_cell,
                                                           inputs=embed_en_seqs,
                                                           sequence_length=self.en_length,
                                                           # initial_state=self.initial_state1,  # 可有可无，自动为0状态
                                                           time_major=False,
                                                           dtype=tf.float32)

        with tf.variable_scope("decoder"):
            # decoder的多层rnn神经网络
            # initial_state是encoder的输出状态
            self.dec_cell = tf.nn.rnn_cell.MultiRNNCell([get_en_cell(self.config.hidden_dim) for _ in range(self.config.num_layers)])
            # 通过dynamic_rnn对cell展开时间维度n
            dec_output, self.dec_state= tf.nn.dynamic_rnn(self.dec_cell,
                                                            inputs=embed_zh_seqs,
                                                            sequence_length=self.zh_length,
                                                            initial_state=self.enc_state,  # 编码层的输出来初始化解码层的隐层状态
                                                            time_major=False,
                                                            dtype=tf.float32)

        with tf.name_scope("sorfmax_weights"):
            #q 没有共享
            if self.config.share_emb_and_softmax:
                self.softmax_weight = tf.transpose(self.zh_embedding)
                #q tf.transpose返回一个转置的tensor
            else:
                self.softmax_weight = tf.get_variable("weight",[self.config.hidden_dim, self.config.vocab_size+1])  #+1是因为对未知的可能输出
            self.softmax_bias = tf.get_variable("bias",[self.config.vocab_size+1])
            # 权重矩阵[128,5001]

        with tf.name_scope("loss"):
            # reshape():给定tensor,这个操作返回一个张量,它与带有形状shape的tensor具有相同的值.
            # out_put: [-1,128]
            out_put = tf.reshape(dec_output, [-1, self.config.hidden_dim])
            # 矩阵乘法 将矩阵 a 乘以矩阵 b,生成a * b
            # softmax_weight:[128,5001] softmax_bias: [5001] output[x,128]
            #logits [x,5001]
            # logits：输出层的输出，softmax的输入 softmax是对logits做归一化处理
            logits = tf.matmul(out_put, self.softmax_weight) + self.softmax_bias

            #q 计算logits和labels之间的稀疏softmax交叉熵。返回一个Tensor，与labels形状相同、和logits具有相同的类型。
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.zh_seqs_label,[-1]), logits=logits)

            # 计算平均损失时，需要将填充位置权重设置为0，以免无效位置预测干扰模型训练
            # tf.shape():返回张量形状
            #  sequence_mask()返回mask张量(只有true,false).用于数据填充 数据长短不一 与损失函数结果进行对照相乘去掉无用损失值。
            label_weights = tf.sequence_mask(self.zh_length, maxlen=tf.shape(self.zh_seqs_label)[1], dtype=tf.float32)
            label_weights = tf.reshape(label_weights, [-1])
            # reduce_mean() 计算张量的各个维度上的元素的平均值.
            self.mean_loss = tf.reduce_mean(loss*label_weights)

        with tf.name_scope("accuracy"):
            #算logits
            out_put = tf.reshape(dec_output, [-1, self.config.hidden_dim])
            logits = tf.matmul(out_put, self.softmax_weight) + self.softmax_bias
            # labels
            labels=tf.reshape(self.zh_seqs_label,[-1])

            # correct_prediction = tf.equal(logits, labels)
            # top_k() 当k=1时等价tf.equal()，但是equal()里两个参数的shape必须一样。
            correct_prediction = tf.nn.in_top_k(logits, labels, 1)

            #tf.reduce_mean()平均值
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        with tf.name_scope("pres"):
            # axis=1的时候，将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。
            self.output_id = tf.argmax(logits, axis=1, output_type=tf.int32)[0]

        with tf.name_scope("optimize"):
            #Adam算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率
            #自适应矩估计
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)
    
    
    def test(self, test_g, model_path, zt):
        batch_en, batch_en_len = test_g
        feed = {self.en_seqs: batch_en,
                self.en_length: batch_en_len,
                self.keep_prob:1.0}
        enc_state = self.session.run(self.enc_state, feed_dict=feed)

        output_ids = []
        dec_state = enc_state
        dec_input, dec_len = zt.text_de_to_arr(['<s>',])  # decoder层初始输入
        dec_input = np.array([dec_input[:-1], ])
        dec_len = np.array([dec_len, ])
        for i in range(self.config.seq_length):  # 最多输出50长度，防止极端情况下死循环
            feed = {self.enc_state: dec_state,
                    self.zh_seqs: dec_input,
                    self.zh_length: dec_len,
                    self.keep_prob: 1.0}
            dec_state, output_id= self.session.run([self.dec_state, self.output_id], feed_dict=feed)

            char = zt.int_to_word(output_id)
            if char == '</s>':
                break
            output_ids.append(output_id)

            arr = [output_id]+[len(zt.vocab)] * (self.config.seq_length - 1)
            dec_input = np.array([arr, ])
        return output_ids
   

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
