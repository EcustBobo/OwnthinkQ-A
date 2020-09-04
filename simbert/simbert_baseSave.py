#! -*- coding: utf-8 -*-
# SimBERT base 基本例子
import os
import numpy as np
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
# from tensorflow.python.keras.backend import set_session
# sess = tf.Session()
# graph = tf.get_default_graph()

# global graph
# graph = tf.get_default_graph()


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 分配50%
# config.gpu_options.allow_growth = True
# session=tf.Session(config=config)
# KTF.set_session(session)

maxlen = 32

dir_path = os.getcwd()
# print('目前路径：',dir_path)

# bert配置
config_path = dir_path + \
    '/src/qa/simbert/chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = dir_path + \
    '/src/qa/simbert/chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = dir_path + '/src/qa/simbert/chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = tokenizer.encode(text, max_length=maxlen)
        output_ids = self.random_sample(
            [token_ids, segment_ids], n, topk)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


# synonyms_generator = SynonymsGenerator(start_id=None,
#                                        end_id=tokenizer._token_end_id,
#                                        maxlen=maxlen)


class generateSimSentence(object):
    def __init__(self,):
        self.me = SynonymsGenerator(start_id=None,
                                    end_id=tokenizer._token_end_id,
                                    maxlen=maxlen)
        # set_session(sess)
        # self.encoder = keras.models.Model(
        #     bert.model.inputs, bert.model.outputs[0])

    def gen_synonyms(self, text, n=10, k=5):
        """"含义： 产生sent的n个相似句，然后返回最相似的k个。
        做法：用seq2seq生成，并用encoder算相似度并排序。
        """
        r = self.me.generate(text, n)
        r = [i for i in set(r) if i != text]
        r = [text] + r
        X, S = [], []
        for t in r:
            x, s = tokenizer.encode(t)
            X.append(x)
            S.append(s)
        X = sequence_padding(X)
        S = sequence_padding(S)
        # global sess
        # global graph
        # with graph.as_default():
        #     set_session(sess)
        Z = encoder.predict([X, S])
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        argsort = np.dot(Z[1:], -Z[0]).argsort()
        return [r[i + 1] for i in argsort[:k]]

    def gen_sim_value(self, text_01, text_02):
        X, S = [], []
        x_1, s_1 = tokenizer.encode(text_01)
        x_2, s_2 = tokenizer.encode(text_02)
        X.append(x_1)
        X.append(x_2)
        S.append(s_1)
        S.append(s_2)
        X = sequence_padding(X)
        S = sequence_padding(S)
        # # global sess
        # # global graph
        # with graph.as_default():
        #     #     set_session(sess)
        Z = encoder.predict([X, S])
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        return '%.2f' % (np.dot(Z[1], Z[0]))

    def gen_all_sim_value(self, ques: list):
        X, S = [], []
        for que in ques:
            x, s = tokenizer.encode(que)
            X.append(x)
            S.append(s)
        X = sequence_padding(X)
        S = sequence_padding(S)
        # global sess
        # global graph
        # with graph.as_default():
        #     set_session(sess)
        Z = encoder.predict([X, S])
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
        res = np.dot(Z[1:], Z[0])
        res = list(res)
        return res
