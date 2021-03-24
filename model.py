# -*- coding: utf-8 -*-
# @Time : 2021/3/23 13:24
# @Author : Jclian91
# @File : model.py
# @Place : Yangpu, Shanghai
# main architecture of R-BERT
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Lambda, Dense, Dropout, concatenate, Dot
from keras_bert import load_trained_model_from_checkpoint


# model structure of R-BERT
class RBERT(object):
    def __init__(self, config_path, checkpoint_path, maxlen, num_labels):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.maxlen = maxlen
        self.num_labels = num_labels

    def create_model(self):
        # BERT model
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)
        for layer in bert_model.layers:
            layer.trainable = True
        x1_in = Input(shape=(self.maxlen,))
        x2_in = Input(shape=(self.maxlen,))
        bert_layer = bert_model([x1_in, x2_in])

        # get three vectors
        cls_layer = Lambda(lambda x: x[:, 0])(bert_layer)    # 取出[CLS]对应的向量
        e1_mask = Input(shape=(self.maxlen,))
        e2_mask = Input(shape=(self.maxlen,))
        e1_layer = self.entity_average(bert_layer, e1_mask)  # 取出实体1对应的向量
        e2_layer = self.entity_average(bert_layer, e2_mask)  # 取出实体2对应的向量

        # dropout -> linear -> concatenate
        output_dim = cls_layer.shape[-1].value
        cls_fc_layer = self.crate_fc_layer(cls_layer, output_dim, dropout_rate=0.1)
        e1_fc_layer = self.crate_fc_layer(e1_layer, output_dim, dropout_rate=0.1)
        e2_fc_layer = self.crate_fc_layer(e2_layer, output_dim, dropout_rate=0.1)
        concat_layer = concatenate([cls_fc_layer, e1_fc_layer, e2_fc_layer], axis=-1)

        # FC layer for classification
        fc_layer = Dense(100, activation="relu")(concat_layer)
        output = Dense(self.num_labels, activation="softmax")(fc_layer)
        model = Model([x1_in, x2_in, e1_mask, e2_mask], output)
        model.summary()
        return model

    @staticmethod
    def crate_fc_layer(input_layer, output_dim, dropout_rate=0.0, activation_func="tanh"):
        dropout_layer = Dropout(rate=dropout_rate)(input_layer)
        linear_layer = Dense(output_dim, activation=activation_func)(dropout_layer)
        return linear_layer

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: BERT hidden output
        :param e_mask:
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]/num_of_ones
        :return: entity average layer
        """
        avg_layer = Dot(axes=1)([e_mask, hidden_output])
        return avg_layer


if __name__ == '__main__':
    model_config = "./chinese_L-12_H-768_A-12/bert_config.json"
    model_checkpoint = "./chinese_L-12_H-768_A-12/bert_model.ckpt"
    model = RBERT(model_config, model_checkpoint, 128, 14).create_model()
    plot_model(model, to_file="model_structure.png")