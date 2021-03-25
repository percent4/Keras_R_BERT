# -*- coding: utf-8 -*-
# @Time : 2021/3/24 21:08
# @Author : Jclian91
# @File : config.py
# @Place : Yangpu, Shanghai

# dataset
DATA_DIR = "./data/people_relation"
TRAIN_FILE_PATH = "{}/train.csv".format(DATA_DIR)
TEST_FILE_PATH = "{}/test.csv".format(DATA_DIR)

# model
BERT_MODEL_DIR = "chinese-RoBERTa-wwm-ext"
BERT_CONFIG_PATH = "{}/bert_config.json".format(BERT_MODEL_DIR)
BERT_CHECKPOINT_PATH = "{}/bert_model.ckpt".format(BERT_MODEL_DIR)
BERT_VOCAT_PATH = "{}/vocab.txt".format(BERT_MODEL_DIR)
MAX_LEN = 200
EPOCH = 10
BATCH_SIZE = 16
