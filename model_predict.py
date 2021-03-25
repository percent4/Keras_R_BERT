# -*- coding: utf-8 -*-
# @Time : 2021/3/24 20:40
# @Author : Jclian91
# @File : model_predict.py
# @Place : Yangpu, Shanghai
import os
import json
import numpy as np
from keras.models import load_model
from keras_bert import get_custom_objects

from model_train import token_dict, OurTokenizer
from config import *

# 加载训练好的模型
model = load_model("./models/people_relation-05-0.8546.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open(os.path.join(DATA_DIR, "label.json"), "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())


def predict_single_sample(text):
    # 利用BERT进行tokenize
    text = text.replace("$", "").replace("#", "")
    text = text.replace("<e1>", "$").replace("</e1>", "$").replace("<e2>", "#").replace("</e2>", "#")
    X1, X2 = tokenizer.encode(first=text, max_len=MAX_LEN)
    X1[X1.index(102)] = 0  # 去掉[SEP]
    # 寻找$,#的下标
    special_index_1 = [i for i in range(len(X1)) if X1[i] == 109]   # index of $
    special_index_2 = [i for i in range(len(X1)) if X1[i] == 108]   # index of #
    mask1 = [0] * MAX_LEN
    start1, end1 = special_index_1
    for i in range(start1+1, end1):
        mask1[i] = 1 / (end1 - start1 - 1)
    mask2 = [0] * MAX_LEN
    start2, end2 = special_index_2
    for i in range(start2+1, end2):
        mask2[i] = 1 / (end2 - start2 - 1)

    # 模型预测并输出预测结果
    predicted = model.predict([[X1], [X2], [mask1], [mask2]])
    y = np.argmax(predicted[0])
    return label_dict[str(y)]


if __name__ == '__main__':
    # 预测示例语句
    predict_text = "程砚秋与<e1>果素瑛</e1>生有三子一女，即<e2>程永光</e2>、程永源、程永江和程慧贞。"
    predict_label = predict_single_sample(predict_text)
    print("原句: {}\n预测关系: {}".format(predict_text, predict_label))


